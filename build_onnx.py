#!/usr/bin/env python3
"""Build the selected end-to-end QONNX model for PYNQ-Z2 with FINN."""

import argparse
import json
import math
from functools import reduce
from pathlib import Path

import numpy as np

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
    default_build_dataflow_steps,
)
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import RemoveUnusedTensors


SCRIPT_DIR = Path(__file__).resolve().parent

MODELS = {
    "w1a4": SCRIPT_DIR / "output_e2e_onnx" / "e2e_w1a4_qat.onnx",
    "w2a4": SCRIPT_DIR / "output_e2e_onnx" / "e2e_w2a4_qat.onnx",
    "w4a4": SCRIPT_DIR / "output_e2e_onnx" / "e2e_w4a4_qat.onnx",
}


def _initializer_input(model, node):
    values = []
    data_inputs = []

    for input_name in node.input:
        value = model.get_initializer(input_name)
        if value is None:
            data_inputs.append(input_name)
        else:
            values.append((input_name, np.asarray(value, dtype=np.float32)))

    if len(data_inputs) != 1 or len(values) != 1:
        raise RuntimeError(
            f"Expected one data input and one initializer for {node.name}"
        )

    return data_inputs[0], values[0][0], values[0][1]


def _common_float32_factor(arrays):
    """Return the exact positive greatest common factor of float32 values."""
    ratios = []

    for array in arrays:
        for value in np.asarray(array, dtype=np.float32).reshape(-1):
            value = float(value)
            if value != 0.0:
                ratios.append(value.as_integer_ratio())

    if not ratios:
        return 1.0

    common_denominator = max(denominator for _, denominator in ratios)
    numerators = [
        numerator * (common_denominator // denominator)
        for numerator, denominator in ratios
    ]
    common_numerator = reduce(math.gcd, (abs(value) for value in numerators))

    return common_numerator / common_denominator


def _to_exact_integer(array, factor, name):
    original = np.asarray(array, dtype=np.float32)
    integer = np.rint(original.astype(np.float64) / factor).astype(np.int64)

    reconstructed = integer.astype(np.float64) * factor
    if not np.array_equal(reconstructed, original.astype(np.float64)):
        raise RuntimeError(f"{name} cannot be represented by one exact factor")

    stored = integer.astype(np.float32)
    if not np.array_equal(stored.astype(np.int64), integer):
        raise RuntimeError(f"{name} integer values are not exactly representable in float32")

    return integer, stored


def _remove_node_and_reconnect(model, node, replacement_tensor):
    old_output = node.output[0]

    for consumer in model.find_consumers(old_output):
        for index, input_name in enumerate(consumer.input):
            if input_name == old_output:
                consumer.input[index] = replacement_tensor

    model.graph.node.remove(node)


def step_integerize_output_affine(model, cfg):
    """Make the affine operation before TopK exactly integer-valued.

    A common positive factor is removed because it cannot change TopK order.
    The remaining per-channel integer Mul/Add operations are converted to
    FINN ChannelwiseOp nodes.
    """
    topk_nodes = model.get_nodes_by_op_type("TopK")
    if len(topk_nodes) != 1:
        raise RuntimeError(f"Expected one TopK node, found {len(topk_nodes)}")

    topk = topk_nodes[0]
    tail = model.find_producer(topk.input[0])
    add_node = None

    if tail is not None and tail.op_type == "Add":
        add_node = tail
        add_data, add_name, bias = _initializer_input(model, add_node)
        tail = model.find_producer(add_data)
    else:
        add_name = None
        bias = None

    if tail is None or tail.op_type != "Mul":
        actual = "None" if tail is None else tail.op_type
        raise RuntimeError(f"Expected Mul before TopK affine, found {actual}")

    mul_node = tail
    mul_data, mul_name, scale = _initializer_input(model, mul_node)

    factor_inputs = [scale]
    if bias is not None:
        factor_inputs.append(bias)
    factor = _common_float32_factor(factor_inputs)

    if not np.isfinite(factor) or factor <= 0.0:
        raise RuntimeError(f"Invalid common output factor: {factor}")

    scale_integer, scale_stored = _to_exact_integer(scale, factor, "scale")
    model.set_initializer(mul_name, scale_stored)

    bias_integer = None
    if bias is not None:
        bias_integer, bias_stored = _to_exact_integer(bias, factor, "bias")
        model.set_initializer(add_name, bias_stored)

    if np.all(scale_integer == 1):
        _remove_node_and_reconnect(model, mul_node, mul_data)
        print("  removed identity integer Mul")
    else:
        print("  retained integer channel-wise Mul")

    if add_node is not None and np.all(bias_integer == 0):
        add_data, _, _ = _initializer_input(model, add_node)
        _remove_node_and_reconnect(model, add_node, add_data)
        add_node = None
        print("  removed zero integer Add")

    model = model.transform(RemoveUnusedTensors())

    before = len(model.get_nodes_by_op_type("ChannelwiseOp"))
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    created = len(model.get_nodes_by_op_type("ChannelwiseOp")) - before

    expected = int(not np.all(scale_integer == 1)) + int(add_node is not None)
    if created != expected:
        raise RuntimeError(
            f"Expected {expected} new ChannelwiseOp nodes, created {created}"
        )

    print("Output affine conversion:")
    print(f"  common factor: {factor}")
    print(
        "  scale integer range: "
        f"{int(scale_integer.min())} to {int(scale_integer.max())}"
    )
    if bias_integer is not None:
        print(
            "  bias integer range: "
            f"{int(bias_integer.min())} to {int(bias_integer.max())}"
        )
    print(f"  created ChannelwiseOps: {created}")

    return model


def _smallest_legal_simd(matrix_width, minimum):
    for simd in range(max(1, minimum), matrix_width + 1):
        if matrix_width % simd == 0:
            return simd
    raise RuntimeError(f"No legal SIMD found for MW={matrix_width}")


def _update_folding_file(cfg, changes):
    folding_file = getattr(cfg, "folding_config_file", None)
    if folding_file is None:
        folding_file = Path(cfg.output_dir) / "auto_folding_config.json"
    else:
        folding_file = Path(folding_file)

    if not folding_file.is_file():
        raise RuntimeError(f"Folding configuration not found: {folding_file}")

    with folding_file.open("r", encoding="utf-8") as handle:
        folding = json.load(handle)

    for node_name, simd in changes.items():
        if node_name not in folding:
            raise RuntimeError(f"Node missing from folding configuration: {node_name}")
        folding[node_name]["SIMD"] = simd

    with folding_file.open("w", encoding="utf-8") as handle:
        json.dump(folding, handle, indent=2)
        handle.write("\n")


def step_ensure_legal_mvau_simd(model, cfg):
    """Raise too-small HLS MVAU SIMD values and persist them in folding JSON."""
    changes = {}

    for node in model.get_nodes_by_op_type("MVAU_hls"):
        instance = getCustomOp(node)
        matrix_width = instance.get_nodeattr("MW")
        current_simd = instance.get_nodeattr("SIMD")
        minimum_simd = math.ceil(matrix_width / 1024)

        if current_simd < minimum_simd:
            new_simd = _smallest_legal_simd(matrix_width, minimum_simd)
            instance.set_nodeattr("SIMD", new_simd)
            changes[node.name] = new_simd
            print(
                f"Adjusted {node.name}: SIMD {current_simd} -> {new_simd}, "
                f"MW={matrix_width}"
            )

    if changes:
        _update_folding_file(cfg, changes)
    else:
        print("All MVAU_hls SIMD values are already legal")

    return model


def make_build_steps():
    steps = list(default_build_dataflow_steps)
    steps.insert(steps.index("step_convert_to_hw"), step_integerize_output_affine)
    steps.insert(
        steps.index("step_apply_folding_config"),
        step_ensure_legal_mvau_simd,
    )
    return steps


def build_model(name, clock_ns, target_fps):
    model_path = MODELS[name]
    output_dir = SCRIPT_DIR / f"output_{name}_e2e_pynqz2"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("\n" + "=" * 70)
    print(f"Model: {name}")
    print(f"Input: {model_path}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    cfg = DataflowBuildConfig(
        output_dir=str(output_dir),
        synth_clk_period_ns=clock_ns,
        target_fps=target_fps,
        board="Pynq-Z2",
        shell_flow_type=ShellFlowType.VIVADO_ZYNQ,
        steps=make_build_steps(),
        generate_outputs=[
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.BITFILE,
            DataflowOutputType.PYNQ_DRIVER,
            DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
        auto_fifo_depths=True,
        split_large_fifos=True,
        save_intermediate_models=True,
        enable_build_pdb_debug=False,
        verbose=True,
    )

    result = build_dataflow_cfg(str(model_path), cfg)
    if result != 0:
        raise RuntimeError(f"FINN build failed for {name}")

    bitfile = output_dir / "bitfile" / "finn-accel.bit"
    driver = output_dir / "driver" / "driver.py"
    deployment = output_dir / "deploy"

    missing = [path for path in (bitfile, driver, deployment) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"Build returned success but outputs are missing: {missing_text}")

    print(f"Build completed successfully: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build W1A4, W2A4 or W4A4 end-to-end FINN accelerators"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["w1a4", "w2a4", "w4a4", "all"],
        help="model to build; 'all' builds the three models sequentially",
    )
    parser.add_argument(
        "--clock-ns",
        type=float,
        default=10.0,
        help="target clock period in nanoseconds (default: 10.0)",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=30,
        help="target folding throughput in frames per second (default: 30)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected = list(MODELS) if args.model == "all" else [args.model]

    for model_name in selected:
        build_model(model_name, args.clock_ns, args.target_fps)

    print("\nAll requested builds completed successfully.")
