#!/usr/bin/env python3
"""Run a deterministic GTSRB smoke test on a FINN PYNQ deployment."""

import argparse
import importlib.util
import os
import pwd
import sys
import time
from pathlib import Path

import numpy as np


def user_home():
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        return Path(pwd.getpwnam(sudo_user).pw_dir)
    return Path.home()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a deterministic smoke test on a FINN PYNQ bitstream"
    )
    parser.add_argument(
        "--deploy-dir",
        required=True,
        type=Path,
        help="FINN deployment directory containing bitfile/ and driver/",
    )
    parser.add_argument(
        "--inputs",
        required=True,
        type=Path,
        help="full GTSRB UINT8 NHWC test input NPY",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="full GTSRB test label NPY",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        help="optional full software prediction NPY for agreement checking",
    )
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=user_home() / "pynq_smoke_results",
    )
    return parser.parse_args()


def load_class_vector(path, sample_count, name):
    values = np.load(str(path), allow_pickle=False)

    if values.ndim >= 2 and values.shape[-1] == 43:
        values = np.argmax(values, axis=-1)

    values = np.asarray(values).reshape(-1)
    if len(values) != sample_count:
        raise ValueError(
            "{} contains {} samples, expected {}".format(
                name, len(values), sample_count
            )
        )
    if not np.equal(values, np.round(values)).all():
        raise ValueError("{} must contain integer class indices".format(name))

    values = values.astype(np.int64)
    if values.min() < 0 or values.max() > 42:
        raise ValueError("{} class indices must be in 0..42".format(name))
    return values


def select_indices(labels, sample_count, seed):
    total = len(labels)
    if sample_count <= 0 or sample_count > total:
        raise ValueError(
            "sample count must be between 1 and {}, got {}".format(
                total, sample_count
            )
        )

    rng = np.random.RandomState(seed)
    classes = np.unique(labels).copy()
    rng.shuffle(classes)
    selected = []

    for class_id in classes[:sample_count]:
        candidates = np.flatnonzero(labels == class_id)
        selected.append(int(rng.choice(candidates)))

    if len(selected) < sample_count:
        remaining = np.setdiff1d(
            np.arange(total), np.asarray(selected), assume_unique=False
        )
        extra = rng.choice(
            remaining,
            size=sample_count - len(selected),
            replace=False,
        )
        selected.extend(int(index) for index in extra)

    return np.asarray(selected, dtype=np.int64)


def load_generated_driver(driver_path):
    driver_dir = driver_path.parent.resolve()
    sys.path.insert(0, str(driver_dir))

    specification = importlib.util.spec_from_file_location(
        "finn_generated_driver", str(driver_path)
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load generated FINN driver: {}".format(driver_path))

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def restore_output_ownership(paths):
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if sudo_uid is None or sudo_gid is None:
        return

    uid = int(sudo_uid)
    gid = int(sudo_gid)
    for path in paths:
        os.chown(str(path), uid, gid)


def main():
    args = parse_args()
    deploy_dir = args.deploy_dir.resolve()
    bitfile = deploy_dir / "bitfile" / "finn-accel.bit"
    hwh = deploy_dir / "bitfile" / "finn-accel.hwh"
    driver_path = deploy_dir / "driver" / "driver.py"
    runtime_weights = deploy_dir / "driver" / "runtime_weights"

    for required in (bitfile, hwh, driver_path):
        if not required.is_file():
            raise FileNotFoundError("required deployment file not found: {}".format(required))

    inputs = np.load(str(args.inputs), mmap_mode="r", allow_pickle=False)
    if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (48, 48, 3):
        raise ValueError(
            "inputs must have shape (N, 48, 48, 3), got {}".format(inputs.shape)
        )
    if inputs.dtype != np.uint8:
        raise ValueError("inputs must be UINT8, got {}".format(inputs.dtype))

    total = len(inputs)
    labels = load_class_vector(args.labels, total, "labels")
    expected = None
    if args.expected is not None:
        expected = load_class_vector(args.expected, total, "expected predictions")

    indices = select_indices(labels, args.sample_count, args.seed)
    smoke_inputs = np.asarray(inputs[indices], dtype=np.uint8)
    smoke_labels = labels[indices]
    smoke_expected = expected[indices] if expected is not None else None

    generated_driver = load_generated_driver(driver_path)

    from driver_base import FINNExampleOverlay
    from pynq.pl_server.device import Device

    device = Device.devices[args.device]
    accelerator = FINNExampleOverlay(
        bitfile_name=str(bitfile),
        platform="zynq-iodma",
        io_shape_dict=generated_driver.io_shape_dict,
        batch_size=args.sample_count,
        runtime_weight_dir=str(runtime_weights),
        device=device,
    )

    start = time.perf_counter()
    output = accelerator.execute(smoke_inputs)
    elapsed = time.perf_counter() - start

    output = np.asarray(output)
    if output.shape != (args.sample_count, 1):
        raise RuntimeError(
            "FPGA output must have shape ({}, 1), got {}".format(
                args.sample_count, output.shape
            )
        )
    if not np.isfinite(output).all():
        raise RuntimeError("FPGA output contains NaN or infinity")
    if not np.equal(output, np.round(output)).all():
        raise RuntimeError("FPGA output contains non-integer class values")

    predictions = output.reshape(-1).astype(np.int64)
    if predictions.min() < 0 or predictions.max() > 42:
        raise RuntimeError(
            "FPGA class indices are outside 0..42: {} to {}".format(
                predictions.min(), predictions.max()
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [args.output_dir]
    arrays = {
        "smoke_indices.npy": indices,
        "smoke_inputs_uint8_nhwc.npy": smoke_inputs,
        "smoke_labels.npy": smoke_labels,
        "smoke_fpga_predictions.npy": predictions,
    }
    if smoke_expected is not None:
        arrays["smoke_software_predictions.npy"] = smoke_expected

    for filename, array in arrays.items():
        path = args.output_dir / filename
        np.save(str(path), array)
        output_paths.append(path)

    restore_output_ownership(output_paths)

    accuracy = float(np.mean(predictions == smoke_labels))
    throughput = args.sample_count / elapsed

    print("Smoke indices:      ", indices)
    print("True labels:        ", smoke_labels)
    print("FPGA predictions:   ", predictions)
    print("FPGA smoke accuracy: {:.4f}".format(accuracy))
    print("Batch runtime:       {:.3f} ms".format(elapsed * 1000.0))
    print("Observed throughput: {:.3f} images/s".format(throughput))
    print("Saved results:       {}".format(args.output_dir))

    if smoke_expected is not None:
        agreement = float(np.mean(predictions == smoke_expected))
        print("Software predictions:", smoke_expected)
        print("Software-FPGA agreement: {:.4f}".format(agreement))
        if agreement != 1.0:
            mismatch = indices[predictions != smoke_expected]
            raise RuntimeError(
                "software-FPGA mismatch at full-test indices {}".format(
                    mismatch.tolist()
                )
            )
        print("Functional smoke test: PASS")
    else:
        print("Runtime smoke test: PASS")
        print(
            "Software-FPGA equivalence was not checked because --expected was not provided."
        )


if __name__ == "__main__":
    main()
