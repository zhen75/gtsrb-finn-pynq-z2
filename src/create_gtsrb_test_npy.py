#!/usr/bin/env python3
"""Export the GTSRB test split as E2E FINN input and label NPY files."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from quant_vgg import QuantVgg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS = {
    "w1a4": (1, 4, PROJECT_ROOT / "qat_experiments/experiment_24/w1a4_qat.pth"),
    "w2a4": (2, 4, PROJECT_ROOT / "qat_experiments/experiment_21/w2a4_qat.pth"),
    "w4a4": (4, 4, PROJECT_ROOT / "qat_experiments/experiment_18/w4a4_qat.pth"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create UINT8 NHWC data and label NPY files from GTSRB test"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="existing torchvision GTSRB root (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_npy"),
        help="output directory (default: test_npy)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device used to generate software predictions",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download GTSRB only when it is not already present",
    )
    return parser.parse_args()


def load_models(device):
    models = {}

    for name, (weight_bit, activate_bit, weight_path) in MODEL_CONFIGS.items():
        if not weight_path.is_file():
            raise FileNotFoundError(f"{name} weight file not found: {weight_path}")

        model = QuantVgg(
            weight_bit=weight_bit,
            activate_bit=activate_bit,
            dropout=0.2,
            class_weights=torch.ones(43),
        )
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)
        models[name] = model

        print(f"Loaded {name}: {weight_path}")

    return models


def main():
    args = parse_args()
    device = torch.device(args.device)
    models = load_models(device)

    mean = torch.tensor([0.3399, 0.3121, 0.3214], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2760, 0.2625, 0.2690], device=device).view(1, 3, 1, 1)

    transform = transforms.Compose(
        [
            transforms.Resize((48, 48)),
            transforms.PILToTensor(),
        ]
    )

    dataset = GTSRB(
        root=str(args.data_root),
        split="test",
        download=args.download,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    sample_count = len(dataset)
    inputs = np.empty((sample_count, 48, 48, 3), dtype=np.uint8)
    labels = np.empty((sample_count,), dtype=np.int64)
    predictions = {
        name: np.empty((sample_count,), dtype=np.int64) for name in models
    }

    offset = 0
    with torch.inference_mode():
        for images, batch_labels in loader:
            batch_count = images.shape[0]

            if images.dtype.is_floating_point:
                raise RuntimeError("PILToTensor unexpectedly returned floating-point data")
            if tuple(images.shape[1:]) != (3, 48, 48):
                raise RuntimeError(f"unexpected image batch shape: {tuple(images.shape)}")

            batch_inputs = images.permute(0, 2, 3, 1).contiguous().numpy()
            inputs[offset : offset + batch_count] = batch_inputs
            labels[offset : offset + batch_count] = batch_labels.numpy()

            normalized = images.to(device=device, dtype=torch.float32) / 255.0
            normalized = (normalized - mean) / std
            for name, model in models.items():
                logits = model(normalized)
                predictions[name][offset : offset + batch_count] = (
                    logits.argmax(dim=1).cpu().numpy()
                )

            offset += batch_count

    if offset != sample_count:
        raise RuntimeError(f"exported {offset} samples, expected {sample_count}")
    if labels.min() < 0 or labels.max() >= 43:
        raise RuntimeError(
            f"unexpected GTSRB label range: {labels.min()} to {labels.max()}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = args.output_dir / "gtsrb_test_inputs_uint8_nhwc.npy"
    labels_path = args.output_dir / "gtsrb_test_labels.npy"

    np.save(inputs_path, inputs)
    np.save(labels_path, labels)
    prediction_paths = {}
    for name, values in predictions.items():
        prediction_path = args.output_dir / f"gtsrb_test_expected_{name}.npy"
        np.save(prediction_path, values)
        prediction_paths[name] = prediction_path

    saved_inputs = np.load(inputs_path, mmap_mode="r")
    saved_labels = np.load(labels_path, mmap_mode="r")

    if saved_inputs.shape != (sample_count, 48, 48, 3):
        raise RuntimeError(f"saved input shape is incorrect: {saved_inputs.shape}")
    if saved_inputs.dtype != np.uint8:
        raise RuntimeError(f"saved input dtype is incorrect: {saved_inputs.dtype}")
    if saved_labels.shape != (sample_count,):
        raise RuntimeError(f"saved label shape is incorrect: {saved_labels.shape}")
    if saved_labels.dtype != np.int64:
        raise RuntimeError(f"saved label dtype is incorrect: {saved_labels.dtype}")

    print(f"Saved inputs: {inputs_path}")
    print(f"  shape: {saved_inputs.shape}")
    print(f"  dtype: {saved_inputs.dtype}")
    print(f"  range: {saved_inputs.min()} to {saved_inputs.max()}")
    print(f"Saved labels: {labels_path}")
    print(f"  shape: {saved_labels.shape}")
    print(f"  dtype: {saved_labels.dtype}")
    print(f"  range: {saved_labels.min()} to {saved_labels.max()}")

    for name, prediction_path in prediction_paths.items():
        saved_predictions = np.load(prediction_path, mmap_mode="r")
        if saved_predictions.shape != (sample_count,):
            raise RuntimeError(
                f"saved {name} prediction shape is incorrect: {saved_predictions.shape}"
            )
        if saved_predictions.dtype != np.int64:
            raise RuntimeError(
                f"saved {name} prediction dtype is incorrect: {saved_predictions.dtype}"
            )
        if saved_predictions.min() < 0 or saved_predictions.max() >= 43:
            raise RuntimeError(
                f"saved {name} prediction range is incorrect: "
                f"{saved_predictions.min()} to {saved_predictions.max()}"
            )

        accuracy = np.mean(saved_predictions == saved_labels)
        print(f"Saved {name} expected predictions: {prediction_path}")
        print(f"  shape: {saved_predictions.shape}")
        print(f"  dtype: {saved_predictions.dtype}")
        print(f"  test accuracy: {accuracy:.6f}")


if __name__ == "__main__":
    main()
