# Quantized Traffic Sign Recognition on PYNQ-Z2

This project trains a small CNN for GTSRB traffic sign recognition. The current
training code uses PyTorch Lightning. The next step is quantization with
Brevitas and FPGA deployment with FINN on a PYNQ-Z2 board.

## Current features

- 43-class GTSRB classification
- 48 x 48 RGB input
- Sequence-aware train and validation split
- Data augmentation for scale, blur, color, rotation, and translation
- Class weights and targeted oversampling
- TensorBoard logging
- Accuracy and macro F1 checkpoints
- Separate checkpoint test script

## Project structure

```text
.
|-- README.md
|-- docs/
|   `-- learning-notes.md
`-- src/
    |-- gtsrb_utils.py   # Data split, transforms, loaders, and class weights
    |-- tiny_vgg.py      # Lightning model
    |-- train.py         # Training and checkpoint creation
    `-- test.py          # Test one saved checkpoint without training
```

Data, checkpoints, TensorBoard logs, and FPGA build files are not committed.

## Setup

Install the required Python packages in a virtual environment:

```bash
pip install torch torchvision lightning torchmetrics tensorboard numpy
```

## Training

Run from the repository root:

```bash
python src/train.py
```

The dataset is downloaded into `data/`. TensorBoard logs are written to `logs/`
and model checkpoints are written to `checkpoints/`.

Open TensorBoard with:

```bash
tensorboard --logdir logs
```

## Testing

Set `checkpoint_path` in `src/test.py`, then run:

```bash
python src/test.py
```

The test script loads the selected checkpoint and does not run training.

## FPGA plan

1. Keep the floating-point model as the baseline.
2. Build W8A8 and W4A8 models with Brevitas.
3. Export the quantized model to QONNX.
4. Build and simulate the accelerator with FINN.
5. Deploy it on PYNQ-Z2.
6. Measure accuracy, macro F1, LUT, FF, BRAM, DSP, latency, FPS, and power.

## Status

The floating-point training pipeline is complete. Quantization and FPGA
deployment are future work.
