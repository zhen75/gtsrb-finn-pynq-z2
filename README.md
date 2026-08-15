# Quantized Traffic Sign Recognition on PYNQ-Z2

This project trains a small CNN for GTSRB traffic sign recognition, quantizes it
with Brevitas, and deploys it with FINN on a PYNQ-Z2 board.

## Current features

- 43-class GTSRB classification
- 48 x 48 RGB input
- Sequence-aware train and validation split
- Data augmentation for scale, blur, color, rotation, and translation
- Class weights and targeted oversampling
- TensorBoard logging
- Accuracy and macro F1 checkpoints
- Separate checkpoint test script
- QAT experiments from W8A8 to binary-weight W1A4
- QONNX preprocessing export for NHWC UINT8 input
- FINN W4A4 bitstream build and PYNQ-Z2 smoke test

## QAT trade-off

The chart uses the weight-bit × activation-bit product as a simple quantization
cost proxy. Red points are non-dominated accuracy-cost configurations on the
official GTSRB test split.

![QAT accuracy and Pareto frontier](analysis/qat_accuracy_pareto.png)

All 16 configurations were trained for 60 epochs without early stopping under
one seed. The observed accuracy Pareto frontier is `W1A1`, `W1A2`, `W1A4`,
`W2A4`, `W4A4`, and `W4A8`; the practical high-accuracy frontier starts at
`W1A4`. The W1 experiments use binary per-tensor weights; W2/W4/W8 use
per-channel weight scaling.

## Project structure

```text
.
|-- README.md
|-- docs/
|   `-- learning-notes.md
`-- src/
    |-- gtsrb_utils.py   # Data split, transforms, loaders, and class weights
    |-- tiny_vgg.py      # Lightning model
    |-- quant_vgg.py     # Brevitas QAT model, including binary quantization
    |-- run_experiment.py # QAT/PTQ experiment, test, and QONNX export
    |-- preprocessor.py  # Export NHWC UINT8 preprocessing to QONNX
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

## Mixed-precision QAT

`QuantVgg` accepts one weight bit width per quantized weight layer and one
activation bit width per `QuantReLU` layer:

```text
weights:    Conv1, Conv2, Conv3, Conv4, Conv5, FC1, FC2
activations: ReLU1, ReLU2, ReLU3, ReLU4, ReLU5, FC1 ReLU
```

For example, run the W1A1 model with W8A8 first and last boundaries:

```bash
bash run_w1a1_boundary_experiments.sh
```

Run the uniform W/A sweep with:

```bash
bash run_experiment.sh
```

## Testing

Set `checkpoint_path` in `src/test.py`, then run:

```bash
python src/test.py
```

The test script loads the selected checkpoint and does not run training.

## FPGA status

The W4A4 QAT classifier has completed QONNX export, FINN compilation, Vivado
synthesis, and a PYNQ-Z2 smoke test. The next deployment candidates are W2A4
and W1A4. FPGA build artifacts, models, datasets, and local experiment outputs
are intentionally not committed.

## Status

The floating-point, QAT, and first FPGA deployment stages are complete.
