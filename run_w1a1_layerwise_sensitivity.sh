#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhen/miniconda3/envs/torch/bin/python}"
cd "$SCRIPT_DIR/src"

# W1A1 layer-wise sensitivity experiments.
# Exactly one weight or activation quantizer is changed from 1 bit to 8 bits.
# Weight order: Conv1, Conv2, Conv3, Conv4, Conv5, FC1, FC2.
# Activation order: Conv1 Act, Conv2 Act, Conv3 Act, Conv4 Act, Conv5 Act, FC1 Act.
experiments=(
    # Conv1-W8 completed in qat_experiments/experiment_38.
    # Conv1-A8 was interrupted by a Windows Update reboot and is restarted here.
    "w1a1_conv1_a8|1,1,1,1,1,1,1|8,1,1,1,1,1"
    "w1a1_conv2_w8|1,8,1,1,1,1,1|1,1,1,1,1,1"
    "w1a1_conv2_a8|1,1,1,1,1,1,1|1,8,1,1,1,1"
    "w1a1_conv3_w8|1,1,8,1,1,1,1|1,1,1,1,1,1"
    "w1a1_conv3_a8|1,1,1,1,1,1,1|1,1,8,1,1,1"
    "w1a1_conv4_w8|1,1,1,8,1,1,1|1,1,1,1,1,1"
    "w1a1_conv4_a8|1,1,1,1,1,1,1|1,1,1,8,1,1"
    "w1a1_conv5_w8|1,1,1,1,8,1,1|1,1,1,1,1,1"
    "w1a1_conv5_a8|1,1,1,1,1,1,1|1,1,1,1,8,1"
    "w1a1_fc1_w8|1,1,1,1,1,8,1|1,1,1,1,1,1"
    "w1a1_fc1_a8|1,1,1,1,1,1,1|1,1,1,1,1,8"
    "w1a1_fc2_w8|1,1,1,1,1,1,8|1,1,1,1,1,1"
)

echo "Running ${#experiments[@]} W1A1 layer-wise sensitivity experiments"

for experiment in "${experiments[@]}"; do
    IFS="|" read -r name weights activations <<< "$experiment"
    echo "launching experiment: ${name}"

    "$PYTHON_BIN" run_experiment.py \
        --mode qat \
        --name "$name" \
        --weight "$weights" \
        --activate "$activations"

    echo "${name} experiment complete"
done
