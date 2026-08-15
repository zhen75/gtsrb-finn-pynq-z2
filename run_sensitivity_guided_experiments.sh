#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhen/miniconda3/envs/torch/bin/python}"
cd "$SCRIPT_DIR/src"

# Sensitivity-guided mixed-precision QAT experiments.
# Weight order: Conv1, Conv2, Conv3, Conv4, Conv5, FC1, FC2.
# Activation order: Conv1 Act, Conv2 Act, Conv3 Act, Conv4 Act, Conv5 Act, FC1 Act.
experiments=(
    "sensitivity_capacity|2,2,1,1,1,1,1|2,2,2,2,1,1"
    "sensitivity_balanced|2,2,1,1,1,1,1|2,4,4,4,1,1"
    "sensitivity_high_precision|2,2,1,1,1,1,1|2,8,8,4,1,1"
)

echo "Running ${#experiments[@]} sensitivity-guided experiments"

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
