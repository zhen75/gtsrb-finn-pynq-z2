#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"

# Weight order: Conv1, Conv2, Conv3, Conv4, Conv5, FC1, FC2.
# Activation order: ReLU1, ReLU2, ReLU3, ReLU4, ReLU5, FC1 ReLU.
# The tail activation is FC1 ReLU because it is the input to FC2.
experiments=(
    "head_tail_w8a8|8,1,1,1,1,1,8|8,1,1,1,1,8"
    "tail_w8a8|1,1,1,1,1,1,8|1,1,1,1,1,8"
)

for experiment in "${experiments[@]}"; do
    IFS="|" read -r name weights activations <<< "$experiment"
    echo "launching experiment: ${name}"

    python run_experiment.py \
        --mode qat \
        --name "$name" \
        --weight "$weights" \
        --activate "$activations"

    echo "${name} experiment complete"
done
