#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhen/miniconda3/envs/torch/bin/python}"
cd "$SCRIPT_DIR/src"

# Uniform-bit experiments over the three most sensitive W1A1 activation
# positions: Conv2 Act, Conv3 Act, and Conv4 Act.
#
# All seven weight layers remain at 1 bit.
# Conv1 Act, Conv5 Act, and FC1 Act remain at 1 bit.
# The three selected activations are all set to 2, 4, or 8 bits together.
bit_widths=(2 4 8)
experiment_count=0

for bit_width in "${bit_widths[@]}"; do
    name="top3_activations_a${bit_width}"
    weights="1,1,1,1,1,1,1"
    activations="1,${bit_width},${bit_width},${bit_width},1,1"

    experiment_count=$((experiment_count + 1))
    echo "[${experiment_count}/3] launching experiment: ${name}"

    "$PYTHON_BIN" run_experiment.py \
        --mode qat \
        --name "$name" \
        --weight "$weights" \
        --activate "$activations"

    echo "${name} experiment complete"
done

echo "All ${experiment_count} top-3 activation experiments complete"
