#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"

# A1 uses QuantHardTanh with signed binary activations.
# Each scalar bit width is expanded to all seven weight layers or six activation layers.
for weight in 8 4 2 1; do
    name="w${weight}a1_hardtanh"
    echo "launching experiment: ${name}"

    python run_experiment.py \
        --mode qat \
        --name "$name" \
        --weight "$weight" \
        --activate 1

    echo "${name} experiment complete"
done
