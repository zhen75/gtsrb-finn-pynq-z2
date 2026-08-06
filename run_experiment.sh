#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"
for mode in "ptq" "qat"; do
    for weight in "w8" "w4" "w2"; do
        for activate in "a8" "a4" "a2"; do
            echo "launching experiment: mode=${mode} weight=${weight} activate=${activate}"

            python run_experiment.py \
            --mode "${mode}" \
            --weight "${weight}" \
            --activate "${activate}"

            echo "the weight ${weight} activate ${activate} experiment was complete"
        done
    done
done
