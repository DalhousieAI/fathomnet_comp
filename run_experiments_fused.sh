#!/bin/bash

cfg_paths=(
    "cfg/training/fused.json"
)

# Loop over each cfg path
for cfg_path in "${cfg_paths[@]}"; do
    echo "Running with cfg path: $cfg_path"
    # Run the Python script with the current cfg path
    sbatch ./train_fused_model.sh \
        --train_cfg "$cfg_path"
done