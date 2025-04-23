#!/bin/bash

cfg_paths=(
    "cfg/training/efficientnet-b0_s42.json"
    "cfg/training/efficientnet-b1_s42.json"
    "cfg/training/efficientnet-b1_s42_e20.json"
)

# Loop over each cfg path
for cfg_path in "${cfg_paths[@]}"; do
    echo "Running with cfg path: $cfg_path"
    # Run the Python script with the current cfg path
    sbatch ./train_model.sh \
        --train_cfg "$cfg_path"
done