#!/bin/bash

cfg_paths=(
    "cfg/training/vit-h_s42_cce1.json"
    "cfg/training/vit-h_s42_cce1_e20.json"
    "cfg/training/vit-h_s42_ncce1.json"
    "cfg/training/vit-h_s42_ncce1_e20.json"
)

# Loop over each cfg path
for cfg_path in "${cfg_paths[@]}"; do
    echo "Running with cfg path: $cfg_path"
    # Run the Python script with the current cfg path
    sbatch ./train_model.sh \
        --train_cfg "$cfg_path"
done