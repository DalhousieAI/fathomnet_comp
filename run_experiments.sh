#!/bin/bash

cfg_paths=(
    "cfg/training/vit_l_rankClass.json"
    "cfg/training/vit_l_rankOrder.json"
    "cfg/training/vit_l_rankFamily.json"
    "cfg/training/vit_l_rankGenus.json"
    "cfg/training/vit_l_rankSpecies.json"
)

# Loop over each cfg path
for cfg_path in "${cfg_paths[@]}"; do
    echo "Running with cfg path: $cfg_path"
    # Run the Python script with the current cfg path
    sbatch ./train_model.sh \
        --train_cfg "$cfg_path"
done