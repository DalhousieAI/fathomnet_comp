#!/bin/bash

cfg_paths=(
    "cfg/training/wide_rn101Class.json"
    "cfg/training/wide_rn101Order.json"
    "cfg/training/wide_rn101Family.json"
    "cfg/training/wide_rn101Genus.json"
    "cfg/training/wide_rn101Species.json"
)

# Loop over each cfg path
for cfg_path in "${cfg_paths[@]}"; do
    echo "Running with cfg path: $cfg_path"
    # Run the Python script with the current cfg path
    sbatch ./train_model.sh \
        --train_cfg "$cfg_path"
done