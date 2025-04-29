#!/bin/bash

cfg_paths=(
    "cfg/training/efficientnet_v2_m_s42_e75_isz64.json"
    "cfg/training/efficientnet_v2_m_s42_e150_isz64.json"
    "cfg/training/efficientnet_v2_m_s42_e75_isz128.json"
    "cfg/training/efficientnet_v2_m_s42_e150_isz128.json"
    "cfg/training/efficientnet_v2_l_s42_e75_isz64.json"
    "cfg/training/efficientnet_v2_l_s42_e150_isz64.json"
    "cfg/training/efficientnet_v2_l_s42_e75_isz128.json"
    "cfg/training/efficientnet_v2_l_s42_e150_isz128.json"
)

# Loop over each cfg path
for cfg_path in "${cfg_paths[@]}"; do
    echo "Running with cfg path: $cfg_path"
    # Run the Python script with the current cfg path
    sbatch ./train_model.sh \
        --train_cfg "$cfg_path"
done