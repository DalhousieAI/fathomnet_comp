#!/bin/bash
#SBATCH --account=rrg-whidden
#SBATCH --gpus-per-node=a100:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=10:00:00
#SBATCH --output=faiss.out

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11.5
module load scipy-stack/2024a 
module load faiss/1.8.0 
virtualenv --no-download ~/fathomnet_comp_env
source ~/fathomnet_comp_env/bin/activate
pip install --no-index torch==2.6.0 
pip install --no-index torchvision==0.21.0
pip install --no-index torchaudio==2.6.0
pip install --no-index scikit_learn==1.6.1

python faiss_exp.py