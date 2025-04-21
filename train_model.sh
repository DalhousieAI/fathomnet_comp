#!/bin/bash
#SBATCH --time=03-00:00:00          # max walltime, hh:mm:ss
#SBATCH --nodes 1                   # Number of nodes to request
#SBATCH --gpus-per-node=a100:1      # Number of GPUs per node to request
#SBATCH --tasks-per-node=1          # Number of processes to spawn per node
#SBATCH --cpus-per-task=1           # Number of CPUs per GPU
#SBATCH --mem=64G                   # Memory per node
#SBATCH --output=./logs/%x_%A-%a_%n-%t.out
#SBATCH --job-name=fathomnet_comp
#SBATCH --account=          	    # Use default account

# Set and activate the virtual environment
ENVNAME=gen_env
source ~/venvs/gen_env/bin/activate

# Multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_ADDR=$(hostname -s)
export MAIN_HOST="$MASTER_ADDR"

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    echo "No arguments supplied. Usage:"
    echo "./run_model.sh \\
    --train_cfg CFG_PATH"
    exit 1
fi

# Run the Python script with all arguments passed to this script
python main.py "$@"