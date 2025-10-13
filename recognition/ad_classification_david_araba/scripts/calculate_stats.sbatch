#!/bin/bash
#SBATCH --job-name=calc-adni-stats
#SBATCH --partition=cpu         # Use a CPU-only partition for this simple task
#SBATCH --cpus-per-task=4       # Request 4 CPUs
#SBATCH --time=00:10:00         # Request 10 minutes of wall time
#SBATCH --output=stats_log-%j.out  # Where to save the output

# --- Your Commands Start Here ---

echo "Starting job on $(hostname)"

# 1. Activate your Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate adni-env

# 2. Run your Python script
# The '--cpus-per-task' SBATCH directive will set the number of workers automatically
python utils.py

echo "Job finished."
