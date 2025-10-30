#!/bin/bash
#SBATCH --job-name=adni-evaluate # Job name
#SBATCH --partition=cpu          # Use a CPU node
#SBATCH --cpus-per-task=1        # Only need 1 CPU core for this
#SBATCH --time=1:30:00          # Request 1hr 30 minutes 
#SBATCH --output=logs/evaluation_log-%j.out # Save output to logs folder

echo "Starting evaluation job on $(hostname)"

# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate adni-env

# Navigate to script's directory
cd ~/alzheimer/PatternAnalysis-2025/recognition/ad_classification_david_araba

# Run the evaluation script
python evaluate_best.py

echo "Evaluation job finished."
