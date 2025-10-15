#!/bin/bash
#SBATCH --job-name=adni-convnext-train
#SBATCH --partition=a100      # Request a powerful A100 GPU
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --cpus-per-task=4     # Request 4 CPUs for data loading
#SBATCH --time=02:00:00       # Request 2 hours of wall time
#SBATCH --output=training_log-%j.out

echo "Starting training job on $(hostname)"

# Activate your Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate adni-env

# Navigate to your script's directory
cd ~/alzheimer/PatternAnalysis-2025/recognition/ad_classification_david_araba

# Run the training script
python train.py

echo "Training job finished."
