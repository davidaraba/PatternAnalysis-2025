#!/bin/bash
#SBATCH --job-name=adni-convnext-train
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/training_log-%j.out  

echo "Starting training job on $(hostname)"

# Create the logs directory if it doesn't exist
mkdir -p logs 

# Activate your Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate adni-env

# Navigate to your script's directory
cd ~/alzheimer/PatternAnalysis-2025/recognition/ad_classification_david_araba

# Run the training script
python train.py

echo "Training job finished."
