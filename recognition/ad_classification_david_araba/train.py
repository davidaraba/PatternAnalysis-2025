"""
train.py

This script handles the training and validation of the ConvNeXt model on the ADNI dataset.
It trains the model for a specified number of epochs, saves the best performing model,
and plots the training and validation metrics.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Import from your other project files
from dataset import get_adni_dataloader
from modules import ConvNeXt

# --- Configuration ---
# Hyperparameters for the training process
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 50 # Start with 50, you might need more or less

# Paths for saving outputs
MODEL_SAVE_PATH = "best_model.pth"
PLOT_SAVE_PATH = "training_history.png"
CHECKPOINT_DIR = "checkpoints"

# Ensure the checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, MODEL_SAVE_PATH)

# --- 1. The Training Function ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs one full epoch of training.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Use tqdm for a progress bar
    for images, labels in tqdm(dataloader, desc="Training"):
        # Move data to the selected device (GPU or CPU)
        images, labels = images.to(device), labels.to(device)

        # 1. Forward pass: compute predicted outputs
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 2. Backward pass and optimisation
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights

        # 3. Track statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc
