"""
train.py

This script handles the training and validation of the ConvNeXt model on the ADNI dataset.
It trains the model for a specified number of epochs, saves the best performing model,
and plots the training and validation metrics. This version includes a learning rate
warmup and refined regularization.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
# Import additional schedulers for warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Import from your other project files
from dataset import get_adni_dataloader
from modules import ConvNeXt

# --- Configuration ---
# Hyperparameters for the training process
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
EPOCHS = 350 # Increased epochs for longer training

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
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

# --- 2. The Validation/Evaluation Function ---
def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model's performance on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

# --- 3. The Main Execution Block ---
if __name__ == '__main__':
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_loader, val_loader = get_adni_dataloader(batch_size=BATCH_SIZE, train=True)
    test_loader = get_adni_dataloader(batch_size=BATCH_SIZE, train=False)

    # Initialise model, loss function, and optimiser
    print("Initialising model...")
    # Increased drop_path_rate for more regularization
    model = ConvNeXt(in_chans=1, num_classes=2, depths=[3, 3, 27, 3], drop_path_rate=0.2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    # Implement learning rate warmup combined with cosine annealing
    warmup_epochs = 5
    main_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    # Lists to store training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")

        # Update the learning rate scheduler at the end of every epoch
        scheduler.step()

    print("\nTraining finished!")

    # --- 4. Plotting and Saving Results ---
    print("Plotting training history...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy History')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    print(f"Training plot saved to {PLOT_SAVE_PATH}")

    # --- 5. Final Test Evaluation ---
    print("\nEvaluating on the test set with the best model...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}")
