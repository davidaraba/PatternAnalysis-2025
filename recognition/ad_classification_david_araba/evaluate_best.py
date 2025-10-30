"""
evaluate_best.py

Loads the best trained model, calculates detailed metrics (Accuracy, Precision, 
Recall, F1), finds the optimal classification threshold on the validation set, 
and applies this threshold to evaluate performance on the test set.
"""

import torch
import torch.nn as nn
import os
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns # For plotting confusion matrix

# Import from your project files
from dataset import get_adni_dataloader, DATASET_MEAN, DATASET_STD
from modules import ConvNeXt

# --- Function to get predictions and labels ---
def get_predictions(model, dataloader, device):
    """
    Runs the model on the dataloader and returns raw probabilities and true labels.

    Args:
        model (torch.nn.Module): The trained model (in eval mode).
        dataloader (torch.utils.data.DataLoader): Data loader (validation or test).
        device (torch.device): The device to run inference on.

    Returns:
        tuple: (probabilities, labels)
            - probabilities (torch.Tensor): Softmax probabilities for each sample.
            - labels (torch.Tensor): True labels for each sample.
    """
    model.eval() # Ensure model is in evaluation mode
    all_outputs = []
    all_labels = []
    
    # Disable gradient calculations
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Getting Predictions"):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Store results
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
            
    # Concatenate results from all batches
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(all_outputs, dim=1)
    
    return probabilities, all_labels

# --- Function to find the best threshold ---
def find_best_threshold(probabilities, labels):
    """
    Finds the optimal classification threshold on the validation set
    that maximizes the F1-score for the positive class.

    Args:
        probabilities (torch.Tensor): Softmax probabilities from the model.
        labels (torch.Tensor): True labels.

    Returns:
        float: The threshold (between 0.0 and 1.0) that gives the best F1-score.
    """
    best_threshold = 0.5
    best_f1 = 0.0
    
    # We assume class 1 ('AD') is the positive class
    positive_class_index = 1 
    
    # Iterate through possible thresholds
    for threshold in np.arange(0.1, 0.9, 0.05):
        # Apply threshold to the probability of the positive class
        preds = (probabilities[:, positive_class_index] >= threshold).long()
        
        # Calculate F1 score for this threshold
        f1 = f1_score(labels.numpy(), preds.numpy(), pos_label=positive_class_index, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    print(f"Best threshold found on validation set: {best_threshold:.2f} (F1 Score: {best_f1:.4f})")
    return best_threshold

# --- Function to calculate metrics ---
def calculate_metrics(probabilities, labels, threshold=0.5):
    """
    Calculates Accuracy, Precision, Recall, F1, and Confusion Matrix
    using a given classification threshold.

    Args:
        probabilities (torch.Tensor): Softmax probabilities from the model.
        labels (torch.Tensor): True labels.
        threshold (float): The classification threshold to use.

    Returns:
        tuple: (accuracy, precision, recall, f1, cm, preds)
            - accuracy (float)
            - precision (float)
            - recall (float)
            - f1 (float)
            - cm (np.ndarray): The confusion matrix.
            - preds (np.ndarray): The binary predictions.
    """
    positive_class_index = 1 # Assuming class 1 is positive ('AD')
    num_classes = probabilities.shape[1]
    
    # Apply threshold
    preds = (probabilities[:, positive_class_index] >= threshold).long()
    
    # Calculate metrics
    accuracy = accuracy_score(labels.numpy(), preds.numpy())
    precision = precision_score(labels.numpy(), preds.numpy(), pos_label=positive_class_index, zero_division=0)
    recall = recall_score(labels.numpy(), preds.numpy(), pos_label=positive_class_index, zero_division=0)
    f1 = f1_score(labels.numpy(), preds.numpy(), pos_label=positive_class_index, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=list(range(num_classes)))
    
    return accuracy, precision, recall, f1, cm, preds.numpy()

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the best ConvNeXt model.")
    parser.add_argument(
        '--model-path', 
        type=str, 
        default=os.path.join("checkpoints", "best_model.pth"),
        help="Path to the saved model file (.pth)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    print("Loading model...")
    # The drop_path_rate MUST match the trained model (0.4 from train.py)
    model = ConvNeXt(in_chans=1, num_classes=2, depths=[3, 3, 27, 3], drop_path_rate=0.4).to(device) 
    
    # Load the saved weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    # --- Find Best Threshold on Validation Set ---
    print("Loading validation data...")
    # We set num_workers=0 for safety in a main script (avoids multiprocessing issues)
    _, val_loader = get_adni_dataloader(batch_size=32, train=True, val_split=0.2, num_workers=0) 

    print("Finding best threshold on validation set...")
    val_probs, val_labels = get_predictions(model, val_loader, device)
    best_thresh = find_best_threshold(val_probs, val_labels)

    # --- Evaluate on Test Set ---
    print("\nLoading test data...")
    test_loader = get_adni_dataloader(batch_size=32, train=False, num_workers=0)

    print("Evaluating on test set using the best threshold...")
    test_probs, test_labels = get_predictions(model, test_loader, device)
    accuracy, precision, recall, f1, cm, test_preds = calculate_metrics(test_probs, test_labels, threshold=best_thresh)

    # --- Print Final Results ---
    print("\n--- Final Test Set Performance ---")
    print(f"Threshold: {best_thresh:.2f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (for AD class)")
    print(f"Recall:    {recall:.4f} (for AD class)")
    print(f"F1 Score:  {f1:.4f} (for AD class)")

    # --- Plot Confusion Matrix ---
    print("\nPlotting confusion matrix...")
    plt.figure(figsize=(6, 5))
    
    # Get class names dynamically from the test_loader's dataset object
    class_names = {v: k for k, v in test_loader.dataset.class_to_idx.items()}
    display_labels = [class_names[i] for i in range(len(class_names))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=display_labels, yticklabels=display_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to confusion_matrix.png")