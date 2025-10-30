"""
predict.py

This script demonstrates how to use the trained ConvNeXt model for inference.
It loads the best model weights, takes a random sample of images from the test set,
makes predictions, and visualises the results in a dedicated output directory.
"""

import torch
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np

# Import from other project files
from dataset import get_adni_dataloader, DATASET_MEAN, DATASET_STD
from modules import ConvNeXt

# --- Main Prediction and Visualisation Function ---
def predict_and_visualise(model_path, output_dir, num_images=9):
    """
    Loads the model, makes predictions on a random test batch, and plots the results.

    Args:
        model_path (str): Path to the saved .pth model file.
        output_dir (str): Directory to save the prediction plot.
        num_images (int): The number of images to plot (should be a perfect square, e.g., 9).
    """

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Initialise and load the model ---
    print(f"Loading model from {model_path}...")
    
    # The model architecture MUST match the saved weights.
    model = ConvNeXt(
        in_chans=1, 
        num_classes=2, 
        depths=[3, 3, 27, 3], 
        drop_path_rate=0.4
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    # --- 2. Load a random batch of test data ---
    print("Loading a random batch of test data...")
    # Using shuffle=True on the test loader produces random samples
    # Set num_workers=0 for safety in a main script
    test_loader = get_adni_dataloader(batch_size=num_images, train=False, shuffle=True, num_workers=0)
    
    try:
        images, labels = next(iter(test_loader))
    except StopIteration:
        print("Error: Test data loader is empty.")
        return

    images, labels = images.to(device), labels.to(device)

    # Get class names (e.g., {0: 'CN', 1: 'AD'})
    class_names = {v: k for k, v in test_loader.dataset.class_to_idx.items()}
    
    # --- 3. Make predictions ---
    print("Making predictions...")
    with torch.no_grad():
        outputs = model(images)
        _, predicted_indices = torch.max(outputs, 1)

    # Move data to CPU for plotting with matplotlib
    images_cpu = images.cpu()
    labels_cpu = labels.cpu()
    predicted_indices_cpu = predicted_indices.cpu()

    # --- 4. Create the visualisation plot ---
    print("Generating visualisation...")
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "prediction_examples.png")

    # Determine grid size (e.g., 3x3 for 9 images)
    grid_size = int(np.sqrt(num_images))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]
        
        # Un-normalise the image for correct display
        img = images_cpu[i].squeeze()
        mean = torch.tensor(DATASET_MEAN)
        std = torch.tensor(DATASET_STD)
        img = img * std + mean # img = (img * std) + mean
        
        # Get string names for labels
        pred_class = class_names[predicted_indices_cpu[i].item()]
        true_class = class_names[labels_cpu[i].item()]

        ax.imshow(img, cmap='gray') # Use 'gray' colormap for grayscale
        
        # Set title color based on correctness
        color = "green" if pred_class == true_class else "red"
        ax.set_title(f"Pred: {pred_class}, True: {true_class}", color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Prediction visualisation saved to {save_path}")

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run prediction for a trained ConvNeXt model.")
    
    # Argument for model path
    parser.add_argument(
        '--model-path', 
        type=str, 
        default=os.path.join("checkpoints", "best_model.pth"),
        help="Path to the saved model file (.pth)."
    )

    # Argument for the output directory
    parser.add_argument(
        '--output-dir',
        type=str,
        default="prediction_outputs",
        help="Directory to save the output visualisations."
    )
    args = parser.parse_args()

    # Check if the model file exists before running
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please run train.py to generate the model file first.")
    else:
        predict_and_visualise(model_path=args.model_path, output_dir=args.output_dir)