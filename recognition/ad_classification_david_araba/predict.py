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

# Import from your other project files
from dataset import get_adni_dataloader, DATASET_MEAN, DATASET_STD
from modules import ConvNeXt

# --- Main Prediction and Visualisation Function ---
def predict_and_visualise(model_path, output_dir, num_images=9):
    """
    Loads the model, makes predictions on a random test batch, and plots the results.
    """

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Initialise and load the model
    print(f"Loading model from {model_path}...")
    # Ensure load the same 'Small' architecture
    model = ConvNeXt(in_chans=1, num_classes=2, depths=[3, 3, 27, 3]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load a random batch of test data
    print("Loading a random batch of test data...")
    # Using shuffle=True on the test loader produces random samples
    test_loader = get_adni_dataloader(batch_size=num_images, train=False, shuffle=True, num_workers=0)
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    class_names = {v: k for k, v in test_loader.dataset.class_to_idx.items()}
    
    # 3. Make predictions
    print("Making predictions...")
    with torch.no_grad():
        outputs = model(images)
        _, predicted_indices = torch.max(outputs, 1)

    images_cpu = images.cpu()
    labels_cpu = labels.cpu()
    predicted_indices_cpu = predicted_indices.cpu()

    # 4. Create the visualisation plot
    print("Generating visualisation...")
    # NEW: Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "prediction_examples.png")

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]
        
        # Un-normalise the image for correct display
        img = images_cpu[i].squeeze()
        mean = torch.tensor(DATASET_MEAN)
        std = torch.tensor(DATASET_STD)
        img = img * std + mean
        
        pred_class = class_names[predicted_indices_cpu[i].item()]
        true_class = class_names[labels_cpu[i].item()]

        ax.imshow(img, cmap='viridis')
        ax.set_title(f"Pred: {pred_class}, True: {true_class}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Prediction visualisation saved to {save_path}")

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run prediction for a trained ConvNeXt model.")
    parser.add_argument(
        '--model-path', 
        type=str, 
        default=os.path.join("checkpoints", "best_model.pth"),
        help="Path to the saved model file (.pth)."
    )

    # Add an argument for the output directory
    parser.add_argument(
        '--output-dir',
        type=str,
        default="prediction_outputs",
        help="Directory to save the output visualisations."
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please run train.py to generate the model file first.")
    else:
        predict_and_visualise(model_path=args.model_path, output_dir=args.output_dir)
