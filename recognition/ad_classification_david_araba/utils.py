"""
utils.py

A collection of utility functions and helper scripts for the ADNI
classification project, such as calculating dataset statistics.
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

# This import assumes utils.py and dataset.py are in the same folder.
from dataset import ADNIDataset, ADNI_ROOT_PATH

def calculate_mean_std():
    """
    Calculates the mean and standard deviation of the training dataset.
    This is used to get the normalization constants for the transforms.

    Returns:
        tuple: (mean, std)
    """
    print("Calculating dataset statistics...")
    
    # 1. Create a dataset instance with a minimal transform (just ToTensor)
    # We don't want augmentation to affect the true statistics.
    # ToTensor() also scales pixel values from [0, 255] to [0.0, 1.0].
    
    # Pass the ToTensor transform directly into the constructor,
    # as the __init__ signature in dataset.py was changed.
    dataset = ADNIDataset(
        root_dir=ADNI_ROOT_PATH,
        train=True,
        transform=transforms.ToTensor()
    )

    # 2. Use a DataLoader to iterate through the data efficiently
    loader = DataLoader(
        dataset,
        batch_size=64, # A reasonable batch size
        num_workers=4,
        shuffle=False # No need to shuffle for this calculation
    )

    # 3. Initialise variables to hold running sums
    # We calculate the sum and sum of squares to derive mean and std
    channel_sum = 0.
    channel_sum_sq = 0.
    num_pixels = 0

    # 4. Loop through the dataset
    for images, _ in tqdm(loader, desc="Calculating Stats"):
        # images shape: [batch_size, channels, height, width]
        
        # Sum up all pixel values in the batch and add to the running total
        channel_sum += torch.sum(images)
        channel_sum_sq += torch.sum(images**2)
        
        # Count the number of pixels in the batch
        # images.nelement() gives the total number of elements (pixels) in the tensor
        num_pixels += images.nelement()

    # 5. Calculate the final mean and standard deviation
    mean = channel_sum / num_pixels
    # The formula for standard deviation is sqrt(E[X^2] - (E[X])^2)
    std = torch.sqrt((channel_sum_sq / num_pixels) - (mean**2))

    print(f"\nCalculation complete.")
    print(f"Dataset Mean: {mean.item():.4f}")
    print(f"Dataset Std Dev: {std.item():.4f}")
    
    return mean.item(), std.item()

if __name__ == '__main__':
    # Run the calculation
    calculated_mean, calculated_std = calculate_mean_std()
    
    print("\nNext step: Update the DATASET_MEAN and DATASET_STD")
    print("variables in your dataset.py file with these values:")
    print(f"DATASET_MEAN = ({calculated_mean:.4f},)")
    print(f"DATASET_STD = ({calculated_std:.4f},)")