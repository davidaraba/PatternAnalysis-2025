"""
dataset.py

Handles the loading, preprocessing, and augmentation of the ADNI dataset for
Alzheimer's disease classification.

Combines robust class discovery with a clean, reusable dataloader structure.
"""

import torch 
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment
import os

# --- Configuration ---
# Path to the root of the ADNI dataset 
ADNI_ROOT_PATH = Path('/home/groups/comp3710/ADNI/AD_NC')

# Define standard image size 
IMG_SIZE = 224

# Mean and standard deviation calculated from the ADNI training set.
# These values are used for data normalisation.
DATASET_MEAN = (0.1155,)
DATASET_STD = (0.2254,)

# --- Transformations ---
# Define separate, clear pipelines for training and testing.
# The training transform includes data augmentation to help the model generalise (prevents overfitting).
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
    transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
])

# The test transform performs minimal, deterministic preprocessing.
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
])

# --- Dataset Class ---
class ADNIDataset(Dataset):
    """
    Custom PyTorch Dataset for the ADNI dataset.
    It automatically finds class folders (e.g., 'AD', 'NC') and loads images.
    """

    def __init__(self, root_dir, train=True, transform=None):
        """
        Initialises the dataset object.

        Args:
            root_dir (str or Path): The root directory of the ADNI dataset,
                                    containing 'train' and 'test' subfolders.
            train (bool): If True, loads the training set; otherwise, loads the test set.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.data_path = Path(root_dir) / ('train' if train else 'test')
        self.transform = transform # Use the provided transform

        self.image_paths = []
        self.labels = []

        # --- Automatic Class Discovery ---
        self.classes = sorted([d.name for d in self.data_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        if not self.classes:
            raise FileNotFoundError(f"No class folders found in {self.data_path}")

        # --- FIX: Sort the file list to ensure reproducible splits ---
        # Load image paths and corresponding labels
        for class_name in self.classes:
            class_path = self.data_path / class_name
            class_idx = self.class_to_idx[class_name]
            
            # Create a sorted list of image files first
            # This ensures the dataset order is deterministic
            image_files = sorted([
                f for f in class_path.iterdir() 
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ])
            
            # Add the sorted files and their labels to the lists
            for img_file in image_files:
                self.image_paths.append(img_file)
                self.labels.append(class_idx)
        
    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label by index, applying the necessary transforms.
        
        Args:
            idx (int): The index of the item.
            
        Returns:
            tuple: (image_tensor, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image in grayscale ('L' mode) as medical images are typically single-channel
        image = Image.open(image_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- Dataloader Function ---
def get_adni_dataloader(batch_size, train=True, val_split=0.2, num_workers=4, shuffle=None):
    """
    Creates and returns PyTorch DataLoader objects for the ADNI dataset.
    
    Args:
        batch_size (int): The number of samples per batch.
        train (bool): If True, returns (train_loader, val_loader).
                      If False, returns test_loader.
        val_split (float): The fraction of the training data to use for validation.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        A DataLoader or a tuple of DataLoaders.
    """
    if train:
        # Create a dataset instance *just for validation* with TEST_TRANSFORM
        val_dataset = ADNIDataset(root_dir=ADNI_ROOT_PATH, train=True, transform=TEST_TRANSFORM)
        
        # Create a dataset instance *just for training* with TRAIN_TRANSFORM
        train_dataset = ADNIDataset(root_dir=ADNI_ROOT_PATH, train=True, transform=TRAIN_TRANSFORM)

        # Get the total length
        total_len = len(train_dataset)
        
        # Split the full training set into training and validation sets
        train_size = int((1 - val_split) * total_len)
        val_size = total_len - train_size
        
        # Use random_split to get the indices for each set
        # We must use the same generator to ensure the splits are disjoint
        generator = torch.Generator().manual_seed(42) # Use a fixed seed for reproducibility
        train_indices, val_indices = random_split(range(total_len), [train_size, val_size], generator=generator)

        # Create the final Subset objects using the correct indices and datasets
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        
        print(f"Full training dataset size: {total_len}")
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        
        # We need to get the classes from one of the dataset objects
        # We'll grab it from the val_dataset's underlying dataset
        print(f"Classes: {val_dataset.dataset.class_to_idx}")
        
        train_shuffle = True if shuffle is None else shuffle

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False, # No need to shuffle validation data
            num_workers=num_workers
        )
        return train_loader, val_loader
    else:
        # Test set correctly uses TEST_TRANSFORM
        test_dataset = ADNIDataset(root_dir=ADNI_ROOT_PATH, train=False, transform=TEST_TRANSFORM)
        print(f"Test dataset size: {len(test_dataset)}")
        print(f"Classes: {test_dataset.class_to_idx}")

        test_shuffle = False if shuffle is None else shuffle
        
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=test_shuffle,
            num_workers=num_workers
        )
        return test_loader

# --- Self-Testing Block ---
# A great practice to ensure your file works independently.
if __name__ == '__main__':
    print("--- Testing Training/Validation Dataloader ---")
    train_loader, val_loader = get_adni_dataloader(batch_size=32, train=True)
    
    # Fetch one batch from the train loader
    train_images, train_labels = next(iter(train_loader))
    print(f"Train batch shape: {train_images.shape}") # Should be [32, 1, 224, 224]
    print(f"Train labels shape: {train_labels.shape}") # Should be [32]
    
    # Check if augmentation is applied (mean should be random)
    print(f"Train batch mean: {train_images.mean()}") 

    # Fetch one batch from the val loader
    val_images, val_labels = next(iter(val_loader))
    print(f"Val batch shape: {val_images.shape}") # Should be [32, 1, 224, 224]
    print(f"Val labels shape: {val_labels.shape}") # Should be [32]
    
    # Check if augmentation is NOT applied (mean should be more stable)
    print(f"Val batch mean: {val_images.mean()}") 

    print("\n--- Testing Test Dataloader ---")
    test_loader = get_adni_dataloader(batch_size=32, train=False)

    # Fetch one batch from the test loader
    test_images, test_labels = next(iter(test_loader))
    print(f"Test batch shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
