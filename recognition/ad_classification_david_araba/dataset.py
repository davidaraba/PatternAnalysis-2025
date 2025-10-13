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
import os

# --- Configuration ---
# Path to the root of the ADNI dataset 
ADNI_ROOT_PATH = Path('/home/groups/comp3710/ADNI/AD_NC')

# Define standard image size 
IMG_SIZE = 224

# These values just placeholders for not.
# Will create utility function to get actual 
# values for normalisation later
DATASET_MEAN = (0.115,)
DATASET_STD = (0.220,)

# --- Transformations ---
# Define separate, clear pipelines for training and testing.
# The training transform includes data augmentation to help the model generalise (prevents overfitting).
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
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

    def __init__(self, root_dir, train=True):
        """
        Initialises the dataset object.

        Args:
            root_dir (str or Path): The root directory of the ADNI dataset,
                                    containing 'train' and 'test' subfolders.
            train (bool): If True, loads the training set; otherwise, loads the test set.
        """
        self.data_path = Path(root_dir) / ('train' if train else 'test')
        self.transform = TRAIN_TRANSFORM if train else TEST_TRANSFORM

        self.image_paths = []
        self.labels = []

        # --- Automatic Class Discovery ---
        self.classes = sorted([d.name for d in self.data_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        if not self.classes:
            raise FileNotFoundError(f"No class folders found in {self.data_path}")

        # Load image paths and corresponding labels
        for class_name in self.classes:
            class_path = self.data_path / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_file in class_path.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
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
