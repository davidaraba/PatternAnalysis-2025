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
    transforms.ToTensor,
    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
])
