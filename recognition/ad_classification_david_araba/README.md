# Alzheimer's Disease Classification using ConvNeXt Architecture

**Author**: David Araba  
**Institution**: University of Queensland  
**Course**: COMP3710 - Pattern Analysis  
**Academic Year**: 2025

## Abstract

This repository contains a PyTorch implementation of a custom ConvNeXt architecture for identifying Alzheimer's disease from 2D MRI scans of the brain. Alzheimer's disease is a progressive neurodegenerative disease that destroys memory and many other important mental functions [1]. Early detection of Alzheimer's is a critical step for providing proper treatment to patients. This project aims to address the problem of fast classification of Alzheimer's Disease from brain scans using the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset [2]. The dataset contains a number of sliced MRI brain scan images separated into Cognitive Normal (CN) and Alzheimer's Disease (AD) images. This model is built following the ConvNeXt design "A ConvNet for the 2020s" [3].

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Background and Motivation](#background-and-motivation)
3. [Methodology](#methodology)
4. [Implementation Details](#implementation-details)
5. [Dataset and Preprocessing](#dataset-and-preprocessing)
6. [References](#references)

## Problem Statement

Alzheimer's disease represents one of the most significant global health challenges of the 21st century, affecting over 50 million people worldwide [1]. Early and accurate diagnosis is crucial for effective treatment planning and patient care management. Traditional diagnostic methods rely heavily on clinical assessment and neuropsychological testing, which can be subjective, time-consuming, and may miss early-stage indicators.

The primary objective of this project is to develop an automated, reliable system for classifying brain scan images to distinguish between individuals with Alzheimer's disease and cognitively normal subjects. This classification task addresses the critical need for objective, scalable diagnostic tools that can assist medical professionals in making informed clinical decisions.

## Background and Motivation

### Understanding Alzheimer's Disease

Alzheimer's disease is a progressive neurodegenerative disorder characterized by the gradual deterioration of cognitive function and memory. The disease manifests through several pathological hallmarks:

- **Amylˈoid Plaques**: Abnormal protein deposits that accumulate between nerve cells
- **Neurofibrillary Tangles**: Twisted protein fibers that form inside brain cells
- **Neuronal Loss**: Progressive death of brain cells, particularly in memory-related regions
- **Synaptic Dysfunction**: Impaired communication between neurons

### Clinical Significance

Early detection of Alzheimer's disease is paramount because:

- Current treatments are most effective in early stages
- Patients and families can better plan for future care needs
- Early intervention may slow disease progression
- Accurate diagnosis prevents misdiagnosis of treatable conditions

### Technological Motivation

Medical imaging, particularly magnetic resonance imaging (MRI), provides non-invasive visualization of brain structure and can reveal subtle changes associated with Alzheimer's disease. However, manual interpretation of these images is:

- Time-intensive for radiologists
- Subject to inter-observer variability
- Limited by human perceptual capabilities
- Expensive for healthcare systems

Automated classification systems address these limitations by providing consistent, rapid, and objective analysis of medical images.

## Methodology

### Approach Overview

This project employs a supervised learning approach using the ConvNeXt architecture [3], a modern convolutional neural network that combines the efficiency of CNNs with design principles from Vision Transformers [5]. The methodology follows a systematic pipeline:

1. **Data Acquisition and Preprocessing**: Standardized loading and augmentation of ADNI dataset images
2. **Architecture Implementation**: Custom ConvNeXt implementation optimized for medical imaging
3. **Training Strategy**: Advanced optimization with learning rate scheduling and regularization
4. **Evaluation Protocol**: Comprehensive assessment using multiple metrics
5. **Validation Framework**: Robust train/validation/test split methodology

### Technical Innovation

The implementation incorporates several advanced techniques:

- **Custom LayerNorm**: Flexible normalization supporting both channels_first and channels_last formats [6]
- **Depthwise Convolutions**: Efficient feature extraction with reduced parameter count
- **Layer Scaling**: Improved gradient flow and training stability
- **Advanced Data Augmentation**: Robust preprocessing pipeline for medical images

## Implementation Details

### Project Structure

```
ad_classification_david_araba/
├── modules.py         # ConvNeXt architecture implementation
├── dataset.py         # ADNI dataset loading and preprocessing
├── train.py           # Training script with advanced optimization
├── predict.py         # Inference and visualization utilities
├── utils.py           # Statistical analysis and helper functions
├── requirements.txt   # Python dependencies
├── README.md          # Comprehensive documentation
└── scripts/           # SLURM job scripts for HPC execution
    ├── train.sbatch
    └── calculate_stats.sbatch
```

### Core Components

#### 1. Model Architecture (`modules.py`)

The ConvNeXt implementation features:

- **Custom LayerNorm**: Supports both data format conventions
- **Block Design**: Combines depthwise convolution with pointwise operations
- **Efficient Stem**: 4x4 convolution with stride 4 for initial downsampling
- **Progressive Downsampling**: Systematic reduction of spatial dimensions
- **Global Average Pooling**: Efficient feature aggregation

#### 2. Data Pipeline (`dataset.py`)

Sophisticated data handling includes:

- **Automatic Class Discovery**: Dynamic detection of class folders
- **Robust Preprocessing**: Normalization using dataset-specific statistics
- **Advanced Augmentation**: Training-time transformations for generalization
- **Memory Optimization**: Efficient loading with pre-allocated arrays

#### 3. Training Framework (`train.py`)

Advanced training methodology:

- **AdamW Optimizer**: Improved weight decay implementation
- **Learning Rate Scheduling**: Warmup followed by cosine annealing
- **Label Smoothing**: Regularization technique for better generalization
- **Comprehensive Logging**: Detailed metrics tracking and visualization

## About the Dataset

The ADNI dataset has been used in this project to train and test our model. ADNI is a well-known Alzheimer's disease research dataset that includes thousands of Magnetic Resonance Imaging (MRI) brain scans [2]. The data has been separated into two groups; Cognitively Normal (CN), which are brain images of healthy individuals, and Alzheimer's Disease (AD), which are individuals who have been diagnosed with Alzheimer's disease.

The ADNI dataset can be downloaded from their website, [ADNI website](https://adni.loni.usc.edu/).

Here is an example image of what the data looks like from the training set:

![Example](assets/example_image.png)
_Example brain scan from the ADNI dataset showing a typical MRI slice used for classification_

## Dataset and Preprocessing

### ADNI Dataset Description

The Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset is a comprehensive collection of neuroimaging and biomarker data designed to accelerate research into Alzheimer's disease [2]. This implementation utilizes the preprocessed version containing:

- **Image Format**: Grayscale medical images
- **Classes**: Alzheimer's Disease (AD) and Cognitively Normal (CN)
- **Dataset Split**: Separate train/test directories
- **Image Dimensions**: Variable sizes, standardized to 224×224 pixels

### Dataset Structure

This model is assumed to be run on the UQ Rangpur HPC with the dataset directory location:
**'/home/groups/comp3710/ADNI/AD_NC'.**
If this directory location does not work for you, this can be changed in [dataset.py](dataset.py).

The ADNI dataset structure will require to have the following:

```
 AD_NC/
    ├── test/
    │   ├── AD/
    │   └── CN/
    ├── train/
        ├── AD/
        └── CN/
```

### Pre-processing the Data

The images get pre-processed prior to training and testing. This step is completed in the training stage when running [train.py](train.py) which calls [dataset.py](dataset.py) to process the data. _Note: The model assumes there is a training and testing split already in the data directory. This will be explained under the "Usage" heading._ The preprocessing for the training and testing data includes:

- Splitting the training set to 20% validation, 80% training. This is to help evaluate the model's performance.
- Resizing the images to 224×224 pixels to ensure consistency across all images.
- Setting the Images to greyscale to ensure all images are consistent and to reduce computation time. Minimal information loss would occur as the images are already presented in a grey-scale.
- Normalizing the images to a mean of 0.1155 and a standard deviation of 0.2254. This was calculated in the [utils.py](utils.py) file by iterating through the training images and averaging their means and standard deviations. This is to help the network during training by receiving consistent input.

**Other preprocessing applied to only the training dataset**:

- Random augmentation, random cropping and random horizontal flips. This was to improve the generalization of the model to the testing data.

#### Training Transformations

```python
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),                    # Initial resize for cropping
    transforms.RandomCrop(224),               # Random cropping for augmentation
    transforms.RandomHorizontalFlip(),        # Horizontal flipping
    transforms.RandomRotation(15),            # Rotation augmentation
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Color variation
    transforms.ToTensor(),                    # Convert to tensor
    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)  # Normalization
])
```

#### Test Transformations

```python
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),                   # Consistent resizing
    transforms.CenterCrop(224),              # Center cropping
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)  # Normalization
])
```

### Data Splitting Strategy

The implementation employs a robust data splitting methodology:

- **Training Set**: 80% of available training data
- **Validation Set**: 20% of available training data (for hyperparameter tuning)
- **Test Set**: Dedicated test directory (for final evaluation)

This approach ensures:

- Unbiased performance estimation
- Proper hyperparameter validation
- Generalization assessment on unseen data

## References

[1] National Institute of Aging. (April 5, 2023). Alzheimer's Disease Fact Sheet. National Institute on Aging. <https://www.nia.nih.gov/health/alzheimers-and-dementia/alzheimers-disease-fact-sheet>

[2] Alzheimer's Disease Neuroimaging Initiative. (2024). ADNI. <https://adni.loni.usc.edu/>

[3] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 11976-11986.
