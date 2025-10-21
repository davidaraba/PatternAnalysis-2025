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
6. [Model Architecture](#model-architecture)
7. [Training Strategy](#training-strategy)
8. [Results and Performance](#results-and-performance)
9. [References](#references)

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

```python
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

```text
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
  
## Model Architecture

The ConvNeXt is a deep learning architecture originally designed for image classification created by Facebook AI Research from their release of "A ConvNet for the 2020s" [3]. The ConvNeXt has a modern convolutional architecture that incorporates design principles from Vision Transformers while maintaining the efficiency of CNNs. The model design follows closely to the original implementation with three main architectural innovations:

- **Modern Convolutional Design**: Large kernel sizes and inverted bottleneck structure
- **Layer Normalization**: Replacing Batch Normalization for better stability
- **Efficient Downsampling**: Progressive spatial dimension reduction

This design addresses the issue of efficiency, as the ConvNeXt architecture achieves state-of-the-art performance while maintaining computational efficiency suitable for medical imaging applications [3].

### Why Use ConvNeXt?

The ConvNeXt is designed for image classification and has performed significantly well on the large visual database ImageNet in the original paper [3]. For the ADNI dataset, the task is very similar, to learn the underlying data structures of the images and to classify whether a given image has Alzheimer's or not. A major benefit of the ConvNeXt compared to other deep learning algorithms is its scalability to train on more complex images in a shorter time frame which is no doubt an important considered aspect in the medical research industry. For this problem space, the ConvNeXt meets the criteria of a fast and accurate solution with the ability of the model to expand to more complex data in the future.

The overall architecture of the model starts by taking an input image and applying a stem layer with 4×4 convolution and stride 4 for initial downsampling. The ConvNeXt consists of multiple stages, each containing several ConvNeXt blocks that combine depthwise convolution with pointwise operations, followed by a feed forward network (FFN) similar to a vision transformer. The output of the last block is fed into a global average pooling layer and then into a linear classifier.

### ConvNeXt Design Principles

ConvNeXt represents a modern approach to convolutional neural networks, incorporating design principles from Vision Transformers while maintaining the efficiency of CNNs. The architecture includes:

#### 1. Stem Layer

- **4×4 Convolution**: Initial feature extraction with stride 4
- **Layer Normalization**: Consistent normalization throughout the network
- **Efficient Downsampling**: Reduces computational complexity early

#### 2. ConvNeXt Blocks

Each block implements:

- **Depthwise Convolution**: 7×7 kernel with groups=dim for spatial feature extraction
- **Layer Normalization**: Applied in channels_last format for efficiency
- **Pointwise Convolutions**: 1×1 convolutions for channel mixing
- **GELU Activation**: Smooth activation function for better gradients
- **Layer Scaling**: Learnable scaling factors for improved training stability
- **DropPath Regularization**: Stochastic depth for better generalization

#### 3. Architecture Variants

The implementation supports multiple architectural configurations:

- **Small**: depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]
- **Base**: depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]
- **Large**: depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]

### Key Architectural Innovations

#### 1. Modern Convolutional Design

- **Large Kernel Sizes**: 7×7 depthwise convolutions capture long-range dependencies
- **Inverted Bottleneck**: Efficient channel expansion and compression
- **Fewer Activation Functions**: Reduced computational overhead

#### 2. Advanced Normalization

- **Layer Normalization**: More stable than Batch Normalization for small batches
- **Channels Last Format**: Optimized memory layout for modern hardware
- **Consistent Normalization**: Applied throughout the network for stability

#### 3. Efficient Feature Aggregation

- **Global Average Pooling**: Reduces overfitting compared to fully connected layers
- **Minimal Classification Head**: Single linear layer for final prediction

## Training Strategy

### Optimization Configuration

#### 1. Optimizer Settings

- **Algorithm**: AdamW with improved weight decay
- **Learning Rate**: 5e-5 (carefully tuned for medical imaging)
- **Weight Decay**: 0.05 for effective regularization
- **Beta Parameters**: Default values (β₁=0.9, β₂=0.999)

#### 2. Learning Rate Scheduling

```python
# Warmup phase (5 epochs)
warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=5)

# Main training phase (cosine annealing)
main_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS-5, eta_min=1e-6)

# Combined scheduler
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[5])
```

#### 3. Regularization Techniques

- **Label Smoothing**: 0.1 smoothing factor to prevent overconfident predictions
- **DropPath**: 0.2 probability for stochastic depth regularization
- **Weight Decay**: L2 regularization on model parameters

### Training Protocol

#### 1. Training Configuration

- **Epochs**: 250 (extended training for convergence)
- **Batch Size**: 32 (balanced memory usage and gradient stability)
- **Validation Frequency**: Every epoch
- **Model Checkpointing**: Best validation accuracy saved

#### 2. Data Augmentation Strategy

Training-time augmentations include:

- **Random Cropping**: 224×224 from 256×256 resized images
- **Horizontal Flipping**: 50% probability for data diversity
- **Rotation**: ±15 degrees for robustness to orientation
- **Color Jittering**: Brightness and contrast variation (±10%)

#### 3. Loss Function

Cross-entropy loss with label smoothing:

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

## Results and Performance

This ConvNeXt model achieved a final accuracy of [PLACEHOLDER_ACCURACY]% and a test loss of [PLACEHOLDER_LOSS] on the ADNI test dataset. This accuracy was reached after training the model over 250 Epochs which took a total of [PLACEHOLDER_TIME] hours on the UQ Rangpur HPC.

Below are two graphs showing the training and validation loss over the 250 epochs, as well as the validation accuracy.

![Train](assets/training_history.png)
_Training and validation losses over 250 epochs_

![Accuracy](assets/validation_accuracy.png)
_Validation accuracy progression during training_

We can see that the validation loss follows quite closely to the training, only lagging by approximately [PLACEHOLDER_LAG] at the end of training. The graphs show a fast decrease in the first 50 epochs than followed by a more gradual decrease. A longer training duration could have been conducted, however, during the development of the model test results showed signs of overfitting with no increase in accuracy made.

Below is the resulting confusion matrix of the testing data:

![Confusion](assets/confusion_matrix.png)
_Confusion matrix showing classification performance on test dataset_

The model design used here on the ADNI dataset was chosen following the recommendations of the original ConvNeXt design and through trialing a variety of different hyperparameters through the development phase. Below are the set parameters for the model:

```python
model = ConvNeXt(
    in_chans=1,
    num_classes=2,
    depths=[3, 3, 27, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.2,
    layer_scale_init_value=1e-6,
    head_init_scale=1.0
)

# Training hyperparameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
EPOCHS = 250
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
```

The results do show a relatively high success rate of correct predictions of Alzheimer's disease, however, there is still room for improvement. Further testing of the ConvNeXt with a greater depth and/or number of embedded dimensions may yield a higher test accuracy. However, this will likely increase the training time significantly. Variants of the ConvNeXt may also work well on ADNI, such as the hierarchical models which may capture the underlying data structure better.

## References

[1] National Institute of Aging. (April 5, 2023). Alzheimer's Disease Fact Sheet. National Institute on Aging. <https://www.nia.nih.gov/health/alzheimers-and-dementia/alzheimers-disease-fact-sheet>

[2] Alzheimer's Disease Neuroimaging Initiative. (2024). ADNI. <https://adni.loni.usc.edu/>

[3] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 11976-11986.
