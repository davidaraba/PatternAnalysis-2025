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
3. [References](#references)

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

## References

[1] National Institute of Aging. (April 5, 2023). Alzheimer's Disease Fact Sheet. National Institute on Aging. <https://www.nia.nih.gov/health/alzheimers-and-dementia/alzheimers-disease-fact-sheet>

[2] Alzheimer's Disease Neuroimaging Initiative. (2024). ADNI. <https://adni.loni.usc.edu/>

[3] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 11976-11986.
