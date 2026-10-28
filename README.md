
# Single Image Super-Resolution for Aerial Images Using Deep Learning

## Overview

This project implements **Single Image Super-Resolution (SISR)** using the Enhanced Deep Super-Resolution (EDSR) network to improve the visual quality of aerial and satellite images.

The model is trained and evaluated using the **DOTA v1 dataset**. Low-resolution images are generated from high-resolution images using a ×2 downscaling process. The EDSR model learns to reconstruct higher-resolution images while preserving structural details, edges, and textures.

Image quality is evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM).

## Objectives

- Improve aerial and satellite image resolution using deep learning.
- Implement EDSR for ×2 single-image super-resolution.
- Train the model using paired low-resolution and high-resolution images.
- Evaluate reconstruction quality using PSNR and SSIM.
- Compare results with existing super-resolution methods.

## Dataset

**DOTA v1 (Dataset for Object Detection in Aerial Images)**

The dataset contains high-resolution aerial images featuring vehicles, roads, ships, buildings, and other objects.

- **High-resolution images:** Used as ground-truth references.
- **Low-resolution images:** Generated using Gaussian blur and bicubic downsampling.
- **Upscaling factor:** ×2
- **Image format:** RGB

## Methodology

The project follows four main stages:

1. **Data Preparation:** Generate paired low-resolution and high-resolution images.
2. **Model Architecture:** Implement EDSR using residual blocks.
3. **Model Training:** Train the network using L1 loss and the Adam optimizer.
4. **Evaluation:** Measure reconstruction quality using PSNR and SSIM.

## Model Configuration

| Parameter | Value |
|---|---|
| Model | EDSR |
| Dataset | DOTA v1 |
| Upscaling Factor | ×2 |
| Residual Blocks | 16 |
| Feature Channels | 64 |
| Batch Size | 16 |
| Epochs | 100 |
| LR Patch Size | 48 × 48 |
| Loss Function | L1 Loss |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Residual Scaling | 0.1 |

## Results

The proposed EDSR implementation was evaluated using PSNR and SSIM.

| Metric | EDSR (Proposed Implementation) |
|---|---:|
| PSNR | 33.8499 dB |
| SSIM | 0.8984 |

The results indicate improved image reconstruction quality, with enhanced sharpness and preservation of structural details.


## Applications

- Aerial and satellite image enhancement
- Remote sensing image analysis
- Visualization of small objects and surface features
- Image preprocessing for computer vision applications

## Technologies Used

- Python
- Deep Learning
- Convolutional Neural Networks (CNNs)
- Enhanced Deep Super-Resolution (EDSR)
- DOTA v1 Dataset
- PSNR and SSIM

## Future Improvements

- Evaluate additional upscaling factors.
- Compare EDSR with newer super-resolution architectures.
- Experiment with different image degradation methods.
- Investigate the effect of super-resolution on aerial object detection.

## References

1. Lim, B., Son, S., Kim, H., Nah, S., and Lee, K. M. (2017). *Enhanced Deep Residual Networks for Single Image Super-Resolution*. CVPR Workshops.
2. DOTA: A Large-Scale Dataset for Object Detection in Aerial Images.
3. Additional references are provided in the project paper.

## Author

Developed as a deep learning project focused on single-image super-resolution for aerial and satellite imagery.
