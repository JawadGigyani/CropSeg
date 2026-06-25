# CropSeg: UAV Crop/Weed Semantic Segmentation

A U-Net semantic segmentation model trained on the [PhenoBench](https://www.phenobench.org/) dataset to classify UAV imagery of sugar beet fields into **background**, **crop**, and **weed** pixels.

## Objective

Develop a deep learning pipeline for automated crop/weed segmentation from UAV-captured field imagery — a core task in precision agriculture and plant phenomics.

## Model Architecture

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Architecture | U-Net | Standard for dense segmentation with skip connections |
| Encoder | ResNet34 (ImageNet pretrained) | Good accuracy/speed tradeoff, pretrained features |
| Loss | Dice + Cross-Entropy | Handles severe class imbalance (96% background) |
| Optimizer | AdamW | Decoupled weight decay for better generalization |
| Scheduler | Cosine Annealing | Smooth LR decay without sudden drops |
| Input Size | 512 x 512 | Resized from 1024x1024 for GPU memory efficiency |
| Classes | 3 (Background, Crop, Weed) | Merged from PhenoBench's 5-class labels |

## Dataset

**PhenoBench v1.10** — high-resolution UAV imagery of sugar beet fields:
- 1,407 training images (1024x1024 px)
- 772 validation images
- Captured from ~3m altitude with GSD ~0.3 mm/px

### Class Distribution (severe imbalance)
| Class | Pixel % | Imbalance Ratio |
|-------|---------|-----------------|
| Background | 96.4% | 448x |
| Crop | 3.3% | 16x |
| Weed | 0.2% | 1x (baseline) |

## Results

Trained for **40 epochs** on Google Colab (T4 GPU), ~2.5 hours total.

### Quantitative Performance

| Metric | Score |
|--------|-------|
| **Mean IoU (mIoU)** | **0.822** |
| Background IoU | 0.993 |
| Crop IoU | 0.727 |
| Weed IoU | 0.746 |
| Best Validation Loss | 0.097 |

### Key Findings
- The combined Dice + CE loss effectively handles the extreme class imbalance
- The model achieves strong crop/weed separation despite weeds comprising only 0.2% of pixels
- Failure cases typically involve small, isolated weed patches or dense crop-weed boundaries

## Repository Structure

```
CropSeg/
├── notebooks/
│   └── training.ipynb    # Complete ML pipeline (EDA → Training → Evaluation → Inference)
├── README.md
└── .gitignore
```

## Notebook Contents

The notebook (`notebooks/training.ipynb`) contains the full ML development lifecycle:

1. **Environment Setup & Data Acquisition** — Mount Drive, install dependencies, download PhenoBench with progress bar
2. **Exploratory Data Analysis (EDA)** — Class distribution analysis, per-image statistics, sample diversity visualization
3. **Data Preprocessing & Augmentation** — Custom Dataset class, label merging, augmentation pipeline (flips, rotations, color jitter, blur)
4. **Augmentation Preview** — Visual demonstration of random transforms applied to training samples
5. **Model Architecture** — U-Net + ResNet34 definition with architecture summary
6. **Training Configuration** — Loss functions, optimizer, scheduler, metric definitions
7. **Training Loop** — 40-epoch training with resume support, Google Drive checkpointing, and progress tracking
8. **Training Analysis** — Loss and IoU convergence curves
9. **Model Evaluation** — Confusion matrix, per-class IoU bar chart
10. **Qualitative Results** — Side-by-side prediction grid (image / ground truth / prediction)
11. **Failure Analysis** — Identifies and visualizes worst-performing predictions by IoU
12. **Inference on Unseen Images** — Interactive cell to upload and predict on new images

## Trained Weights

The trained model checkpoint (~280 MB) is hosted on Google Drive:

**[Download Trained Weights](https://drive.google.com/drive/folders/1fGvRF82Xw8xv0RZ5-UVTLnWPkJk7pJnB?usp=sharing)**

The folder contains:
- `best_model.pt` — Best checkpoint (highest validation mIoU)
- `epoch_10.pt` through `epoch_40.pt` — Intermediate checkpoints
- `training_curves.png` — Loss/IoU plots
- `prediction_grid.png` — Visual results

## Setup & Usage

### Option 1: Run on Google Colab (Recommended)

1. Open `notebooks/training.ipynb` in Google Colab
2. The notebook automatically:
   - Mounts Google Drive
   - Downloads PhenoBench (~7.6 GB, first time only)
   - Trains the model with resume support
3. To **skip training** and use pretrained weights:
   - Upload `best_model.pt` to `MyDrive/CropSeg/checkpoints/`
   - The training loop will detect the checkpoint and resume from epoch 40 (effectively skipping)
4. Use the final "Inference on Unseen Images" cell to predict on new images

### Option 2: Local Inference Only

```bash
pip install torch torchvision segmentation-models-pytorch albumentations opencv-python matplotlib numpy
```

```python
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

# Load model
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
checkpoint = torch.load("best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Predict
image = cv2.imread("your_image.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (512, 512))
tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
tensor = tensor.unsqueeze(0)

with torch.no_grad():
    pred = model(tensor)
    mask = pred.argmax(dim=1).squeeze().numpy()

# mask values: 0=Background, 1=Crop, 2=Weed
```

## Technologies

- **PyTorch** — Deep learning framework
- **Segmentation Models PyTorch** — U-Net implementation with pretrained encoders
- **Albumentations** — Fast image augmentation
- **OpenCV** — Image I/O and preprocessing
- **Google Colab** — Free T4 GPU training environment
- **PhenoBench** — UAV crop/weed segmentation benchmark dataset

## References

- Weyler, J., et al. "PhenoBench: A Large Dataset and Benchmarks for Semantic Image Interpretation in the Agricultural Domain." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024.
- Ronneberger, O., Fischer, P., & Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*, 2015.
- He, K., et al. "Deep Residual Learning for Image Recognition." *CVPR*, 2016.
