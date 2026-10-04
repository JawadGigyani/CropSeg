# CropSeg: UAV Crop/Weed Semantic Segmentation

A U-Net semantic segmentation model trained on the [PhenoBench](https://www.phenobench.org/) dataset to classify UAV imagery of sugar beet fields into **soil (background)**, **crop**, and **weed** pixels.

## Objective

Develop a deep learning pipeline for automated crop/weed segmentation from UAV-captured field imagery — a core task in precision agriculture and plant phenomics.

## Model Architecture

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Architecture | U-Net | Standard for dense segmentation with skip connections |
| Encoder | ResNet34 (ImageNet pretrained) | Good accuracy/speed tradeoff, pretrained features |
| Loss | Dice + Cross-Entropy | Dice weights each class equally, to address severe class imbalance (weeds ≈0.5% of pixels) |
| Optimizer | AdamW (lr 3e-4, weight decay 1e-4) | Decoupled weight decay |
| Scheduler | Cosine Annealing (T_max = 40 epochs) | Smooth LR decay without sudden drops |
| Input Size | 512 x 512 | Resized from 1024x1024 for GPU memory efficiency |
| Classes | 3 (Soil, Crop, Weed) | PhenoBench's 5 labels merged: partial crop → crop, partial weed → weed (the official benchmark rule) |

## Dataset

**PhenoBench v1.1.0** — high-resolution UAV imagery of sugar beet fields ([Weyler et al., 2024](https://arxiv.org/abs/2306.04557)):
- 1,407 training and 772 validation images (1024x1024 px); the 693 test images have hidden labels and are scored by the benchmark server
- Captured from ~21 m altitude, giving a ground sampling distance of ~1 mm/px
- Images come from three dates in 2020 (05-15, 05-26, 06-05), i.e. three growth stages; the date is the filename prefix

### Class Distribution (severe imbalance)

| Split | Soil | Crop | Weed | Soil : weed |
|-------|------|------|------|-------------|
| Train (all 1,407 images) | 87.7% | 11.9% | 0.50% | ~176 : 1 |
| Validation (all 772 images) | 89.9% | 9.6% | 0.53% | ~170 : 1 |

Imbalance is strongest at the earliest growth stage:

| Date | Train soil / crop / weed | Train soil : weed | Val soil / crop / weed | Val soil : weed |
|------|------|------|------|------|
| 05-15 | 96.7 / 3.1 / 0.19% | 517 : 1 | 96.8 / 3.0 / 0.13% | 744 : 1 |
| 05-26 | 88.1 / 11.5 / 0.36% | 245 : 1 | 89.8 / 9.8 / 0.37% | 244 : 1 |
| 06-05 | 73.3 / 25.6 / 1.11% | 66 : 1 | 76.4 / 22.2 / 1.45% | 53 : 1 |

The EDA cell in the notebook samples the first 200 training files in filename order. These are all from 05-15, so its output (96.4% / 3.3% / 0.2%, 448:1) describes only the earliest growth stage.

## Results

Trained for **40 epochs** on Google Colab (T4 GPU). All results are on the PhenoBench **validation** split, with predictions and labels at 512x512.

### Quantitative Performance

| Metric | Score |
|--------|-------|
| **Mean IoU (mIoU)** | **0.871** |
| Soil IoU | 0.993 |
| Crop IoU | 0.941 |
| Weed IoU | 0.680 |
| Pixel accuracy | 0.993 |
| Best validation loss (Dice + CE) | 0.144 |

### How mIoU is computed

The notebook reports three mIoU values. They differ only in how IoU is averaged:

| Definition | Value | Where in the notebook |
|------------|-------|-------|
| **Dataset-level** — IoU from one confusion matrix over every validation pixel. This is the PhenoBench benchmark's definition. | **0.871** | Section 7, per-class IoU bar chart |
| Batch-averaged — IoU per validation batch, averaged over 97 batches. Used to select the checkpoint (epoch 39). | 0.822 | Training log |
| Per-image average | 0.794 | Failure analysis |

Batch and per-image averaging penalise the weed class: when a batch or image has only a few weed pixels, a handful of errors drives its weed IoU toward zero. That's why weed IoU is 0.68 at dataset level but 0.55 in the batch-averaged training log.

### Key Findings
- Crop is segmented reliably (IoU 0.94). Weed is the hard class (IoU 0.68), consistent with it covering only ~0.5% of pixels.
- Most weed errors are **weed↔soil**, not weed↔crop. Of weed pixels the model missed, 73% were predicted as soil. Of pixels wrongly predicted as weed, 73% were soil. Crop↔weed confusion is ~50–58k pixels in each direction, against ~138–156k for weed↔soil.
- The six worst validation images (by per-image mIoU) all come from the earliest growth stage (05-15). They are almost bare soil with a few small seedlings, so a handful of misclassified pixels drives per-image IoU down.

### Limitations
- **No held-out test result.** The checkpoint was selected on the validation split that is also reported. PhenoBench's own baselines (ERFNet 85.98, DeepLabV3+ 85.97 mIoU) are on the hidden test set at full resolution, so they aren't directly comparable to these validation numbers.
- **Evaluated at 512x512**, not PhenoBench's native 1024x1024. Downsampling removes detail from the smallest plants.
- **Single run, no fixed random seed**, so there is no variance estimate.
- **No loss ablation.** Dice + CE was chosen to address class imbalance but was not compared against plain cross-entropy.
- **Failure analysis is partial.** It's based on the six worst images and the confusion matrix. Errors were not measured by plant size or distance to crop–weed boundaries.

### Notes on the notebook text
The notebook's code and outputs are the record of the run. A few of its markdown comments disagree with those outputs:
- Section 6 says the loss curves "stabilize by epoch ~15". The validation loss actually settles around epoch 26.
- Section 6 says weed IoU "continues improving throughout". It's flat at ~0.55 from about epoch 24.
- Section 6 says "Dice handles this" (background vs loss). This was never measured.

## Repository Structure

```
CropSeg/
├── notebooks/
│   └── training.ipynb    # Complete ML pipeline (EDA → Training → Evaluation → Inference)
├── README.md
├── README_original.md    # Earlier README, kept for reference (contains uncorrected figures)
└── .gitignore
```

## Notebook Contents

The notebook (`notebooks/training.ipynb`) contains the full ML development lifecycle:

1. **Environment Setup & Data Acquisition** — Mount Drive, install dependencies, download PhenoBench with progress bar
2. **Exploratory Data Analysis (EDA)** — Class distribution (200-image sample, earliest date only), per-image statistics, sample diversity visualization
3. **Data Preprocessing & Augmentation** — Custom Dataset class, label merging, augmentation pipeline (horizontal/vertical flips, 90° rotations, brightness/contrast, color jitter)
4. **Augmentation Preview** — Visual demonstration of random transforms applied to training samples
5. **Model Architecture** — U-Net + ResNet34 definition
6. **Training Configuration** — Loss functions, optimizer, scheduler, metric definitions
7. **Training Loop** — 40-epoch training with Google Drive checkpointing and progress tracking
8. **Training Analysis** — Loss and IoU convergence curves
9. **Model Evaluation** — Confusion matrix, dataset-level per-class IoU bar chart
10. **Qualitative Results** — Side-by-side prediction grid (image / ground truth / prediction)
11. **Failure Analysis** — Worst predictions by per-image mIoU
12. **Inference on Unseen Images** — Upload an image and predict; the saved demo uses `phenoBench_00685.png` from the test split

## Trained Weights

The trained model checkpoint (~280 MB) is hosted on Google Drive:

**[Download Trained Weights](https://drive.google.com/drive/folders/1fGvRF82Xw8xv0RZ5-UVTLnWPkJk7pJnB?usp=sharing)**

The `checkpoints` folder contains:
- `best_model.pt` — Best checkpoint (highest batch-averaged validation mIoU, epoch 39). It stores `epoch` (0-based, 38), `val_loss` (0.1437), `val_miou` (0.8225) and `config`.
- `epoch_10.pt` through `epoch_40.pt` — Intermediate checkpoints
- `training_curves.png` — Loss/IoU plots
- `prediction_grid.png` — Visual results

## Setup & Usage

### Option 1: Run on Google Colab (Recommended)

1. Open `notebooks/training.ipynb` in Google Colab
2. The notebook automatically:
   - Mounts Google Drive
   - Downloads PhenoBench (~7.6 GB, first time only)
   - Trains the model
3. To **evaluate the pretrained weights without training**:
   - Put `best_model.pt` in `MyDrive/CropSeg/checkpoints/`
   - Run Sections 1–4, skip Sections 5–6, then run Section 7 onward. Section 5 is the training cell; Section 6 is the training-curves cell, which needs the training history. Section 7 loads `best_model.pt` directly.
   - Don't run the training cell while `best_model.pt` is present. It resumes from that checkpoint, trains another epoch, and can overwrite the file.
4. Use the final "Inference on Unseen Images" cell to predict on new images

### Option 2: Local Inference Only

```bash
pip install torch torchvision segmentation-models-pytorch opencv-python numpy
```

```python
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Load model
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
checkpoint = torch.load("best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Preprocess exactly as in training: RGB, resize to 512, ImageNet normalisation
image_rgb = cv2.cvtColor(cv2.imread("your_image.png"), cv2.COLOR_BGR2RGB)
h, w = image_rgb.shape[:2]
x = cv2.resize(image_rgb, (512, 512)).astype(np.float32) / 255.0
x = (x - IMAGENET_MEAN) / IMAGENET_STD
x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

with torch.no_grad():
    mask = model(x).argmax(dim=1).squeeze(0).numpy().astype(np.uint8)
mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

# mask values: 0=Soil, 1=Crop, 2=Weed
```

## Technologies

- **PyTorch** — Deep learning framework
- **Segmentation Models PyTorch** — U-Net implementation with pretrained encoders
- **Albumentations** — Image augmentation
- **OpenCV** — Image I/O and preprocessing
- **NumPy, scikit-learn, Matplotlib** — Metrics, confusion matrix, plots
- **Google Colab** — T4 GPU training environment
- **PhenoBench** — UAV crop/weed segmentation benchmark dataset

## References

- Weyler, J., et al. "PhenoBench: A Large Dataset and Benchmarks for Semantic Image Interpretation in the Agricultural Domain." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024.
- Ronneberger, O., Fischer, P., & Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*, 2015.
- He, K., et al. "Deep Residual Learning for Image Recognition." *CVPR*, 2016.
