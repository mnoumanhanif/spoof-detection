# Architecture

This document explains the technical architecture of the Spoof Detection system.

## Overview

The system classifies facial images as **Real** or **Spoofed** using transformer-based models. It compares two architectures:

1. **Vision Transformer (ViT)** — global self-attention over image patches.
2. **Swin Transformer** — hierarchical attention with shifted windows.

## Pipeline

```
┌──────────────┐     ┌───────────────┐     ┌─────────────┐     ┌────────────┐
│   Dataset     │────▶│  Preprocessor  │────▶│   Model      │────▶│  Evaluation │
│  (HuggingFace)│     │  (Processor)   │     │  (ViT/Swin)  │     │  (Metrics)  │
└──────────────┘     └───────────────┘     └─────────────┘     └────────────┘
```

### 1. Dataset Loading (`data.py`)

- Loads the CelebA-Spoof dataset from Hugging Face.
- Creates a balanced subset (equal real and spoof samples).
- Cleans corrupted images.
- Splits into train/test sets (80/20).

### 2. Preprocessing (`models.py`)

- Each model has a corresponding image processor.
- Processors handle resizing (224×224), normalization, and tensor conversion.
- The `create_transform_fn()` function generates dataset-compatible transforms.

### 3. Model Training (`training.py`)

- Uses Hugging Face's `Trainer` API for training.
- Evaluates with accuracy, precision, recall, and F1-score.
- Supports configurable hyperparameters via `configs/training_config.yaml`.

### 4. Inference (`inference.py`)

- Loads a fine-tuned model.
- Accepts a single image path.
- Returns the predicted label: `"real"` or `"spoof"`.

## Model Details

### Vision Transformer (ViT)

- **Checkpoint:** `google/vit-base-patch16-224-in21k`
- **Approach:** Splits images into 16×16 patches, applies global self-attention.
- **Strengths:** Captures global relationships across the entire image.

### Swin Transformer

- **Checkpoint:** `microsoft/swin-base-patch4-window7-224-in22k`
- **Approach:** Uses 4×4 patches with hierarchical 7×7 window attention.
- **Strengths:** More efficient attention; better at capturing local features.

## Module Structure

```
src/spoof_detection/
├── __init__.py       # Package metadata and version
├── data.py           # Dataset loading, cleaning, and splitting
├── models.py         # Model and processor loading
├── training.py       # Training loop and metrics
└── inference.py      # Single-image prediction
```

## Key Design Decisions

1. **Balanced Dataset:** Equal sampling of real and spoof images prevents class bias.
2. **Data Cleaning:** Corrupted images are filtered before training.
3. **Transfer Learning:** Both models are fine-tuned from ImageNet-pretrained checkpoints.
4. **Evaluation Metrics:** Multiple metrics (not just accuracy) provide a fuller picture of performance.
