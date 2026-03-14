# 🧠 Spoof Detection using Vision Transformers

A facial spoof detection system that classifies images as **Real** or **Spoofed** using **Vision Transformer (ViT)** and **Swin Transformer** architectures, enhancing the security of AI-powered facial authentication.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements a binary classification system that detects presentation attacks (spoofing) on biometric facial recognition systems. It compares two state-of-the-art transformer architectures to identify spoofing cues such as texture inconsistencies, reflections, and illumination artifacts.

**Key finding:** Swin Transformer showed superior accuracy and faster convergence compared to ViT.

## Key Features

- **Binary classification** of facial images (Real vs. Spoof)
- **Two transformer architectures** — ViT and Swin Transformer
- **Transfer learning** from ImageNet-pretrained checkpoints
- **Comprehensive evaluation** with Accuracy, Precision, Recall, and F1-score
- **Interactive testing** with user-uploaded images
- **Modular codebase** for easy extension and reuse

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch |
| Models | ViT (`google/vit-base-patch16-224-in21k`), Swin (`microsoft/swin-base-patch4-window7-224-in22k`) |
| Model Hub | Hugging Face Transformers |
| Dataset | Hugging Face Datasets |
| Metrics | scikit-learn, Hugging Face Evaluate |
| Visualization | Matplotlib |

## Project Structure

```
spoof-detection/
├── src/spoof_detection/    # Core Python modules
│   ├── __init__.py         #   Package metadata
│   ├── data.py             #   Dataset loading and preparation
│   ├── models.py           #   Model and processor loading
│   ├── training.py         #   Training loop and metrics
│   └── inference.py        #   Single-image prediction
├── notebooks/              # Jupyter notebooks
│   └── Spoof_Detection_ViT_Swin.ipynb
├── tests/                  # Unit tests
├── configs/                # Configuration files
│   └── training_config.yaml
├── docs/                   # Documentation
│   ├── setup.md
│   ├── architecture.md
│   └── development.md
├── examples/               # Usage examples
│   ├── train_vit.py
│   └── predict.py
├── .github/                # CI/CD and templates
│   ├── workflows/ci.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- (Recommended) NVIDIA GPU with CUDA for training

### Quick Start

```bash
# Clone the repository
git clone https://github.com/mnoumanhanif/spoof-detection.git
cd spoof-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Google Colab

1. Upload `notebooks/Spoof_Detection_ViT_Swin.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run all cells — dependencies install automatically.

## Usage

### Using the Jupyter Notebook

The notebook provides a complete pipeline: data loading → training → evaluation → interactive testing.

```bash
jupyter notebook notebooks/Spoof_Detection_ViT_Swin.ipynb
```

### Using the Python Modules

```python
from src.spoof_detection.data import load_spoof_dataset
from src.spoof_detection.models import load_vit_model, create_transform_fn
from src.spoof_detection.training import train_model, get_training_args

# Load and prepare data
dataset = load_spoof_dataset(subset_size=4000)

# Load model
model, processor = load_vit_model()

# Preprocess and train
transform_fn = create_transform_fn(processor)
prepared = dataset.map(transform_fn, batched=True, remove_columns=["image", "spoof"])
args = get_training_args(output_dir="./vit-spoof-detector")
trainer, results = train_model(model, args, prepared["train"], prepared["test"])
```

### Inference

```python
from src.spoof_detection.inference import predict_image
from src.spoof_detection.models import load_vit_model

model, processor = load_vit_model()
prediction = predict_image(model, processor, "path/to/face.jpg")
print(f"Prediction: {prediction}")  # "real" or "spoof"
```

See the `examples/` directory for complete runnable scripts.

## Dataset

**[CelebA-Spoof](https://huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing-test)** from Hugging Face — contains real and spoofed facial images under various lighting and presentation attack conditions.

- **Subset size:** 4,000 images (2,000 real + 2,000 spoof)
- **Split:** 80% training, 20% testing
- **Preprocessing:** Corrupted images are filtered automatically

## Models

### Vision Transformer (ViT)

- Splits images into 16×16 patches and applies global self-attention
- Captures high-level visual representations across the full image

### Swin Transformer

- Uses 4×4 patches with hierarchical 7×7 window attention
- More efficient; better at capturing local spoofing features

## Development

### Install Dev Dependencies

```bash
pip install -r requirements-dev.txt
```

### Code Style

```bash
black src/ tests/           # Format code
isort src/ tests/           # Sort imports
flake8 src/ tests/ --max-line-length=100  # Lint
```

See [docs/development.md](docs/development.md) for the full development guide.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/spoof_detection --cov-report=term-missing
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Developed as part of the **Generative AI (Fall 2025)** course.
- Built with [Hugging Face](https://huggingface.co/) and [PyTorch](https://pytorch.org/).
- Dataset by [nguyenkhoa](https://huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing-test).

---

If you find this work helpful, please ⭐ the repository and consider citing it.
