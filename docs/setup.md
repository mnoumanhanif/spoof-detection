# Setup Guide

This guide covers how to set up the Spoof Detection project for local development or Google Colab.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Recommended) NVIDIA GPU with CUDA support for training

## Local Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mnoumanhanif/spoof-detection.git
cd spoof-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

For running the project:

```bash
pip install -r requirements.txt
```

For development (includes testing and linting tools):

```bash
pip install -r requirements-dev.txt
```

### 4. Verify Installation

```bash
python -c "from src.spoof_detection import __version__; print(__version__)"
```

## Google Colab Setup

The Jupyter notebook in `notebooks/` is designed for Google Colab:

1. Upload `notebooks/Spoof_Detection_ViT_Swin.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run all cells sequentially — dependencies install automatically.

## Configuration

Training parameters can be customized in `configs/training_config.yaml`:

```yaml
training:
  batch_size: 16
  epochs: 3
  learning_rate: 0.0002
```

See [development.md](development.md) for more details on customization.
