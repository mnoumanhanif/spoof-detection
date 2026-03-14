# Development Guide

This guide covers development workflows, testing, and code quality for the Spoof Detection project.

## Project Layout

```
spoof-detection/
├── src/spoof_detection/    # Core Python modules
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
├── configs/                # Configuration files
├── docs/                   # Documentation
├── examples/               # Usage examples
├── scripts/                # Helper scripts
├── .github/                # GitHub Actions and templates
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
└── pyproject.toml          # Project and tool configuration
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/spoof_detection --cov-report=term-missing

# Run a specific test file
pytest tests/test_data.py
```

## Code Style

This project uses:

- **Black** for formatting (line length: 88)
- **isort** for import ordering
- **flake8** for linting (max line length: 100)

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check code
flake8 src/ tests/ --max-line-length=100
```

## Training Models

### Using the Notebook

The easiest way to train is via the Jupyter notebook on Google Colab:

1. Open `notebooks/Spoof_Detection_ViT_Swin.ipynb` in Colab.
2. Enable GPU runtime.
3. Run all cells.

### Using the Python Modules

```python
from src.spoof_detection.data import load_spoof_dataset
from src.spoof_detection.models import load_vit_model, create_transform_fn
from src.spoof_detection.training import train_model, get_training_args

# Load data
dataset = load_spoof_dataset()

# Load model
model, processor = load_vit_model()

# Prepare data
transform_fn = create_transform_fn(processor)
prepared = dataset.map(transform_fn, batched=True, remove_columns=["image", "spoof"])

# Train
args = get_training_args(output_dir="./vit-spoof-detector")
trainer, results = train_model(model, args, prepared["train"], prepared["test"])
print(results)
```

## Configuration

Training parameters are defined in `configs/training_config.yaml`. You can modify:

- `dataset.subset_size` — number of training samples
- `training.batch_size` — batch size per GPU
- `training.epochs` — number of training epochs
- `training.learning_rate` — optimizer learning rate

## Adding New Models

To add a new transformer model:

1. Add a loader function in `src/spoof_detection/models.py`.
2. Update `configs/training_config.yaml` with the new checkpoint.
3. Add tests in `tests/test_models.py`.

## CI/CD

GitHub Actions runs automatically on pushes and PRs to `main`:

- **Lint job:** Checks formatting with Black, isort, and flake8.
- **Test job:** Runs the full test suite with pytest.
