"""Example: Train a ViT model for spoof detection.

This script demonstrates how to use the spoof_detection modules
to train a Vision Transformer model on the CelebA-Spoof dataset.

Usage:
    python examples/train_vit.py
"""

from src.spoof_detection.data import load_spoof_dataset, run_diagnostics
from src.spoof_detection.models import create_transform_fn, load_vit_model
from src.spoof_detection.training import get_training_args, train_model


def main():
    # Step 1: Load and prepare dataset
    print("Loading dataset...")
    dataset = load_spoof_dataset(subset_size=4000)
    run_diagnostics(dataset)

    # Step 2: Load ViT model and processor
    print("Loading ViT model...")
    model, processor = load_vit_model()

    # Step 3: Preprocess dataset
    print("Preprocessing dataset...")
    transform_fn = create_transform_fn(processor)
    prepared = dataset.map(
        transform_fn, batched=True, remove_columns=["image", "spoof"]
    )

    # Step 4: Train
    print("Starting training...")
    args = get_training_args(output_dir="./vit-spoof-detector")
    trainer, results = train_model(
        model, args, prepared["train"], prepared["test"]
    )

    # Step 5: Print results
    print("\n--- ViT Training Results ---")
    for metric, value in results.items():
        if metric.startswith("eval_"):
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
