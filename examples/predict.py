"""Example: Run inference with a trained spoof detection model.

This script demonstrates how to use a fine-tuned model to predict
whether a facial image is real or spoofed.

Usage:
    python examples/predict.py <image_path>
"""

import sys

from src.spoof_detection.inference import predict_image
from src.spoof_detection.models import load_vit_model


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Loading model...")
    model, processor = load_vit_model()

    print(f"Analyzing: {image_path}")
    prediction = predict_image(model, processor, image_path)
    print(f"Prediction: {prediction.upper()}")


if __name__ == "__main__":
    main()
