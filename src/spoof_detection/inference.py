"""Inference utilities for spoof detection."""

import torch
from PIL import Image


def predict_image(model, processor, image_path):
    """Predict whether a facial image is real or spoofed.

    Args:
        model: A fine-tuned classification model.
        processor: The corresponding image processor.
        image_path: Path to the image file.

    Returns:
        The predicted label string ("real" or "spoof").
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]
