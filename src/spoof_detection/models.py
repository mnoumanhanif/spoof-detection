"""Model loading and configuration for spoof detection."""

from transformers import (
    AutoImageProcessor,
    SwinForImageClassification,
    ViTForImageClassification,
    ViTImageProcessor,
)

from .data import ID2LABEL, LABEL2ID, LABELS

# Default model checkpoints
VIT_CHECKPOINT = "google/vit-base-patch16-224-in21k"
SWIN_CHECKPOINT = "microsoft/swin-base-patch4-window7-224-in22k"


def load_vit_model(checkpoint=VIT_CHECKPOINT):
    """Load ViT model and processor for spoof detection.

    Args:
        checkpoint: Hugging Face model checkpoint name.

    Returns:
        Tuple of (model, processor).
    """
    processor = ViTImageProcessor.from_pretrained(checkpoint)
    model = ViTForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    return model, processor


def load_swin_model(checkpoint=SWIN_CHECKPOINT):
    """Load Swin Transformer model and processor for spoof detection.

    Args:
        checkpoint: Hugging Face model checkpoint name.

    Returns:
        Tuple of (model, processor).
    """
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = SwinForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    return model, processor


def create_transform_fn(processor):
    """Create a dataset transformation function for a given processor.

    Args:
        processor: An image processor (ViT or Swin).

    Returns:
        A function suitable for use with dataset.map().
    """

    def transform(examples):
        inputs = processor(images=examples["image"], return_tensors="pt")
        inputs["labels"] = examples["spoof"]
        return inputs

    return transform
