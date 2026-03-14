"""Tests for the spoof_detection.models module."""

import pytest

models = pytest.importorskip(
    "src.spoof_detection.models",
    reason="ML dependencies (transformers) not installed",
)


class TestModelCheckpoints:
    """Tests for model checkpoint constants."""

    def test_vit_checkpoint(self):
        assert models.VIT_CHECKPOINT == "google/vit-base-patch16-224-in21k"

    def test_swin_checkpoint(self):
        assert models.SWIN_CHECKPOINT == "microsoft/swin-base-patch4-window7-224-in22k"
