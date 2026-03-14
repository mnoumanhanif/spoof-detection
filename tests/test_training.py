"""Tests for the spoof_detection.training module."""

import pytest

training = pytest.importorskip(
    "src.spoof_detection.training",
    reason="ML dependencies (numpy, transformers) not installed",
)


class TestTrainingDefaults:
    """Tests for default training hyperparameters."""

    def test_default_batch_size(self):
        assert training.DEFAULT_BATCH_SIZE == 16

    def test_default_epochs(self):
        assert training.DEFAULT_EPOCHS == 3

    def test_default_learning_rate(self):
        assert training.DEFAULT_LEARNING_RATE == 2e-4

    def test_default_save_steps(self):
        assert training.DEFAULT_SAVE_STEPS == 100

    def test_default_eval_steps(self):
        assert training.DEFAULT_EVAL_STEPS == 100

    def test_default_logging_steps(self):
        assert training.DEFAULT_LOGGING_STEPS == 10
