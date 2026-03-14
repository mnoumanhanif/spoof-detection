"""Tests for the spoof_detection.data module."""

import pytest

data = pytest.importorskip(
    "src.spoof_detection.data",
    reason="ML dependencies (datasets) not installed",
)


class TestLabelMappings:
    """Tests for label configuration constants."""

    def test_labels_contains_real_and_spoof(self):
        assert data.LABELS == ["real", "spoof"]

    def test_label2id_mapping(self):
        assert data.LABEL2ID == {"real": 0, "spoof": 1}

    def test_id2label_mapping(self):
        assert data.ID2LABEL == {0: "real", 1: "spoof"}

    def test_label_mappings_are_consistent(self):
        for label, idx in data.LABEL2ID.items():
            assert data.ID2LABEL[idx] == label


class TestDefaultConfig:
    """Tests for default dataset configuration values."""

    def test_default_dataset_name(self):
        assert (
            data.DEFAULT_DATASET
            == "nguyenkhoa/celeba-spoof-for-face-antispoofing-test"
        )

    def test_default_subset_size(self):
        assert data.DEFAULT_SUBSET_SIZE == 4000

    def test_default_test_size(self):
        assert data.DEFAULT_TEST_SIZE == 0.2

    def test_default_seed(self):
        assert data.DEFAULT_SEED == 42
