"""Dataset loading and preparation for spoof detection."""

from datasets import Image, concatenate_datasets, load_dataset


# Label mappings
LABELS = ["real", "spoof"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# Default dataset configuration
DEFAULT_DATASET = "nguyenkhoa/celeba-spoof-for-face-antispoofing-test"
DEFAULT_SUBSET_SIZE = 4000
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42


def load_spoof_dataset(
    dataset_name=DEFAULT_DATASET,
    subset_size=DEFAULT_SUBSET_SIZE,
    test_size=DEFAULT_TEST_SIZE,
    seed=DEFAULT_SEED,
):
    """Load and prepare the spoof detection dataset.

    Args:
        dataset_name: Hugging Face dataset identifier.
        subset_size: Total number of samples (split evenly between real/spoof).
        test_size: Fraction of data reserved for testing.
        seed: Random seed for reproducibility.

    Returns:
        A DatasetDict with 'train' and 'test' splits.
    """
    full_dataset = load_dataset(dataset_name, split="test")

    half = subset_size // 2
    real_samples = (
        full_dataset.filter(lambda x: x["labels"] == 0)
        .shuffle(seed=seed)
        .select(range(half))
    )
    spoof_samples = (
        full_dataset.filter(lambda x: x["labels"] == 1)
        .shuffle(seed=seed)
        .select(range(half))
    )

    combined = concatenate_datasets([real_samples, spoof_samples]).shuffle(seed=seed)

    # Remove corrupted images
    cleaned = combined.filter(lambda x: x["cropped_image"] is not None)

    dataset = cleaned.train_test_split(test_size=test_size)
    dataset = dataset.rename_column("cropped_image", "image")
    dataset = dataset.rename_column("labels", "spoof")
    dataset = dataset.cast_column("image", Image())

    return dataset


def run_diagnostics(dataset, split="train", num_samples=500):
    """Check data quality by validating image samples.

    Args:
        dataset: The dataset to check.
        split: Which split to validate.
        num_samples: Number of samples to check.

    Returns:
        True if no problems found, False otherwise.
    """
    print(f"--- Running Data Diagnostics on the '{split}' split ---")
    problem_found = False

    for i, example in enumerate(dataset[split]):
        if i >= num_samples:
            break

        image_data = example["image"]

        if image_data is None:
            print(f"!!! PROBLEM at index {i}: Image data is None.")
            problem_found = True
            break
        if not hasattr(image_data, "convert"):
            print(
                f"!!! PROBLEM at index {i}: Not a valid image. "
                f"Type: {type(image_data)}"
            )
            problem_found = True
            break

    if not problem_found:
        print(
            f"--- Diagnostics Complete: No problems found in "
            f"the first {num_samples} samples. ---"
        )
    else:
        print("--- Diagnostics Complete: Data corruption identified. ---")

    return not problem_found
