"""Training utilities for spoof detection models."""

import numpy as np
import evaluate
from transformers import DefaultDataCollator, Trainer, TrainingArguments

# Default training hyperparameters
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_SAVE_STEPS = 100
DEFAULT_EVAL_STEPS = 100
DEFAULT_LOGGING_STEPS = 10


def get_compute_metrics_fn():
    """Create a metrics computation function for the Trainer.

    Returns:
        A function that computes accuracy, precision, recall, and F1.
    """
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(
                predictions=predictions, references=labels
            )["accuracy"],
            "precision": precision_metric.compute(
                predictions=predictions, references=labels
            )["precision"],
            "recall": recall_metric.compute(
                predictions=predictions, references=labels
            )["recall"],
            "f1": f1_metric.compute(
                predictions=predictions, references=labels
            )["f1"],
        }

    return compute_metrics


def get_training_args(
    output_dir,
    batch_size=DEFAULT_BATCH_SIZE,
    epochs=DEFAULT_EPOCHS,
    learning_rate=DEFAULT_LEARNING_RATE,
):
    """Create training arguments for the Trainer.

    Args:
        output_dir: Directory to save model checkpoints.
        batch_size: Training batch size per device.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.

    Returns:
        A TrainingArguments instance.
    """
    return TrainingArguments(
        output_dir=output_dir,
        report_to="none",
        per_device_train_batch_size=batch_size,
        eval_strategy="steps",
        num_train_epochs=epochs,
        save_steps=DEFAULT_SAVE_STEPS,
        eval_steps=DEFAULT_EVAL_STEPS,
        logging_steps=DEFAULT_LOGGING_STEPS,
        learning_rate=learning_rate,
        save_total_limit=2,
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )


def train_model(model, training_args, train_dataset, eval_dataset):
    """Train and evaluate a model using the Hugging Face Trainer.

    Args:
        model: The model to train.
        training_args: TrainingArguments instance.
        train_dataset: Preprocessed training dataset.
        eval_dataset: Preprocessed evaluation dataset.

    Returns:
        Tuple of (trainer, eval_results).
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        compute_metrics=get_compute_metrics_fn(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    results = trainer.evaluate()
    return trainer, results
