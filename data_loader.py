from pathlib import Path
import pickle

import numpy as np


def load_dataset(
    dataset_dir: str | Path,
    split: str,
    expected_features: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed hand-landmark features and labels.

    Supports both the original 'data' key and dataset-specific
    keys such as 'MyDataset' and 'RoboflowDataset'.

    Samples with incompatible feature dimensions are excluded.
    """

    dataset_dir = Path(dataset_dir)
    file_path = dataset_dir / f"{split}_data.pickle"

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid split: {split}")

    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with file_path.open("rb") as file:
        dataset = pickle.load(file)

    if not isinstance(dataset, dict) or "labels" not in dataset:
        raise ValueError(f"Invalid dataset structure: {file_path}")

    feature_keys = [key for key in dataset if key != "labels"]

    if len(feature_keys) != 1:
        raise ValueError(
            f"Expected exactly one feature key, found: {feature_keys}"
        )

    features = dataset[feature_keys[0]]
    labels = dataset["labels"]

    if len(features) != len(labels):
        raise ValueError("Feature and label counts do not match.")

    valid_samples = [
        (feature, label)
        for feature, label in zip(features, labels)
        if len(feature) == expected_features
    ]

    if not valid_samples:
        raise ValueError("No valid samples found.")

    X, y = zip(*valid_samples)

    return np.asarray(X, dtype=np.float64), np.asarray(y)