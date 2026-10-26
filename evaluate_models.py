from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from data_loader import load_dataset


MODEL_FILES = {
    "Random Forest": "model.p",
    "KNN": "model_knn.p",
    "SVM": "svm_model.p",
    "Logistic Regression": "logistic_model.p",
    "Decision Tree": "decision_tree_model.p",
    "Naive Bayes": "naive_bayes_model.p",
}


def evaluate_models(
    dataset_dir: Path,
    split: str = "test",
    models_dir: Path | None = None,
) -> list[dict]:
    """Evaluate saved classifiers on a preprocessed dataset split."""

    # Use the dataset directory if no separate model directory is provided.
    if models_dir is None:
        models_dir = dataset_dir

    # Load preprocessed features and labels.
    X, y = load_dataset(dataset_dir, split)

    results = []

    for model_name, filename in MODEL_FILES.items():
        model_path = models_dir / filename

        # Skip models that are not available.
        if not model_path.is_file():
            print(f"[SKIP] {model_name}: model file not found.")
            continue

        # Load the pretrained model.
        with model_path.open("rb") as file:
            saved = pickle.load(file)

        model = saved["model"] if isinstance(saved, dict) else saved

        # Generate predictions.
        predictions = model.predict(X)

        # Calculate evaluation metrics.
        results.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y, predictions),
                "macro_precision": precision_score(
                    y,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "macro_recall": recall_score(
                    y,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "macro_f1": f1_score(
                    y,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
            }
        )

    # Rank models by macro F1 score.
    return sorted(
        results,
        key=lambda row: row["macro_f1"],
        reverse=True,
    )


def save_results(results: list[dict], output_path: Path) -> None:
    """Save evaluation results as a CSV file."""

    if not results:
        raise ValueError("No evaluation results to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained sign language classifiers."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing preprocessed dataset splits.",
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory containing saved models. Defaults to dataset directory.",
    )

    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )

    args = parser.parse_args()

    # Evaluate models using the specified dataset and model directories.
    results = evaluate_models(
        dataset_dir=args.dataset_dir,
        split=args.split,
        models_dir=args.models_dir,
    )

    if not results:
        print("No models were evaluated.")
        return

    print(f"\nDataset: {args.dataset_dir.name}")
    print(f"Split: {args.split}\n")

    print(
        f"{'Model':<23}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'Macro F1':>12}"
    )

    print("-" * 71)

    for result in results:
        print(
            f"{result['model']:<23}"
            f"{result['accuracy']:>12.4f}"
            f"{result['macro_precision']:>12.4f}"
            f"{result['macro_recall']:>12.4f}"
            f"{result['macro_f1']:>12.4f}"
        )

    if args.output:
        save_results(results, args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()