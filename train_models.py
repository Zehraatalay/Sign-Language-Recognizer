from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_dataset


MODEL_FILES = {
    "random_forest": "model.p",
    "knn": "model_knn.p",
    "svm": "svm_model.p",
    "logistic_regression": "logistic_model.p",
    "decision_tree": "decision_tree_model.p",
    "naive_bayes": "naive_bayes_model.p",
}


def create_model(model_name: str, random_state: int | None = None):
    """Create a classifier using the original project configuration."""

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=random_state,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=3,
            weights="uniform",
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=random_state,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
        ),
        "decision_tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            random_state=random_state,
        ),
        "naive_bayes": GaussianNB(
            var_smoothing=1e-9,
        ),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return models[model_name]


def train_model(
    model_name: str,
    X_train,
    y_train,
    X_validation,
    y_validation,
    random_state: int | None = None,
):
    """Train a classifier and evaluate it on the validation split."""

    model = create_model(model_name, random_state=random_state)

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    accuracy = accuracy_score(y_validation, predictions)
    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )

    return model, accuracy, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train classical ML models for ASL alphabet recognition."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing preprocessed dataset splits.",
    )

    parser.add_argument(
        "--model",
        choices=[*MODEL_FILES.keys(), "all"],
        default="all",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("trained_models"),
        help="Directory for newly trained models.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for supported estimators.",
    )

    args = parser.parse_args()

    X_train, y_train = load_dataset(args.dataset_dir, "train")
    X_validation, y_validation = load_dataset(
        args.dataset_dir, "validation"
    )

    selected_models = (
        list(MODEL_FILES)
        if args.model == "all"
        else [args.model]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        name: args.output_dir / MODEL_FILES[name]
        for name in selected_models
    }

    existing_files = [
        path for path in output_paths.values() if path.exists()
    ]

    if existing_files:
        raise FileExistsError(
            "Training cancelled. The following model files already exist:\n"
            + "\n".join(str(path) for path in existing_files)
        )
    print(f"\nDataset: {args.dataset_dir.name}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")

    for model_name in selected_models:
        print(f"\nTraining: {model_name}")

        model, accuracy, macro_f1 = train_model(
            model_name,
            X_train,
            y_train,
            X_validation,
            y_validation,
            random_state=args.seed,
        )

        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Validation macro F1: {macro_f1:.4f}")
        output_path = output_paths[model_name]
        with output_path.open("wb") as file:
            pickle.dump({"model": model}, file)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()