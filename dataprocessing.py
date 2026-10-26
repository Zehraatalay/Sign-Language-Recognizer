from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import mediapipe as mp

from feature_extraction import extract_hand_features


SPLITS = ("train", "validation", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_split(
    dataset_dir: Path,
    split: str,
    hands,
    output_path: Path,
) -> None:
    """Extract hand-landmark features from one dataset split."""

    split_dir = dataset_dir / split

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Existing processed data will not be overwritten."
        )

    data = []
    labels = []

    total_images = 0
    skipped_images = 0

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.iterdir()):
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in IMAGE_EXTENSIONS
            ):
                continue

            total_images += 1

            image = cv2.imread(str(image_path))

            if image is None:
                skipped_images += 1
                print(f"[SKIP] Cannot read image: {image_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if not results.multi_hand_landmarks:
                skipped_images += 1
                continue

            # The classifiers were trained on 42 features (one hand).
            # Use the first detected hand to keep the feature size fixed.
            hand_landmarks = results.multi_hand_landmarks[0]

            features = extract_hand_features(hand_landmarks)

            data.append(features.tolist())
            labels.append(class_dir.name)

    output = {
        "data": data,
        "labels": labels,
    }

    with output_path.open("wb") as file:
        pickle.dump(output, file)

    print(f"\nSplit: {split}")
    print(f"Total images: {total_images}")
    print(f"Processed images: {len(data)}")
    print(f"Skipped images: {skipped_images}")
    print(f"Saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from image datasets."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing train, validation and test folders.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where processed pickle files will be saved.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check all output paths before starting processing.
    output_paths = {
        split: args.output_dir / f"{split}_data.pickle"
        for split in SPLITS
    }

    existing_files = [
        path for path in output_paths.values() if path.exists()
    ]

    if existing_files:
        raise FileExistsError(
            "Processing cancelled. Existing output files:\n"
            + "\n".join(str(path) for path in existing_files)
        )

    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    ) as hands:
        for split in SPLITS:
            process_split(
                dataset_dir=args.dataset_dir,
                split=split,
                hands=hands,
                output_path=output_paths[split],
            )


if __name__ == "__main__":
    main()