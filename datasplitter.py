from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "validation", "test")


def split_dataset(
    source_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Split class-organized images into train, validation and test sets."""

    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError(
            "Output directory must be outside the source directory."
        )

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Choose a new directory to avoid mixing dataset splits."
        )

    class_dirs = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name not in SPLITS
    )

    if not class_dirs:
        raise ValueError(f"No class directories found in: {source_dir}")

    rng = random.Random(seed)

    # Validate the input before creating the output directory.
    class_images = {}

    for class_dir in class_dirs:
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            raise ValueError(f"No images found in: {class_dir}")

        class_images[class_dir.name] = images

    output_dir.mkdir(parents=True)

    totals = {split: 0 for split in SPLITS}

    for class_name, images in class_images.items():
        images = images.copy()
        rng.shuffle(images)

        train_end = int(len(images) * 0.8)
        validation_end = int(len(images) * 0.9)

        split_images = {
            "train": images[:train_end],
            "validation": images[train_end:validation_end],
            "test": images[validation_end:],
        }

        for split, files in split_images.items():
            destination = output_dir / split / class_name
            destination.mkdir(parents=True, exist_ok=True)

            for image_path in files:
                shutil.copy2(
                    image_path,
                    destination / image_path.name,
                )

            totals[split] += len(files)

        print(f"Processed class: {class_name} ({len(images)} images)")

    print("\nDataset split completed.")

    for split, count in totals.items():
        print(f"{split.capitalize()}: {count} images")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split class-organized image datasets."
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing one folder per class.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory where dataset splits will be created.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )

    args = parser.parse_args()

    split_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()