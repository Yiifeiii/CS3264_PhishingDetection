import argparse
import random
import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
CLASS_NAMES = ("real", "fake")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a SigLIP train/val/test split from one or more labeled source roots. "
            "Each source root must contain {real,fake} subfolders."
        )
    )
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        help=(
            "Labeled source dataset root. Repeat this flag to merge multiple roots, "
            "for example: --source-root data/chat --source-root data/social"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="data/sglip",
        help="Directory to write the split into.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the existing output root before writing the new split.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without copying files.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f}."
        )


def collect_class_files(source_root: Path, class_name: str):
    class_dir = source_root / class_name
    if not class_dir.exists():
        raise FileNotFoundError(
            f"Expected class directory not found: {class_dir}"
        )

    files = [
        path for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def split_items(items, train_ratio, val_ratio):
    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def build_destination_name(source_root: Path, source_file: Path):
    relative_parent = source_file.parent.relative_to(source_root)
    if relative_parent == Path("."):
        relative_prefix = ""
    else:
        relative_prefix = "__".join(relative_parent.parts) + "__"
    return f"{source_root.name}__{relative_prefix}{source_file.name}"


def ensure_output_dirs(output_root: Path):
    for split_name in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            (output_root / split_name / class_name).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    source_roots = [Path(root).resolve() for root in args.source_root]
    output_root = Path(args.output_root).resolve()

    for source_root in source_roots:
        if not source_root.exists():
            raise FileNotFoundError(f"Source root does not exist: {source_root}")

    if output_root.exists() and args.clear_output and not args.dry_run:
        shutil.rmtree(output_root)

    if not args.dry_run:
        ensure_output_dirs(output_root)

    rng = random.Random(args.seed)
    split_plan = {split_name: {class_name: [] for class_name in CLASS_NAMES} for split_name in ("train", "val", "test")}

    for source_root in source_roots:
        for class_name in CLASS_NAMES:
            files = collect_class_files(source_root, class_name)
            rng.shuffle(files)
            grouped_splits = split_items(files, args.train_ratio, args.val_ratio)
            for split_name, split_files in grouped_splits.items():
                split_plan[split_name][class_name].extend(
                    (source_root, file_path) for file_path in split_files
                )

    for split_name in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            print(f"{split_name}/{class_name}: {len(split_plan[split_name][class_name])} files")

    if args.dry_run:
        print("Dry run only. No files were copied.")
        return

    for split_name, class_mapping in split_plan.items():
        for class_name, entries in class_mapping.items():
            destination_dir = output_root / split_name / class_name
            for source_root, source_file in entries:
                destination_name = build_destination_name(source_root, source_file)
                destination_path = destination_dir / destination_name
                shutil.copy2(source_file, destination_path)

    print(f"Created split at {output_root}")


if __name__ == "__main__":
    main()
