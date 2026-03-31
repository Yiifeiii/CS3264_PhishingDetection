import argparse
import math
import random
import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
CLASS_NAMES = ("real", "fake")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split data/raw/{real,fake} into data/sglip/{train,val,test}/{real,fake}."
    )
    parser.add_argument("--source-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/sglip"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned split without copying files.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio, val_ratio, test_ratio):
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }

    if any(value < 0 for value in ratios.values()):
        raise ValueError("Split ratios must be non-negative.")

    total = sum(ratios.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )

    return ratios


def collect_images(class_dir):
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")

    images = sorted(
        path for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        raise ValueError(f"No supported images found under {class_dir}")

    return images


def compute_split_counts(total, ratios):
    exact = {split: total * ratio for split, ratio in ratios.items()}
    counts = {split: int(math.floor(value)) for split, value in exact.items()}
    remainder = total - sum(counts.values())

    ranked_splits = sorted(
        exact,
        key=lambda split: (exact[split] - counts[split], ratios[split]),
        reverse=True,
    )

    for split_name in ranked_splits[:remainder]:
        counts[split_name] += 1

    return counts


def assign_splits(paths, ratios, rng):
    shuffled = list(paths)
    rng.shuffle(shuffled)

    counts = compute_split_counts(len(shuffled), ratios)
    train_end = counts["train"]
    val_end = train_end + counts["val"]

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def ensure_output_root_is_safe(output_root):
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_root}. "
            "Remove it first or choose a different --output-root."
        )


def next_available_path(destination_dir, source_path):
    candidate = destination_dir / source_path.name
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = destination_dir / f"{source_path.stem}_{counter}{source_path.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def copy_split_files(assignments, output_root, dry_run):
    summary = {}

    for split_name, class_mapping in assignments.items():
        summary[split_name] = {}
        for class_name, paths in class_mapping.items():
            destination_dir = output_root / split_name / class_name
            summary[split_name][class_name] = len(paths)

            if dry_run:
                continue

            destination_dir.mkdir(parents=True, exist_ok=True)
            for source_path in paths:
                destination_path = next_available_path(destination_dir, source_path)
                shutil.copy2(source_path, destination_path)

    return summary


def print_summary(summary, output_root, dry_run):
    mode = "Dry run" if dry_run else "Completed"
    print(f"{mode} dataset split for {output_root}")
    for split_name in ("train", "val", "test"):
        split_counts = summary[split_name]
        total = sum(split_counts.values())
        print(
            f"{split_name}: total={total} "
            f"(real={split_counts['real']}, fake={split_counts['fake']})"
        )


def main():
    args = parse_args()
    ratios = validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    ensure_output_root_is_safe(args.output_root)

    rng = random.Random(args.seed)
    assignments = {}

    for split_name in ("train", "val", "test"):
        assignments[split_name] = {}

    for class_name in CLASS_NAMES:
        class_paths = collect_images(args.source_root / class_name)
        class_split = assign_splits(class_paths, ratios, rng)
        for split_name, split_paths in class_split.items():
            assignments[split_name][class_name] = split_paths

    summary = copy_split_files(assignments, args.output_root, args.dry_run)
    print_summary(summary, args.output_root, args.dry_run)


if __name__ == "__main__":
    main()
