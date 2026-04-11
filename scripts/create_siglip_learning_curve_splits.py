import argparse
import json
import random
import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
CLASS_NAMES = ("real", "fake")
SPLIT_NAMES = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create fixed-holdout, source-balanced learning-curve splits for SigLIP. "
            "Each source root must contain both {real,fake} subfolders."
        )
    )
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        help=(
            "Labeled source dataset root. Repeat this flag to merge multiple sources, "
            "for example: --source-root data/chat --source-root data/social"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="data/learning_curve",
        help="Root directory where all experiment splits will be created.",
    )
    parser.add_argument(
        "--train-sizes",
        required=True,
        help=(
            "Comma-separated total training sizes. For equal source/class balance, "
            "each size must be divisible by (#sources * 2 classes)."
        ),
    )
    parser.add_argument(
        "--val-per-bucket",
        type=int,
        default=None,
        help="Validation images per (source, class) bucket.",
    )
    parser.add_argument(
        "--val-total",
        type=int,
        default=None,
        help=(
            "Total validation images across all source/class buckets. "
            "Must be divisible by (#sources * 2 classes)."
        ),
    )
    parser.add_argument(
        "--test-per-bucket",
        type=int,
        default=None,
        help="Test images per (source, class) bucket.",
    )
    parser.add_argument(
        "--test-total",
        type=int,
        default=None,
        help=(
            "Total test images across all source/class buckets. "
            "Must be divisible by (#sources * 2 classes)."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated training subsets per size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for fixed holdouts and training subset sampling.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the existing output root before writing new splits.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without copying files.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def parse_train_sizes(raw_sizes: str):
    sizes = []
    for item in raw_sizes.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        size = int(stripped)
        if size <= 0:
            raise ValueError(f"Training sizes must be positive integers, got {size}.")
        sizes.append(size)

    if not sizes:
        raise ValueError("At least one training size must be provided.")
    if sizes != sorted(set(sizes)):
        raise ValueError("Training sizes must be unique and sorted in ascending order.")
    return sizes


def collect_class_files(source_root: Path, class_name: str):
    class_dir = source_root / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Expected class directory not found: {class_dir}")

    files = [
        path for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def build_destination_name(source_root: Path, source_file: Path):
    relative_parent = source_file.parent.relative_to(source_root)
    if relative_parent == Path("."):
        relative_prefix = ""
    else:
        relative_prefix = "__".join(relative_parent.parts) + "__"
    return f"{source_root.name}__{relative_prefix}{source_file.name}"


def ensure_split_dirs(root: Path):
    for split_name in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            (root / split_name / class_name).mkdir(parents=True, exist_ok=True)


def copy_split_entries(split_root: Path, entries_by_split):
    ensure_split_dirs(split_root)
    for split_name in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            destination_dir = split_root / split_name / class_name
            for source_root, source_file in entries_by_split[split_name][class_name]:
                destination_path = destination_dir / build_destination_name(source_root, source_file)
                shutil.copy2(source_file, destination_path)


def summarize_counts(entries_by_split):
    summary = {}
    for split_name in SPLIT_NAMES:
        summary[split_name] = {
            class_name: len(entries_by_split[split_name][class_name])
            for class_name in CLASS_NAMES
        }
    return summary


def resolve_holdout_per_bucket(total, per_bucket, bucket_count, split_name, default_per_bucket):
    if total is not None and per_bucket is not None:
        raise ValueError(
            f"Specify only one of --{split_name}-total or --{split_name}-per-bucket."
        )

    if total is not None:
        if total <= 0:
            raise ValueError(f"--{split_name}-total must be positive, got {total}.")
        if total % bucket_count != 0:
            raise ValueError(
                f"--{split_name}-total={total} is not divisible by {bucket_count} buckets. "
                f"Use a total that preserves equal source/class balance."
            )
        return total // bucket_count

    if per_bucket is not None:
        if per_bucket <= 0:
            raise ValueError(
                f"--{split_name}-per-bucket must be positive, got {per_bucket}."
            )
        return per_bucket

    return default_per_bucket


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    source_roots = [resolve_path(project_root, raw_path) for raw_path in args.source_root]
    output_root = resolve_path(project_root, args.output_root)
    train_sizes = parse_train_sizes(args.train_sizes)

    for source_root in source_roots:
        if not source_root.exists():
            raise FileNotFoundError(f"Source root does not exist: {source_root}")

    source_files = {}
    for source_root in source_roots:
        class_files = {
            class_name: collect_class_files(source_root, class_name)
            for class_name in CLASS_NAMES
        }
        missing_classes = [class_name for class_name, files in class_files.items() if not files]
        if missing_classes:
            missing_labels = ", ".join(missing_classes)
            raise ValueError(
                f"Source {source_root.name} is missing labeled data for: {missing_labels}. "
                "All sources must contain both real and fake examples for a balanced learning curve."
            )
        source_files[source_root] = class_files

    bucket_count = len(source_roots) * len(CLASS_NAMES)
    val_per_bucket = resolve_holdout_per_bucket(
        total=args.val_total,
        per_bucket=args.val_per_bucket,
        bucket_count=bucket_count,
        split_name="val",
        default_per_bucket=10,
    )
    test_per_bucket = resolve_holdout_per_bucket(
        total=args.test_total,
        per_bucket=args.test_per_bucket,
        bucket_count=bucket_count,
        split_name="test",
        default_per_bucket=10,
    )
    per_bucket_train_sizes = []
    for total_size in train_sizes:
        if total_size % bucket_count != 0:
            raise ValueError(
                f"Training size {total_size} is not divisible by {bucket_count} buckets "
                f"({len(source_roots)} sources x {len(CLASS_NAMES)} classes). "
                "Use sizes that preserve equal source/class balance."
            )
        per_bucket_train_sizes.append(total_size // bucket_count)

    holdout_rng = random.Random(args.seed)
    buckets = {}
    for source_root, class_mapping in source_files.items():
        for class_name, files in class_mapping.items():
            shuffled = files[:]
            holdout_rng.shuffle(shuffled)

            required = val_per_bucket + test_per_bucket
            if len(shuffled) <= required:
                raise ValueError(
                    f"Source {source_root.name}/{class_name} has {len(shuffled)} files, "
                    f"which is not enough for val={val_per_bucket} and test={test_per_bucket} per bucket."
                )

            test_files = shuffled[:test_per_bucket]
            val_files = shuffled[test_per_bucket:required]
            pool_files = shuffled[required:]

            max_needed = per_bucket_train_sizes[-1]
            if len(pool_files) < max_needed:
                raise ValueError(
                    f"Source {source_root.name}/{class_name} has only {len(pool_files)} training-pool files "
                    f"after holdout, but the largest requested per-bucket training size is {max_needed}."
                )

            buckets[(source_root, class_name)] = {
                "test": test_files,
                "val": val_files,
                "pool": pool_files,
            }

    if output_root.exists() and args.clear_output and not args.dry_run:
        shutil.rmtree(output_root)

    manifest = {
        "sources": [str(source_root) for source_root in source_roots],
        "train_sizes_total": train_sizes,
        "train_sizes_per_bucket": per_bucket_train_sizes,
        "val_per_bucket": val_per_bucket,
        "test_per_bucket": test_per_bucket,
        "val_total": val_per_bucket * bucket_count,
        "test_total": test_per_bucket * bucket_count,
        "repeats": args.repeats,
        "seed": args.seed,
    }

    print("Learning-curve plan:")
    print(f"  sources: {', '.join(source_root.name for source_root in source_roots)}")
    print(f"  buckets: {bucket_count}")
    print(f"  total train sizes: {train_sizes}")
    print(f"  per-bucket train sizes: {per_bucket_train_sizes}")
    print(f"  val per bucket: {val_per_bucket}")
    print(f"  test per bucket: {test_per_bucket}")
    print(f"  val total: {val_per_bucket * bucket_count}")
    print(f"  test total: {test_per_bucket * bucket_count}")
    print(f"  repeats: {args.repeats}")

    experiments = []
    for repeat_idx in range(args.repeats):
        repeat_number = repeat_idx + 1
        repeat_rng = random.Random(args.seed + 10_000 + repeat_idx)
        sampled_pools = {}
        for bucket_key, split_mapping in buckets.items():
            pool_files = split_mapping["pool"][:]
            repeat_rng.shuffle(pool_files)
            sampled_pools[bucket_key] = pool_files

        for total_size, per_bucket_size in zip(train_sizes, per_bucket_train_sizes):
            experiment_name = f"size_{total_size:04d}/run_{repeat_number:02d}"
            split_root = output_root / experiment_name
            entries_by_split = {
                split_name: {class_name: [] for class_name in CLASS_NAMES}
                for split_name in SPLIT_NAMES
            }

            for (source_root, class_name), split_mapping in buckets.items():
                entries_by_split["test"][class_name].extend(
                    (source_root, source_file) for source_file in split_mapping["test"]
                )
                entries_by_split["val"][class_name].extend(
                    (source_root, source_file) for source_file in split_mapping["val"]
                )
                entries_by_split["train"][class_name].extend(
                    (source_root, source_file)
                    for source_file in sampled_pools[(source_root, class_name)][:per_bucket_size]
                )

            summary = summarize_counts(entries_by_split)
            experiments.append(
                {
                    "name": experiment_name,
                    "total_train_size": total_size,
                    "per_bucket_train_size": per_bucket_size,
                    "repeat": repeat_number,
                    "counts": summary,
                }
            )

            print(f"  {experiment_name}: {summary}")

            if not args.dry_run:
                copy_split_entries(split_root, entries_by_split)

    manifest["experiments"] = experiments

    if args.dry_run:
        print("Dry run only. No files were copied.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created learning-curve splits at {output_root}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
