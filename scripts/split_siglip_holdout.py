import argparse
import json
import random
import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
CLASS_NAMES = ("real", "fake")
SPLIT_NAMES = ("train", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic fixed train/test holdout split from one or more "
            "labeled source roots. Each source root must contain {real,fake} subfolders."
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
        default="data/sglip_holdout",
        help="Directory to write the split into.",
    )
    parser.add_argument(
        "--test-total",
        type=int,
        default=None,
        help="Number of fixed test images across all source/class buckets.",
    )
    parser.add_argument(
        "--test-per-bucket",
        type=int,
        default=None,
        help=(
            "Number of fixed test images per source/class bucket. Use this instead of "
            "--test-total when you want direct control."
        ),
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Whether to copy files or create symlinks inside the split directories.",
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
        help="Print the split plan without copying files.",
    )
    return parser.parse_args()


def resolve_holdout_per_bucket(total, per_bucket, bucket_count, default_per_bucket):
    if total is not None and per_bucket is not None:
        raise ValueError("Specify only one of --test-total or --test-per-bucket.")

    if total is not None:
        if total <= 0:
            raise ValueError(f"--test-total must be positive, got {total}.")
        if total % bucket_count != 0:
            raise ValueError(
                f"--test-total={total} is not divisible by {bucket_count} buckets."
            )
        return total // bucket_count

    if per_bucket is not None:
        if per_bucket <= 0:
            raise ValueError(
                f"--test-per-bucket must be positive, got {per_bucket}."
            )
        return per_bucket

    return default_per_bucket


def collect_class_files(source_root: Path, class_name: str):
    class_dir = source_root / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Expected class directory not found: {class_dir}")

    files = [
        path.resolve()
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def resolve_labeled_root(source_root: Path):
    direct_ok = all((source_root / class_name).exists() for class_name in CLASS_NAMES)
    if direct_ok:
        return source_root

    raw_child = source_root / "raw"
    raw_ok = raw_child.exists() and all(
        (raw_child / class_name).exists() for class_name in CLASS_NAMES
    )
    if raw_ok:
        return raw_child

    raise FileNotFoundError(
        f"Could not find {{real,fake}} under {source_root} or {raw_child}."
    )


def ensure_output_dirs(output_root: Path):
    for split_name in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            (output_root / split_name / class_name).mkdir(parents=True, exist_ok=True)


def build_destination_name(source_root: Path, source_file: Path):
    relative_parent = source_file.parent.relative_to(source_root)
    if relative_parent == Path("."):
        relative_prefix = ""
    else:
        relative_prefix = "__".join(relative_parent.parts) + "__"
    return f"{source_root.name}__{relative_prefix}{source_file.name}"


def write_entry(source_file: Path, destination_path: Path, copy_mode: str):
    if copy_mode == "copy":
        shutil.copy2(source_file, destination_path)
        return

    destination_path.symlink_to(source_file)


def relative_paths(paths, base_dir: Path):
    return [str(path.relative_to(base_dir)) for path in paths]


def main():
    args = parse_args()

    source_roots = [Path(root).resolve() for root in args.source_root]
    output_root = Path(args.output_root).resolve()

    labeled_roots = []
    for source_root in source_roots:
        if not source_root.exists():
            raise FileNotFoundError(f"Source root does not exist: {source_root}")
        labeled_roots.append(resolve_labeled_root(source_root))

    bucket_count = len(source_roots) * len(CLASS_NAMES)
    test_per_bucket = resolve_holdout_per_bucket(
        total=args.test_total,
        per_bucket=args.test_per_bucket,
        bucket_count=bucket_count,
        default_per_bucket=10,
    )

    if output_root.exists() and args.clear_output and not args.dry_run:
        shutil.rmtree(output_root)

    if not args.dry_run:
        ensure_output_dirs(output_root)

    rng = random.Random(args.seed)
    manifest_buckets = []
    split_plan = {
        split_name: {class_name: [] for class_name in CLASS_NAMES}
        for split_name in SPLIT_NAMES
    }

    for source_root, labeled_root in zip(source_roots, labeled_roots):
        for class_name in CLASS_NAMES:
            files = collect_class_files(labeled_root, class_name)
            shuffled = files[:]
            rng.shuffle(shuffled)

            if len(shuffled) <= test_per_bucket:
                raise ValueError(
                    f"{source_root.name}/{class_name} has {len(shuffled)} files, not enough "
                    f"to reserve {test_per_bucket} test images."
                )

            test_files = shuffled[:test_per_bucket]
            train_files = shuffled[test_per_bucket:]

            split_plan["train"][class_name].extend(
                (source_root, file_path) for file_path in train_files
            )
            split_plan["test"][class_name].extend(
                (source_root, file_path) for file_path in test_files
            )

            manifest_buckets.append(
                {
                    "source_name": source_root.name,
                    "class_name": class_name,
                    "source_root": str(source_root),
                    "labeled_root": str(labeled_root),
                    "total_count": len(shuffled),
                    "train_count": len(train_files),
                    "test_count": len(test_files),
                    "train_files": relative_paths(train_files, labeled_root),
                    "test_files": relative_paths(test_files, labeled_root),
                }
            )

    print("Fixed holdout split plan:")
    print(f"  sources: {', '.join(root.name for root in source_roots)}")
    print(f"  buckets: {bucket_count}")
    print(f"  fixed test per bucket: {test_per_bucket}")
    print(f"  fixed test total: {test_per_bucket * bucket_count}")
    print(f"  seed: {args.seed}")
    for split_name in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            print(f"  {split_name}/{class_name}: {len(split_plan[split_name][class_name])}")

    manifest = {
        "sources": [str(source_root) for source_root in source_roots],
        "output_root": str(output_root),
        "copy_mode": args.copy_mode,
        "seed": args.seed,
        "test_per_bucket": test_per_bucket,
        "test_total": test_per_bucket * bucket_count,
        "buckets": manifest_buckets,
    }

    if args.dry_run:
        print("Dry run only. No files were copied.")
        return

    for split_name, class_mapping in split_plan.items():
        for class_name, entries in class_mapping.items():
            destination_dir = output_root / split_name / class_name
            for source_root, source_file in entries:
                destination_name = build_destination_name(source_root, source_file)
                destination_path = destination_dir / destination_name
                write_entry(source_file, destination_path, args.copy_mode)

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created split at {output_root}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
