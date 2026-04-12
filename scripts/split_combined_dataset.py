from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config


UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split data/combined into train/val/test folders for DistilBERT fine-tuning."
    )
    parser.add_argument(
        "--input-dir",
        default="data/combined",
        help="Directory containing fake/ and real/ image folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/combined_split",
        help="Output directory for train/val/test folders and manifests.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="How to materialize split files in the output directory.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete the output directory before writing the new split.",
    )
    return parser.parse_args()


def ensure_valid_ratios(args: argparse.Namespace) -> None:
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {args.train_ratio} + "
            f"{args.val_ratio} + {args.test_ratio} = {total}"
        )


def infer_source_key(path: Path) -> str:
    stem = path.stem.lower()

    if stem.startswith("social_"):
        return "social"
    if stem.startswith("img_"):
        return "chat"
    if stem.startswith("screenshot"):
        return "chat_screenshot"
    if UUID_PATTERN.fullmatch(stem):
        return "chat_uuid"
    if stem.startswith("pmo_newsroom"):
        return "pmo_newsroom"
    if stem.startswith("pmo_photos"):
        return "pmo_photos"
    if stem.startswith("mothershipsg"):
        return "mothershipsg"
    if stem.startswith("channelnewsasia"):
        return "channelnewsasia"
    if stem.startswith("straits_times_site"):
        return "straits_times_site"
    if stem.startswith("straits_times"):
        return "straits_times"
    if re.fullmatch(r"neg\d+", stem) or re.fullmatch(r"pos\d+", stem):
        return "legacy_sample"
    if stem.startswith("neg") or stem.startswith("pos"):
        return "legacy_sample"

    generic = re.sub(r"[_-]?(fake|real|neg|pos)\b", "", stem)
    generic = re.sub(r"[_-]?\d+$", "", generic)
    generic = re.sub(r"[_-]+", "_", generic).strip("_")
    return generic or "misc"


def collect_records(root: Path) -> list[dict[str, str]]:
    cfg = Config()
    allowed_exts = {ext.lower() for ext in cfg.SUPPORTED_IMAGE_EXTENSIONS}
    label_dirs = {
        "fake": root / "fake",
        "real": root / "real",
    }
    records: list[dict[str, str]] = []

    for label, directory in label_dirs.items():
        if not directory.exists():
            raise FileNotFoundError(f"Missing label directory: {directory}")

        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in allowed_exts:
                continue
            records.append(
                {
                    "label": label,
                    "path": str(path),
                    "name": path.name,
                    "source": infer_source_key(path),
                }
            )

    if not records:
        raise ValueError(f"No supported images found under {root}")

    return records


def allocate_counts(size: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, int]:
    if size <= 0:
        return {split: 0 for split in SPLIT_NAMES}
    if size == 1:
        return {"train": 1, "val": 0, "test": 0}

    raw_counts = {
        "train": size * train_ratio,
        "val": size * val_ratio,
        "test": size * test_ratio,
    }
    counts = {split: int(math.floor(raw_counts[split])) for split in SPLIT_NAMES}
    counts["train"] = max(counts["train"], 1)

    remainder = size - sum(counts.values())
    ranked_splits = sorted(
        SPLIT_NAMES,
        key=lambda split: (raw_counts[split] - counts[split], raw_counts[split]),
        reverse=True,
    )
    for split in ranked_splits:
        if remainder <= 0:
            break
        counts[split] += 1
        remainder -= 1

    while sum(counts.values()) > size:
        for split in ("test", "val", "train"):
            if counts[split] > 0 and sum(counts.values()) > size:
                counts[split] -= 1

    if size >= 3:
        if counts["val"] == 0 and val_ratio > 0:
            donor = "train" if counts["train"] > 1 else ("test" if counts["test"] > 1 else None)
            if donor:
                counts[donor] -= 1
                counts["val"] += 1
        if counts["test"] == 0 and test_ratio > 0:
            donor = "train" if counts["train"] > 1 else ("val" if counts["val"] > 1 else None)
            if donor:
                counts[donor] -= 1
                counts["test"] += 1

    return counts


def assign_splits(
    records: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[(record["label"], record["source"])].append(record)

    assigned: list[dict[str, str]] = []
    for key in sorted(grouped):
        group_records = list(grouped[key])
        rng.shuffle(group_records)
        counts = allocate_counts(len(group_records), train_ratio, val_ratio, test_ratio)

        cursor = 0
        for split in SPLIT_NAMES:
            next_cursor = cursor + counts[split]
            for record in group_records[cursor:next_cursor]:
                assigned.append({**record, "split": split})
            cursor = next_cursor

    assigned.sort(key=lambda row: (row["split"], row["label"], row["source"], row["name"]))
    return assigned


def clear_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def materialize_record(record: dict[str, str], output_dir: Path, copy_mode: str) -> str:
    source = Path(record["path"])
    destination = output_dir / record["split"] / record["label"] / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)

    if copy_mode == "hardlink":
        if destination.exists():
            destination.unlink()
        destination.hardlink_to(source)
    else:
        shutil.copy2(source, destination)

    return str(destination)


def build_summary(rows: list[dict[str, str]], args: argparse.Namespace, output_dir: Path) -> dict:
    split_label_counts: dict[str, dict[str, int]] = {
        split: {"fake": 0, "real": 0}
        for split in SPLIT_NAMES
    }
    split_source_counts: dict[str, dict[str, int]] = {
        split: {}
        for split in SPLIT_NAMES
    }

    for row in rows:
        split_label_counts[row["split"]][row["label"]] += 1
        split_sources = split_source_counts[row["split"]]
        split_sources[row["source"]] = split_sources.get(row["source"], 0) + 1

    return {
        "input_dir": args.input_dir,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "total_images": len(rows),
        "split_label_counts": split_label_counts,
        "split_source_counts": split_source_counts,
    }


def main() -> int:
    args = parse_args()
    ensure_valid_ratios(args)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")

    if output_dir.exists() and not args.overwrite_output:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Use --overwrite-output to replace it."
        )

    clear_output_dir(output_dir)

    records = collect_records(input_dir)
    assigned_rows = assign_splits(
        records,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    manifest_rows: list[dict[str, str]] = []
    for row in assigned_rows:
        split_path = materialize_record(row, output_dir, args.copy_mode)
        manifest_rows.append(
            {
                "split": row["split"],
                "label": row["label"],
                "source": row["source"],
                "name": row["name"],
                "original_path": row["path"],
                "split_path": split_path,
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "label", "source", "name", "original_path", "split_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = build_summary(manifest_rows, args, output_dir)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved split dataset to: {output_dir.resolve()}")
    for split in SPLIT_NAMES:
        counts = summary["split_label_counts"][split]
        print(
            f"{split}: total={counts['fake'] + counts['real']} "
            f"fake={counts['fake']} real={counts['real']}"
        )
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
