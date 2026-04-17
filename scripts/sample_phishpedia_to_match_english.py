from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config


PREFERRED_IMAGE_NAMES = (
    "shot.png",
    "shot.jpg",
    "shot.jpeg",
    "shot.webp",
    "shot.bmp",
    "shot.avif",
    "shot1.png",
    "shot1.jpg",
    "shot1.jpeg",
    "shot1.webp",
    "shot1.bmp",
    "shot1.avif",
)


@dataclass(frozen=True)
class ExampleCandidate:
    source_key: str
    source_dir: str
    source_path: Path
    source_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a Phishpedia train/val/test split that matches the per-split class counts "
            "from a reference split such as data/english_data_split."
        )
    )
    parser.add_argument(
        "--reference-split-dir",
        default="data/english_data_split",
        help="Reference split directory whose train/val/test counts should be matched.",
    )
    parser.add_argument(
        "--phishing-source-dir",
        default="data/phishpedia_phishing",
        help="Phishpedia phishing source directory.",
    )
    parser.add_argument(
        "--benign-source-dir",
        default="data/phishpedia_benign",
        help="Phishpedia benign source directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/phishpedia_matched_split",
        help="Output split directory to create.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed used for deterministic sampling.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Delete and rebuild the output directory if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned counts without copying files.",
    )
    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    cfg = Config()
    allowed_exts = {ext.lower() for ext in cfg.SUPPORTED_IMAGE_EXTENSIONS}
    return sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_exts
    )


def read_reference_counts(reference_split_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for label in ("fake", "real"):
            label_dir = reference_split_dir / split / label
            if not label_dir.exists():
                raise FileNotFoundError(f"Missing reference label directory: {label_dir}")
            counts[split][label] = len(collect_images(label_dir))
    return counts


def pick_representative_image(candidate_dir: Path) -> Path | None:
    name_to_path = {
        path.name.lower(): path
        for path in candidate_dir.iterdir()
        if path.is_file()
    }
    for preferred_name in PREFERRED_IMAGE_NAMES:
        preferred_path = name_to_path.get(preferred_name)
        if preferred_path is not None:
            return preferred_path

    cfg = Config()
    allowed_exts = {ext.lower() for ext in cfg.SUPPORTED_IMAGE_EXTENSIONS}
    image_paths = sorted(
        path for path in candidate_dir.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_exts
    )
    if image_paths:
        return image_paths[0]
    return None


def collect_phishpedia_candidates(root: Path) -> list[ExampleCandidate]:
    if not root.exists():
        raise FileNotFoundError(f"Missing source directory: {root}")

    cfg = Config()
    allowed_exts = {ext.lower() for ext in cfg.SUPPORTED_IMAGE_EXTENSIONS}
    candidates: list[ExampleCandidate] = []

    for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if child.is_dir():
            source_path = pick_representative_image(child)
            if source_path is None:
                continue
            candidates.append(
                ExampleCandidate(
                    source_key=child.name,
                    source_dir=str(child),
                    source_path=source_path,
                    source_name=child.name,
                )
            )
            continue

        if child.is_file() and child.suffix.lower() in allowed_exts:
            candidates.append(
                ExampleCandidate(
                    source_key=child.stem,
                    source_dir=str(root),
                    source_path=child,
                    source_name=child.stem,
                )
            )

    if not candidates:
        raise ValueError(f"No usable Phishpedia screenshot candidates found in: {root}")
    return candidates


def allocate_split_candidates(
    candidates: list[ExampleCandidate],
    counts_by_split: dict[str, int],
    seed: int,
) -> dict[str, list[ExampleCandidate]]:
    total_needed = sum(int(count) for count in counts_by_split.values())
    if len(candidates) < total_needed:
        raise ValueError(
            f"Not enough candidates to satisfy requested counts: need {total_needed}, have {len(candidates)}."
        )

    shuffled = list(candidates)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    allocations: dict[str, list[ExampleCandidate]] = {}
    cursor = 0
    for split in ("train", "val", "test"):
        count = int(counts_by_split[split])
        allocations[split] = shuffled[cursor:cursor + count]
        cursor += count
    return allocations


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned[:60] or "sample"


def build_destination_name(split: str, label: str, index: int, candidate: ExampleCandidate) -> str:
    source_hash = hashlib.sha1(str(candidate.source_path).encode("utf-8")).hexdigest()[:12]
    safe_stem = sanitize_filename(candidate.source_name)
    return f"{split}_{label}_{index:04d}_{safe_stem}_{source_hash}{candidate.source_path.suffix.lower()}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Use --overwrite-output-dir to rebuild it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    reference_split_dir = Path(args.reference_split_dir)
    phishing_source_dir = Path(args.phishing_source_dir)
    benign_source_dir = Path(args.benign_source_dir)
    output_dir = Path(args.output_dir)

    if not reference_split_dir.exists():
        raise FileNotFoundError(f"Missing reference split directory: {reference_split_dir}")

    reference_counts = read_reference_counts(reference_split_dir)
    phishing_candidates = collect_phishpedia_candidates(phishing_source_dir)
    benign_candidates = collect_phishpedia_candidates(benign_source_dir)

    allocations = {
        "fake": allocate_split_candidates(
            phishing_candidates,
            {split: reference_counts[split]["fake"] for split in ("train", "val", "test")},
            seed=args.seed,
        ),
        "real": allocate_split_candidates(
            benign_candidates,
            {split: reference_counts[split]["real"] for split in ("train", "val", "test")},
            seed=args.seed + 1,
        ),
    }

    print("Reference counts:")
    for split in ("train", "val", "test"):
        print(
            f"- {split}: fake={reference_counts[split]['fake']} "
            f"real={reference_counts[split]['real']}"
        )
    print(f"Phishpedia phishing candidates: {len(phishing_candidates)}")
    print(f"Phishpedia benign candidates: {len(benign_candidates)}")

    if args.dry_run:
        print("Dry run requested; no files copied.")
        return 0

    ensure_output_dir(output_dir, overwrite=args.overwrite_output_dir)

    manifest_rows: list[dict[str, object]] = []
    summary = {
        "reference_split_dir": str(reference_split_dir),
        "phishing_source_dir": str(phishing_source_dir),
        "benign_source_dir": str(benign_source_dir),
        "seed": args.seed,
        "counts": reference_counts,
        "copied_counts": {"train": {"fake": 0, "real": 0}, "val": {"fake": 0, "real": 0}, "test": {"fake": 0, "real": 0}},
    }

    for label in ("fake", "real"):
        for split in ("train", "val", "test"):
            destination_dir = output_dir / split / label
            destination_dir.mkdir(parents=True, exist_ok=True)
            split_candidates = allocations[label][split]
            for index, candidate in enumerate(split_candidates, start=1):
                destination_name = build_destination_name(split, label, index, candidate)
                destination_path = destination_dir / destination_name
                shutil.copy2(candidate.source_path, destination_path)
                manifest_rows.append(
                    {
                        "split": split,
                        "label": label,
                        "source_group": candidate.source_name,
                        "source_dir": candidate.source_dir,
                        "source_path": str(candidate.source_path),
                        "destination_path": str(destination_path),
                    }
                )
            summary["copied_counts"][split][label] = len(split_candidates)

    write_csv(output_dir / "manifest.csv", manifest_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Created sampled split at: {output_dir.resolve()}")
    for split in ("train", "val", "test"):
        print(
            f"- {split}: fake={summary['copied_counts'][split]['fake']} "
            f"real={summary['copied_counts'][split]['real']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
