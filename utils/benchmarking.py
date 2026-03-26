from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
REAL_LABEL_DIRS = {"real", "0_real"}
FAKE_LABEL_DIRS = {"fake", "1_fake"}


@dataclass(frozen=True)
class DatasetSample:
    path: Path
    label: str


def infer_label_from_path(path: Path) -> str | None:
    for part in path.parts:
        normalized = part.lower()
        if normalized in REAL_LABEL_DIRS:
            return "real"
        if normalized in FAKE_LABEL_DIRS:
            return "fake"
    return None


def discover_labeled_images(dataset_root: str | Path) -> list[DatasetSample]:
    root = Path(dataset_root)
    samples = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        label = infer_label_from_path(path)
        if label is None:
            continue
        samples.append(DatasetSample(path=path, label=label))

    return samples


def summarize_predictions(rows: list[dict]) -> dict:
    total = len(rows)
    tp = sum(1 for row in rows if row["label"] == "fake" and row["prediction"] == "fake")
    tn = sum(1 for row in rows if row["label"] == "real" and row["prediction"] == "real")
    fp = sum(1 for row in rows if row["label"] == "real" and row["prediction"] == "fake")
    fn = sum(1 for row in rows if row["label"] == "fake" and row["prediction"] == "real")

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    fake_probs_real = [
        row["probabilities"]["fake"] for row in rows if row["label"] == "real"
    ]
    fake_probs_fake = [
        row["probabilities"]["fake"] for row in rows if row["label"] == "fake"
    ]

    return {
        "count": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "avg_fake_prob_real": (
            sum(fake_probs_real) / len(fake_probs_real) if fake_probs_real else 0.0
        ),
        "avg_fake_prob_fake": (
            sum(fake_probs_fake) / len(fake_probs_fake) if fake_probs_fake else 0.0
        ),
    }


def write_benchmark_csv(rows: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "path",
        "label",
        "prediction",
        "correct",
        "confidence",
        "real_probability",
        "fake_probability",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "path": row["path"],
                    "label": row["label"],
                    "prediction": row["prediction"],
                    "correct": row["correct"],
                    "confidence": row["confidence"],
                    "real_probability": row["probabilities"]["real"],
                    "fake_probability": row["probabilities"]["fake"],
                }
            )


def format_summary_table(summary_by_model: dict[str, dict]) -> str:
    header = (
        f"{'model':<12} {'count':>5} {'acc':>8} {'prec':>8} {'recall':>8} "
        f"{'f1':>8} {'tp':>4} {'tn':>4} {'fp':>4} {'fn':>4}"
    )
    lines = [header, "-" * len(header)]
    for model_name, summary in summary_by_model.items():
        lines.append(
            f"{model_name:<12} "
            f"{summary['count']:>5} "
            f"{summary['accuracy']:>8.3f} "
            f"{summary['precision']:>8.3f} "
            f"{summary['recall']:>8.3f} "
            f"{summary['f1']:>8.3f} "
            f"{summary['tp']:>4} "
            f"{summary['tn']:>4} "
            f"{summary['fp']:>4} "
            f"{summary['fn']:>4}"
        )
    return "\n".join(lines)
