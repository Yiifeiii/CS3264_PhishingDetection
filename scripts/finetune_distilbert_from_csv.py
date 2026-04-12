from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config


LABEL_ALIASES = {
    "0": 0,
    "real": 0,
    "ham": 0,
    "benign": 0,
    "legit": 0,
    "legitimate": 0,
    "non-phishing": 0,
    "non_phishing": 0,
    "not phishing": 0,
    "1": 1,
    "fake": 1,
    "spam": 1,
    "scam": 1,
    "phishing": 1,
    "smishing": 1,
    "malicious": 1,
}

LABEL_ID_TO_NAME = {0: "real", 1: "fake"}


def parse_args() -> argparse.Namespace:
    cfg = Config()
    default_base_model = "models/text_phishing_base"
    if not (PROJECT_ROOT / default_base_model).exists():
        default_base_model = cfg.TEXT_PHISHING_MODEL_NAME

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune DistilBERT from a CSV text dataset with a sampled "
            "train/validation/test split. Saves weights into a separate folder."
        )
    )
    parser.add_argument(
        "--csv-path",
        default="data/df.csv",
        help="CSV file containing labeled text rows.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="CSV column containing model input text.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="CSV column containing binary labels.",
    )
    parser.add_argument(
        "--base-model",
        default=default_base_model,
        help="Local path or Hugging Face id for the starting DistilBERT checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/distilbert_df_sample10",
        help="Directory for saved model, split CSVs, and summary artifacts.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.10,
        help="Fraction of usable rows to sample before splitting. Use 1.0 for all rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and splitting.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of sampled rows used for training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of sampled rows used for validation.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for fine-tuning.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for fine-tuning.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer max sequence length.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=5,
        help="Minimum stripped text length required to keep a row.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def maximize_csv_field_size() -> None:
    limit = sys.maxsize
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Use --overwrite-output-dir to reuse it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def normalize_label(raw_label: object) -> int:
    text = str(raw_label).strip().lower()
    if text in LABEL_ALIASES:
        return LABEL_ALIASES[text]

    try:
        numeric = int(float(text))
    except ValueError as exc:
        raise ValueError(f"Unsupported label value: {raw_label!r}") from exc

    if numeric not in (0, 1):
        raise ValueError(f"Expected binary label 0/1, got: {raw_label!r}")
    return numeric


def load_rows(csv_path: Path, text_column: str, label_column: str, min_text_length: int) -> tuple[list[dict], list[dict]]:
    usable_rows: list[dict] = []
    skipped_rows: list[dict] = []

    maximize_csv_field_size()

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")
        if text_column not in reader.fieldnames:
            raise KeyError(f"Missing text column '{text_column}' in {csv_path}")
        if label_column not in reader.fieldnames:
            raise KeyError(f"Missing label column '{label_column}' in {csv_path}")

        for index, raw_row in enumerate(reader):
            text = str(raw_row.get(text_column) or "").strip()
            row = {
                "row_id": index,
                "label": None,
                "label_name": "",
                "text": text,
            }

            try:
                label = normalize_label(raw_row.get(label_column))
            except ValueError:
                skipped_rows.append(
                    {
                        "row_id": index,
                        "reason": "invalid_label",
                        "label": raw_row.get(label_column),
                        "text_preview": text[:200],
                    }
                )
                continue

            row["label"] = label
            row["label_name"] = LABEL_ID_TO_NAME[label]

            if len(text) < min_text_length:
                skipped_rows.append(
                    {
                        "row_id": index,
                        "reason": "short_text",
                        "label": label,
                        "text_preview": text[:200],
                    }
                )
                continue

            usable_rows.append(row)

    return usable_rows, skipped_rows


def save_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(int(row["label"]) for row in rows)
    return {
        "real": int(counts.get(0, 0)),
        "fake": int(counts.get(1, 0)),
    }


def stratified_sample(rows: list[dict], sample_fraction: float, seed: int) -> list[dict]:
    if not (0 < sample_fraction <= 1.0):
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")
    if sample_fraction >= 1.0:
        return list(rows)

    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)

    sampled: list[dict] = []
    for label, label_rows in by_label.items():
        label_rows = list(label_rows)
        rng.shuffle(label_rows)
        sample_size = max(1, int(round(len(label_rows) * sample_fraction)))
        sample_size = min(sample_size, len(label_rows))
        sampled.extend(label_rows[:sample_size])

    rng.shuffle(sampled)
    return sampled


def allocate_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    raw_counts = [total * ratio for ratio in ratios]
    counts = [int(math.floor(value)) for value in raw_counts]
    remaining = total - sum(counts)

    ranked_indices = sorted(
        range(len(ratios)),
        key=lambda idx: (raw_counts[idx] - counts[idx], ratios[idx]),
        reverse=True,
    )

    for idx in ranked_indices[:remaining]:
        counts[idx] += 1

    positive_targets = [idx for idx, ratio in enumerate(ratios) if ratio > 0]
    if total >= len(positive_targets):
        for idx in positive_targets:
            if counts[idx] > 0:
                continue
            donor = max(
                (candidate for candidate in positive_targets if counts[candidate] > 1),
                key=lambda candidate: counts[candidate],
                default=None,
            )
            if donor is not None:
                counts[donor] -= 1
                counts[idx] += 1

    return counts


def stratified_split(rows: list[dict], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[dict]]:
    test_ratio = 1.0 - train_ratio - val_ratio
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError(
            f"Expected positive train/val/test ratios, got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)

    split_rows = {"train": [], "val": [], "test": []}
    for label_rows in by_label.values():
        label_rows = list(label_rows)
        rng.shuffle(label_rows)
        train_count, val_count, test_count = allocate_counts(
            len(label_rows),
            (train_ratio, val_ratio, test_ratio),
        )

        train_end = train_count
        val_end = train_count + val_count
        split_rows["train"].extend(label_rows[:train_end])
        split_rows["val"].extend(label_rows[train_end:val_end])
        split_rows["test"].extend(label_rows[val_end:val_end + test_count])

    for key in split_rows:
        rng.shuffle(split_rows[key])
    return split_rows


class TokenizedTextDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        texts = [str(row["text"]) for row in rows]
        self.labels = [int(row["label"]) for row in rows]
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        loss = F.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, object]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "positive_predictions": int(sum(y_pred)),
    }


def trainer_metrics_callback(eval_pred) -> dict[str, object]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return compute_binary_metrics(labels.tolist(), preds.tolist())


def build_class_weights(rows: list[dict]) -> torch.Tensor:
    counts = Counter(int(row["label"]) for row in rows)
    total = sum(counts.values())
    class_weights = []
    for label in (0, 1):
        count = counts.get(label, 0)
        if count <= 0:
            class_weights.append(1.0)
        else:
            class_weights.append(total / (2.0 * count))
    return torch.tensor(class_weights, dtype=torch.float)


def build_training_arguments(args: argparse.Namespace, training_dir: Path) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs = {
        "output_dir": str(training_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "report_to": "none",
    }
    if "overwrite_output_dir" in signature.parameters:
        kwargs["overwrite_output_dir"] = True
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def build_trainer(
    model,
    tokenizer,
    args: argparse.Namespace,
    train_dataset: Dataset,
    val_dataset: Dataset,
    class_weights: torch.Tensor,
    training_dir: Path,
):
    training_args = build_training_arguments(args, training_dir)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": trainer_metrics_callback,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    return WeightedTrainer(class_weights=class_weights, **trainer_kwargs)


def logits_to_argmax_metrics(predictions) -> dict[str, object]:
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=1)
    return compute_binary_metrics(labels.tolist(), preds.tolist())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV dataset: {csv_path}")

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite_output_dir)

    model_output_dir = output_dir / "model"
    training_dir = output_dir / "training"
    if Path(args.base_model).resolve() == model_output_dir.resolve():
        raise ValueError("base-model and output model path must be different.")

    usable_rows, skipped_rows = load_rows(
        csv_path=csv_path,
        text_column=args.text_column,
        label_column=args.label_column,
        min_text_length=args.min_text_length,
    )
    if len(usable_rows) < 10:
        raise ValueError(f"Need at least 10 usable rows, found {len(usable_rows)}")

    sampled_rows = stratified_sample(usable_rows, args.sample_fraction, args.seed)
    split_rows = stratified_split(sampled_rows, args.train_ratio, args.val_ratio, args.seed)

    if not split_rows["train"] or not split_rows["val"] or not split_rows["test"]:
        raise ValueError("Train/validation/test split produced an empty split.")

    for split_name, rows in split_rows.items():
        save_rows_csv(output_dir / f"{split_name}_rows.csv", rows)
        print(f"{split_name}: {len(rows)} rows | label counts={label_counts(rows)}")

    save_rows_csv(output_dir / "sampled_rows.csv", sampled_rows)
    save_rows_csv(output_dir / "skipped_rows.csv", skipped_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    train_dataset = TokenizedTextDataset(split_rows["train"], tokenizer, args.max_length)
    val_dataset = TokenizedTextDataset(split_rows["val"], tokenizer, args.max_length)
    test_dataset = TokenizedTextDataset(split_rows["test"], tokenizer, args.max_length)
    class_weights = build_class_weights(split_rows["train"])

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_weights=class_weights,
        training_dir=training_dir,
    )

    train_result = trainer.train()
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    val_predictions = trainer.predict(val_dataset)
    test_predictions = trainer.predict(test_dataset)
    val_metrics = logits_to_argmax_metrics(val_predictions)
    test_metrics = logits_to_argmax_metrics(test_predictions)

    summary = {
        "data_csv": str(csv_path),
        "base_model": args.base_model,
        "output_dir": str(output_dir),
        "sample_fraction": args.sample_fraction,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 6),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "min_text_length": args.min_text_length,
        "total_rows": len(usable_rows) + len(skipped_rows),
        "usable_rows": len(usable_rows),
        "skipped_rows": len(skipped_rows),
        "sampled_rows": len(sampled_rows),
        "usable_label_counts": label_counts(usable_rows),
        "sampled_label_counts": label_counts(sampled_rows),
        "split_sizes": {name: len(rows) for name, rows in split_rows.items()},
        "split_label_counts": {name: label_counts(rows) for name, rows in split_rows.items()},
        "class_weights": [float(value) for value in class_weights.tolist()],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_runtime_metrics": {key: float(value) for key, value in train_result.metrics.items()},
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    write_json(output_dir / "summary.json", summary)

    print(f"Saved fine-tuned model to: {model_output_dir.resolve()}")
    print(
        "Validation metrics: "
        f"accuracy={val_metrics['accuracy']:.4f} "
        f"f1={val_metrics['f1']:.4f}"
    )
    print(
        "Test metrics: "
        f"accuracy={test_metrics['accuracy']:.4f} "
        f"f1={test_metrics['f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
