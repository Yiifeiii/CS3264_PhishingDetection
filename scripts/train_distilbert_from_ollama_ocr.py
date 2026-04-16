from __future__ import annotations

import argparse
import csv
import inspect
import json
from collections import Counter
from pathlib import Path
import re
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


LABEL_TO_ID = {
    "real": 0,
    "fake": 1,
}

LABEL_ID_TO_NAME = {0: "real", 1: "fake"}

LEADING_WRAPPER_PATTERNS = (
    re.compile(r"^\s*the extracted text is\s*:?\s*", re.IGNORECASE),
    re.compile(r"^\s*the text in the image reads\s*:?\s*", re.IGNORECASE),
    re.compile(r"^\s*the image contains the following text\s*:?\s*", re.IGNORECASE),
    re.compile(r"^\s*extracted text\s*:?\s*", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    cfg = Config()
    default_base_model = "models/text_phishing_base"
    if not (PROJECT_ROOT / default_base_model).exists():
        default_base_model = cfg.TEXT_PHISHING_MODEL_NAME

    parser = argparse.ArgumentParser(
        description=(
            "Build train/val/test text rows from an Ollama OCR CSV plus an existing split manifest, "
            "then fine-tune or evaluate DistilBERT on those OCR texts."
        )
    )
    parser.add_argument(
        "--ocr-csv",
        default="artifacts/ollama_ocr/english_data_ocr_deduped.csv",
        help="Deduplicated Ollama OCR CSV with path,model,text,error columns.",
    )
    parser.add_argument(
        "--manifest-csv",
        default="data/english_data_split/manifest.csv",
        help="Split manifest created by split_combined_dataset.py.",
    )
    parser.add_argument(
        "--base-model",
        default=default_base_model,
        help="Starting checkpoint or existing fine-tuned model to evaluate/continue from.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/distilbert_ollama_ocr_english",
        help="Directory for saved model, split CSVs, and summary artifacts.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Number of fine-tuning epochs. Use 0 to skip training and only evaluate the supplied base model.",
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
        help="Minimum OCR text length required to keep a row.",
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        help="Tune a probability threshold on validation, then evaluate test once using that frozen threshold.",
    )
    parser.add_argument(
        "--auto-threshold-objective",
        choices=["accuracy", "f1", "precision", "recall"],
        default="accuracy",
        help="Metric used when --auto-threshold is enabled.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fixed positive-class probability threshold. Use instead of --auto-threshold.",
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


def normalize_path(value: str) -> str:
    return str(Path(value)).replace("/", "\\").lower()


def clean_ollama_text(text: str) -> str:
    cleaned = str(text or "").strip()
    for pattern in LEADING_WRAPPER_PATTERNS:
        cleaned = pattern.sub("", cleaned).strip()

    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()

    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_manifest_rows(manifest_csv: Path) -> list[dict]:
    maximize_csv_field_size()
    with manifest_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_csv}")
    required = {"split", "label", "original_path"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise KeyError(f"Manifest missing required columns: {sorted(missing)}")
    return rows


def load_ocr_rows(ocr_csv: Path) -> dict[str, dict]:
    maximize_csv_field_size()
    by_path: dict[str, dict] = {}
    with ocr_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = str(row.get("path") or "").strip()
            if not image_path:
                continue
            by_path[normalize_path(image_path)] = row
    if not by_path:
        raise ValueError(f"OCR CSV has no usable rows: {ocr_csv}")
    return by_path


def label_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(int(row["label"]) for row in rows)
    return {
        "real": int(counts.get(0, 0)),
        "fake": int(counts.get(1, 0)),
    }


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


def build_split_rows(
    manifest_rows: list[dict],
    ocr_rows_by_path: dict[str, dict],
    min_text_length: int,
) -> tuple[dict[str, list[dict]], list[dict]]:
    split_rows = {"train": [], "val": [], "test": []}
    skipped_rows: list[dict] = []

    for index, manifest_row in enumerate(manifest_rows):
        split = str(manifest_row.get("split") or "").strip().lower()
        label_name = str(manifest_row.get("label") or "").strip().lower()
        original_path = str(manifest_row.get("original_path") or "").strip()

        if split not in split_rows:
            skipped_rows.append(
                {
                    "row_id": index,
                    "reason": "invalid_split",
                    "original_path": original_path,
                    "label": label_name,
                }
            )
            continue

        if label_name not in LABEL_TO_ID:
            skipped_rows.append(
                {
                    "row_id": index,
                    "reason": "invalid_label",
                    "original_path": original_path,
                    "label": label_name,
                }
            )
            continue

        ocr_row = ocr_rows_by_path.get(normalize_path(original_path))
        if ocr_row is None:
            skipped_rows.append(
                {
                    "row_id": index,
                    "reason": "missing_ocr_row",
                    "original_path": original_path,
                    "label": label_name,
                }
            )
            continue

        error_text = str(ocr_row.get("error") or "").strip()
        if error_text:
            skipped_rows.append(
                {
                    "row_id": index,
                    "reason": "ocr_error",
                    "original_path": original_path,
                    "label": label_name,
                    "ocr_error": error_text,
                }
            )
            continue

        text = clean_ollama_text(str(ocr_row.get("text") or ""))
        if len(text) < min_text_length:
            skipped_rows.append(
                {
                    "row_id": index,
                    "reason": "short_text",
                    "original_path": original_path,
                    "label": label_name,
                    "text_preview": text[:200],
                }
            )
            continue

        split_rows[split].append(
            {
                "row_id": index,
                "split": split,
                "label": LABEL_TO_ID[label_name],
                "label_name": label_name,
                "original_path": original_path,
                "text": text,
            }
        )

    return split_rows, skipped_rows


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


def probabilities_from_predictions(predictions) -> list[float]:
    logits = predictions.predictions
    tensor = torch.tensor(logits, dtype=torch.float)
    probabilities = torch.softmax(tensor, dim=1)[:, 1].tolist()
    return [float(value) for value in probabilities]


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


def pick_best_threshold(records: list[dict[str, object]], objective: str) -> tuple[float, dict[str, float]]:
    scores = [float(record["decision_score"]) for record in records]
    labels = [int(record["true_label"]) for record in records]
    if not scores:
        raise ValueError("No scored rows available for threshold search.")

    candidate_thresholds = sorted(set(scores))
    candidate_thresholds = [0.0] + candidate_thresholds + [max(scores) + 1e-6]

    best_threshold = candidate_thresholds[0]
    best_metrics = {"accuracy": -1.0, "precision": -1.0, "recall": -1.0, "f1": -1.0}
    best_key = (-1.0, -1.0, -1.0, -1.0, 0.0)

    for threshold in candidate_thresholds:
        preds = [1 if score >= threshold else 0 for score in scores]
        metrics = compute_binary_metrics(labels, preds)
        comparison_key = (
            metrics[objective],
            metrics["accuracy"],
            metrics["f1"],
            metrics["precision"],
            -threshold,
        )
        if comparison_key > best_key:
            best_key = comparison_key
            best_threshold = threshold
            best_metrics = metrics

    return float(best_threshold), best_metrics


def evaluate_threshold(records: list[dict[str, object]], threshold: float) -> dict[str, object]:
    y_true = [int(record["true_label"]) for record in records]
    y_pred = [1 if float(record["decision_score"]) >= threshold else 0 for record in records]
    metrics = compute_binary_metrics(y_true, y_pred)
    return {
        "metrics": metrics,
        "rows_used": len(records),
        "positive_predictions": int(sum(y_pred)),
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.auto_threshold and args.threshold is not None:
        raise ValueError("Use either --threshold or --auto-threshold, not both.")

    ocr_csv = Path(args.ocr_csv)
    manifest_csv = Path(args.manifest_csv)
    if not ocr_csv.exists():
        raise FileNotFoundError(f"Missing OCR CSV: {ocr_csv}")
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest CSV: {manifest_csv}")

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite_output_dir)
    model_output_dir = output_dir / "model"
    training_dir = output_dir / "training"
    if Path(args.base_model).resolve() == model_output_dir.resolve():
        raise ValueError("base-model and output model path must be different.")

    manifest_rows = load_manifest_rows(manifest_csv)
    ocr_rows_by_path = load_ocr_rows(ocr_csv)
    split_rows, skipped_rows = build_split_rows(
        manifest_rows=manifest_rows,
        ocr_rows_by_path=ocr_rows_by_path,
        min_text_length=args.min_text_length,
    )

    if not split_rows["train"] or not split_rows["val"] or not split_rows["test"]:
        raise ValueError("Train/validation/test split produced an empty usable split after OCR filtering.")

    for split_name, rows in split_rows.items():
        save_rows_csv(output_dir / f"{split_name}_rows.csv", rows)
        print(f"{split_name}: {len(rows)} rows | label counts={label_counts(rows)}")
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

    train_metrics = {}
    best_checkpoint = None
    if args.epochs > 0:
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
        best_checkpoint = trainer.state.best_model_checkpoint
        train_metrics = {key: float(value) for key, value in train_result.metrics.items()}

        val_predictions = trainer.predict(val_dataset)
        test_predictions = trainer.predict(test_dataset)
        val_metrics = logits_to_argmax_metrics(val_predictions)
        test_metrics = logits_to_argmax_metrics(test_predictions)
    else:
        model_output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(model_output_dir))
        model.save_pretrained(str(model_output_dir))

        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        trainer_kwargs = {
            "model": model,
            "args": TrainingArguments(output_dir=str(training_dir), report_to="none"),
            "data_collator": collator,
        }
        trainer_signature = inspect.signature(Trainer.__init__)
        if "tokenizer" in trainer_signature.parameters:
            trainer_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in trainer_signature.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        eval_trainer = Trainer(**trainer_kwargs)
        val_predictions = eval_trainer.predict(val_dataset)
        test_predictions = eval_trainer.predict(test_dataset)
        val_metrics = logits_to_argmax_metrics(val_predictions)
        test_metrics = logits_to_argmax_metrics(test_predictions)

    val_threshold_records = [
        {
            "true_label": int(row["label"]),
            "decision_score": score,
        }
        for row, score in zip(split_rows["val"], probabilities_from_predictions(val_predictions))
    ]
    test_threshold_records = [
        {
            "true_label": int(row["label"]),
            "decision_score": score,
        }
        for row, score in zip(split_rows["test"], probabilities_from_predictions(test_predictions))
    ]

    if args.auto_threshold:
        tuned_threshold, tuned_val_metrics = pick_best_threshold(
            val_threshold_records,
            args.auto_threshold_objective,
        )
        tuned_test_eval = evaluate_threshold(test_threshold_records, tuned_threshold)
        threshold_source = f"auto-threshold ({args.auto_threshold_objective})"
    elif args.threshold is not None:
        tuned_threshold = float(args.threshold)
        tuned_val_metrics = evaluate_threshold(val_threshold_records, tuned_threshold)["metrics"]
        tuned_test_eval = evaluate_threshold(test_threshold_records, tuned_threshold)
        threshold_source = "command line"
    else:
        tuned_threshold = 0.5
        tuned_val_metrics = evaluate_threshold(val_threshold_records, tuned_threshold)["metrics"]
        tuned_test_eval = evaluate_threshold(test_threshold_records, tuned_threshold)
        threshold_source = "default 0.5"

    summary = {
        "ocr_csv": str(ocr_csv),
        "manifest_csv": str(manifest_csv),
        "base_model": args.base_model,
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "min_text_length": args.min_text_length,
        "manifest_rows": len(manifest_rows),
        "ocr_rows": len(ocr_rows_by_path),
        "skipped_rows": len(skipped_rows),
        "split_sizes": {name: len(rows) for name, rows in split_rows.items()},
        "split_label_counts": {name: label_counts(rows) for name, rows in split_rows.items()},
        "class_weights": [float(value) for value in class_weights.tolist()],
        "best_checkpoint": best_checkpoint,
        "train_runtime_metrics": train_metrics,
        "validation_argmax_metrics": val_metrics,
        "test_argmax_metrics": test_metrics,
        "threshold": tuned_threshold,
        "threshold_source": threshold_source,
        "threshold_objective": args.auto_threshold_objective if args.auto_threshold else None,
        "validation_threshold_metrics": tuned_val_metrics,
        "test_threshold_metrics": tuned_test_eval["metrics"],
    }
    write_json(output_dir / "summary.json", summary)

    print(f"Saved model to: {model_output_dir.resolve()}")
    print(
        "Validation argmax metrics: "
        f"accuracy={val_metrics['accuracy']:.4f} "
        f"f1={val_metrics['f1']:.4f}"
    )
    print(
        "Test argmax metrics: "
        f"accuracy={test_metrics['accuracy']:.4f} "
        f"f1={test_metrics['f1']:.4f}"
    )
    print(
        f"Threshold: {tuned_threshold:.6f} ({threshold_source})"
    )
    print(
        "Validation threshold metrics: "
        f"accuracy={tuned_val_metrics['accuracy']:.4f} "
        f"f1={tuned_val_metrics['f1']:.4f}"
    )
    print(
        "Test threshold metrics: "
        f"accuracy={tuned_test_eval['metrics']['accuracy']:.4f} "
        f"f1={tuned_test_eval['metrics']['f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
