from __future__ import annotations

import argparse
import csv
import inspect
import json
from collections import Counter
from pathlib import Path
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

from ocr.ocr_service import OCRService
from utils.config import Config
from utils.ocr_text_processor import OCRTextProcessor
from utils.ocr_runtime import add_ocr_runtime_args, build_ocr_service
from utils.text_risk_analyzer import TextRiskAnalyzer


LABEL_NAME_TO_ID = {"real": 0, "fake": 1}
LABEL_ID_TO_NAME = {0: "real", 1: "fake"}


def parse_args() -> argparse.Namespace:
    cfg = Config()
    default_base_model = "models/text_phishing_base"
    if not (PROJECT_ROOT / default_base_model).exists():
        default_base_model = cfg.TEXT_PHISHING_MODEL_NAME

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune DistilBERT on a train/val/test split from data/combined_split, "
            "then tune the route-based combined text pipeline on validation and report test metrics."
        )
    )
    parser.add_argument(
        "--split-dir",
        default="data/combined_split",
        help="Directory produced by split_combined_dataset.py.",
    )
    parser.add_argument(
        "--base-model",
        default=default_base_model,
        help="Local path or Hugging Face id for the starting DistilBERT checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/distilbert_route_pipeline",
        help="Directory for saved model, CSVs, and JSON summaries.",
    )
    parser.add_argument(
        "--chinese-policy",
        default="route",
        choices=["route"],
        help="Route policy is fixed to the best-performing pipeline setup.",
    )
    parser.add_argument(
        "--objective",
        default="accuracy",
        choices=["accuracy", "f1", "precision", "recall"],
        help="Metric used to choose the best decision boundary on validation.",
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
        help="Minimum length of prepared model text required to keep a sample.",
    )
    parser.add_argument(
        "--rule-weight",
        type=float,
        default=cfg.TEXT_RULE_WEIGHT,
        help="Fixed TEXT_RULE_WEIGHT for the combined pipeline.",
    )
    parser.add_argument(
        "--model-weight",
        type=float,
        default=cfg.TEXT_MODEL_WEIGHT,
        help="Fixed TEXT_MODEL_WEIGHT for the combined pipeline.",
    )
    parser.add_argument(
        "--grid-search-weights",
        action="store_true",
        help="Search rule/model weight combinations instead of keeping the fixed weights above.",
    )
    parser.add_argument(
        "--rule-weight-step",
        type=float,
        default=0.05,
        help="Grid step for TEXT_RULE_WEIGHT during validation tuning when --grid-search-weights is enabled.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Grid step for MEDIUM_RISK_THRESHOLD during validation tuning.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    add_ocr_runtime_args(parser, cfg)
    return parser.parse_args()


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Use --overwrite-output-dir to reuse it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def collect_images(root: Path, allowed_exts: set[str]) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Missing split directory: {root}")
    return sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_exts
    )


def build_split_rows(
    split_dir: Path,
    label_name: str,
    split_name: str,
    allowed_exts: set[str],
    ocr: OCRService,
    processor: OCRTextProcessor,
    analyzer: TextRiskAnalyzer,
    min_text_length: int,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    skipped: list[dict] = []
    label_id = LABEL_NAME_TO_ID[label_name]
    label_dir = split_dir / split_name / label_name
    files = collect_images(label_dir, allowed_exts)
    print(f"Found {len(files)} image(s) in {label_dir}")

    for image_path in files:
        raw_text = ocr.extract_text(str(image_path))
        processed = processor.process(raw_text)
        processed_text = str(processed.get("text") or "").strip()
        prepared = analyzer.prepare_model_input(processed_text)
        model_text = str(prepared.get("model_input_text") or "").strip()

        row = {
            "split": split_name,
            "label": label_id,
            "label_name": label_name,
            "path": str(image_path),
            "name": image_path.name,
            "processed_action": str(processed.get("action") or ""),
            "contains_chinese": bool(processed.get("contains_chinese")),
            "raw_text": raw_text,
            "processed_text": processed_text,
            "filtered_text": prepared["filtered_text"],
            "model_input_text": prepared["model_input_text"],
            "relevant_chunks": " | ".join(prepared["relevant_chunks"]),
        }

        if len(model_text) < min_text_length:
            skipped.append(row)
            continue

        rows.append(row)

    return rows, skipped


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


class TokenizedTextDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        texts = [str(row["model_input_text"]) for row in rows]
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


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict:
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


def trainer_metrics_callback(eval_pred) -> dict:
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


def logits_to_argmax_metrics(predictions) -> dict:
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=1)
    return compute_binary_metrics(labels.tolist(), preds.tolist())


def analyze_pipeline_rows(rows: list[dict], analyzer: TextRiskAnalyzer) -> list[dict]:
    analyzed_rows: list[dict] = []
    for row in rows:
        result = analyzer.analyze(str(row["processed_text"]))
        analyzed_rows.append(
            {
                "path": row["path"],
                "label": int(row["label"]),
                "score": float(result.get("score") or 0.0),
                "rule_score": float(result.get("rule_score") or 0.0),
                "model_score": None if result.get("model_score") is None else float(result["model_score"]),
                "model_score_raw": None if result.get("model_score_raw") is None else float(result["model_score_raw"]),
                "model_route": result.get("model_route"),
                "filtered_text": result.get("filtered_text") or "",
                "model_input_text": result.get("model_input_text") or "",
            }
        )
    return analyzed_rows


def objective_value(metrics: dict, objective: str) -> float:
    return float(metrics[objective])


def tune_combined_pipeline(
    analyzed_rows: list[dict],
    objective: str,
    fixed_rule_weight: float,
    fixed_model_weight: float,
    rule_weight_step: float,
    threshold_step: float,
    grid_search_weights: bool,
) -> dict:
    if not analyzed_rows:
        raise ValueError("Cannot tune pipeline weights without validation rows.")

    y_true = [int(row["label"]) for row in analyzed_rows]
    best: dict | None = None
    threshold_values = np.arange(0.05, 0.951, threshold_step)
    if grid_search_weights:
        rule_steps = max(int(round(1.0 / rule_weight_step)), 1)
        weight_pairs = []
        for idx in range(rule_steps + 1):
            rule_weight = round(idx * rule_weight_step, 6)
            if rule_weight > 1.0:
                rule_weight = 1.0
            model_weight = round(1.0 - rule_weight, 6)
            weight_pairs.append((rule_weight, model_weight))
    else:
        total_weight = fixed_rule_weight + fixed_model_weight
        if total_weight <= 0:
            raise ValueError("rule/model weights must sum to a positive value.")
        weight_pairs = [
            (
                round(fixed_rule_weight / total_weight, 6),
                round(fixed_model_weight / total_weight, 6),
            )
        ]

    for rule_weight, model_weight in weight_pairs:

        combined_scores = []
        for row in analyzed_rows:
            model_score = row["model_score"]
            if model_score is None:
                combined_scores.append(float(row["rule_score"]))
            else:
                combined_scores.append(
                    rule_weight * float(row["rule_score"])
                    + model_weight * float(model_score)
                )

        for threshold in threshold_values:
            threshold_value = round(float(threshold), 6)
            y_pred = [1 if score >= threshold_value else 0 for score in combined_scores]
            metrics = compute_binary_metrics(y_true, y_pred)
            candidate = {
                "text_rule_weight": rule_weight,
                "text_model_weight": model_weight,
                "threshold": threshold_value,
                "metrics": metrics,
            }
            if best is None or is_better_candidate(candidate, best, objective):
                best = candidate

    assert best is not None
    return best


def is_better_candidate(candidate: dict, incumbent: dict, objective: str) -> bool:
    candidate_metrics = candidate["metrics"]
    incumbent_metrics = incumbent["metrics"]
    candidate_key = (
        objective_value(candidate_metrics, objective),
        candidate_metrics["accuracy"],
        candidate_metrics["f1"],
        candidate_metrics["precision"],
        candidate_metrics["recall"],
        -candidate["threshold"],
    )
    incumbent_key = (
        objective_value(incumbent_metrics, objective),
        incumbent_metrics["accuracy"],
        incumbent_metrics["f1"],
        incumbent_metrics["precision"],
        incumbent_metrics["recall"],
        -incumbent["threshold"],
    )
    return candidate_key > incumbent_key


def evaluate_combined_pipeline(analyzed_rows: list[dict], tuned_config: dict) -> dict:
    y_true = [int(row["label"]) for row in analyzed_rows]
    y_pred = []
    scores = []
    for row in analyzed_rows:
        model_score = row["model_score"]
        if model_score is None:
            combined_score = float(row["rule_score"])
        else:
            combined_score = (
                tuned_config["text_rule_weight"] * float(row["rule_score"])
                + tuned_config["text_model_weight"] * float(model_score)
            )
        scores.append(round(combined_score, 6))
        y_pred.append(1 if combined_score >= tuned_config["threshold"] else 0)

    metrics = compute_binary_metrics(y_true, y_pred)
    return {
        "metrics": metrics,
        "score_preview": scores[:10],
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite_output_dir)

    split_dir = Path(args.split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Missing split directory: {split_dir}. Run scripts/split_combined_dataset.py first."
        )

    cfg = Config()
    cfg.OCR_CHINESE_POLICY = args.chinese_policy
    allowed_exts = {ext.lower() for ext in cfg.SUPPORTED_IMAGE_EXTENSIONS}

    ocr = build_ocr_service(cfg, args)
    processor = OCRTextProcessor(cfg)
    prep_analyzer = TextRiskAnalyzer(cfg, load_models=False)

    split_rows: dict[str, list[dict]] = {}
    split_skipped: dict[str, list[dict]] = {}
    for split_name in ("train", "val", "test"):
        rows: list[dict] = []
        skipped: list[dict] = []
        for label_name in ("fake", "real"):
            label_rows, label_skipped = build_split_rows(
                split_dir,
                label_name,
                split_name,
                allowed_exts,
                ocr,
                processor,
                prep_analyzer,
                args.min_text_length,
            )
            rows.extend(label_rows)
            skipped.extend(label_skipped)
        rows.sort(key=lambda row: (row["label"], row["name"]))
        skipped.sort(key=lambda row: (row["label"], row["name"]))
        split_rows[split_name] = rows
        split_skipped[split_name] = skipped
        save_rows_csv(output_dir / f"{split_name}_rows.csv", rows)
        save_rows_csv(output_dir / f"{split_name}_skipped.csv", skipped)

    for split_name in ("train", "val", "test"):
        print(
            f"{split_name}: usable={len(split_rows[split_name])} "
            f"skipped={len(split_skipped[split_name])}"
        )
    print(f"OCR backend: {args.ocr_backend}")
    for label, value in ocr.runtime_details().items():
        print(f"{label}: {value}")

    if not split_rows["train"] or not split_rows["val"] or not split_rows["test"]:
        raise ValueError("Train/val/test must each contain at least one usable OCR row.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    id2label = {0: "non_phishing", 1: "phishing"}
    label2id = {"non_phishing": 0, "phishing": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
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
        training_dir=output_dir / "trainer_runs",
    )

    train_result = trainer.train()
    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    val_predictions = trainer.predict(val_dataset)
    test_predictions = trainer.predict(test_dataset)
    val_argmax_metrics = logits_to_argmax_metrics(val_predictions)
    test_argmax_metrics = logits_to_argmax_metrics(test_predictions)

    tuned_cfg = Config()
    tuned_cfg.OCR_CHINESE_POLICY = args.chinese_policy
    tuned_cfg.TEXT_PHISHING_MODEL_NAME = str(model_dir)
    tuned_analyzer = TextRiskAnalyzer(tuned_cfg)

    val_analyzed_rows = analyze_pipeline_rows(split_rows["val"], tuned_analyzer)
    test_analyzed_rows = analyze_pipeline_rows(split_rows["test"], tuned_analyzer)
    best_combined = tune_combined_pipeline(
        val_analyzed_rows,
        args.objective,
        args.rule_weight,
        args.model_weight,
        args.rule_weight_step,
        args.threshold_step,
        args.grid_search_weights,
    )
    test_combined = evaluate_combined_pipeline(test_analyzed_rows, best_combined)

    suggested_config = {
        "TEXT_PHISHING_MODEL_NAME": str(model_dir),
        "OCR_BACKEND": args.ocr_backend,
        "OCR_CHINESE_POLICY": args.chinese_policy,
        "TEXT_RULE_WEIGHT": best_combined["text_rule_weight"],
        "TEXT_MODEL_WEIGHT": best_combined["text_model_weight"],
        "MEDIUM_RISK_THRESHOLD": best_combined["threshold"],
    }
    if args.ocr_backend == "ollama":
        suggested_config["OCR_OLLAMA_MODEL"] = args.ollama_model
        suggested_config["OCR_OLLAMA_HOST"] = args.ollama_host
        suggested_config["OCR_OLLAMA_TIMEOUT_SECONDS"] = args.ollama_timeout_seconds
        suggested_config["OCR_OLLAMA_CLEAN_OUTPUT"] = not args.ollama_disable_cleaning
    elif args.ocr_backend == "transformers":
        suggested_config["OCR_TRANSFORMERS_MODEL"] = args.transformers_model
        suggested_config["OCR_TRANSFORMERS_TASK_PROMPT"] = args.transformers_task_prompt
        suggested_config["OCR_TRANSFORMERS_MAX_NEW_TOKENS"] = args.transformers_max_new_tokens
        suggested_config["OCR_TRANSFORMERS_NUM_BEAMS"] = args.transformers_num_beams
        suggested_config["OCR_TRANSFORMERS_CLEAN_OUTPUT"] = not args.transformers_disable_cleaning

    summary = {
        "base_model": args.base_model,
        "split_dir": str(split_dir),
        "output_dir": str(output_dir),
        "chinese_policy": args.chinese_policy,
        "objective": args.objective,
        "ocr_backend": args.ocr_backend,
        "ocr_active_languages": ocr.active_languages,
        "ocr_load_warning": ocr.load_error,
        "trainer_best_checkpoint": trainer.state.best_model_checkpoint,
        "class_weights": class_weights.tolist(),
        "split_counts": {
            split: {
                "usable_rows": len(split_rows[split]),
                "skipped_rows": len(split_skipped[split]),
                "label_counts": dict(Counter(int(row["label"]) for row in split_rows[split])),
            }
            for split in ("train", "val", "test")
        },
        "training_metrics": train_result.metrics,
        "validation_argmax_metrics": val_argmax_metrics,
        "test_argmax_metrics": test_argmax_metrics,
        "best_combined_validation": best_combined,
        "test_combined_metrics": test_combined["metrics"],
        "suggested_config": suggested_config,
    }
    if args.ocr_backend == "ollama":
        summary["ocr_ollama_model"] = args.ollama_model
        summary["ocr_ollama_host"] = args.ollama_host
        summary["ocr_ollama_timeout_seconds"] = args.ollama_timeout_seconds
        summary["ocr_ollama_clean_output"] = not args.ollama_disable_cleaning
    elif args.ocr_backend == "transformers":
        summary["ocr_transformers_model"] = args.transformers_model
        summary["ocr_transformers_task_prompt"] = args.transformers_task_prompt
        summary["ocr_transformers_max_new_tokens"] = args.transformers_max_new_tokens
        summary["ocr_transformers_num_beams"] = args.transformers_num_beams
        summary["ocr_transformers_clean_output"] = not args.transformers_disable_cleaning

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "suggested_config.json", suggested_config)

    print()
    print(f"Saved fine-tuned model to: {model_dir.resolve()}")
    print(
        "Best combined validation config: "
        f"rule_weight={best_combined['text_rule_weight']:.2f} "
        f"model_weight={best_combined['text_model_weight']:.2f} "
        f"threshold={best_combined['threshold']:.2f}"
    )
    print(
        "Validation combined metrics: "
        f"accuracy={best_combined['metrics']['accuracy']:.4f} "
        f"f1={best_combined['metrics']['f1']:.4f}"
    )
    print(
        "Test combined metrics: "
        f"accuracy={test_combined['metrics']['accuracy']:.4f} "
        f"f1={test_combined['metrics']['f1']:.4f}"
    )
    print(f"Suggested config saved to: {(output_dir / 'suggested_config.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
