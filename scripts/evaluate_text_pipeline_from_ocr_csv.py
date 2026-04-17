from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config
from utils.ocr_text_processor import OCRTextProcessor
from utils.text_risk_analyzer import TextRiskAnalyzer
from utils.text_pipeline_runtime import TEXT_DECISION_SOURCE_CHOICES, resolve_text_score_key


LABEL_TO_ID = {
    "real": 0,
    "fake": 1,
}


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the full text phishing pipeline from a pre-extracted OCR CSV "
            "without re-running OCR. Tunes threshold on one split and evaluates on another."
        )
    )
    parser.add_argument(
        "--ocr-csv",
        default="artifacts/ollama_ocr/english_data_ocr_strict_rerun_success.csv",
        help="CSV containing OCR text rows with path, model, text, error columns.",
    )
    parser.add_argument(
        "--manifest-csv",
        default="data/english_data_split/manifest.csv",
        help="Manifest CSV created by split_combined_dataset.py.",
    )
    parser.add_argument(
        "--text-model",
        default="",
        help="Optional local path or Hugging Face id for the English text model.",
    )
    parser.add_argument(
        "--text-positive-class-index",
        type=int,
        default=None,
        help="Optional positive/spam class index override for the English text model.",
    )
    parser.add_argument(
        "--decision-source",
        choices=TEXT_DECISION_SOURCE_CHOICES,
        default=cfg.TEXT_DECISION_SOURCE,
        help="Whether predictions should use the full text score, calibrated model score, or raw model probability.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold to use. Defaults to config value unless --auto-threshold is set.",
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        help="Tune threshold on the tune split and then evaluate the eval split once using that frozen threshold.",
    )
    parser.add_argument(
        "--auto-threshold-objective",
        choices=["accuracy", "f1", "precision", "recall"],
        default="accuracy",
        help="Metric used when --auto-threshold is enabled.",
    )
    parser.add_argument(
        "--tune-split",
        choices=["train", "val", "test"],
        default="val",
        help="Manifest split used to tune the threshold when --auto-threshold is enabled.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["train", "val", "test"],
        default="test",
        help="Manifest split used for the final reported evaluation.",
    )
    parser.add_argument(
        "--chinese-policy",
        choices=["strip", "skip", "translate", "route"],
        default=None,
        help="How to handle OCR text containing Chinese characters.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional JSON path to save the evaluation summary.",
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


def normalize_path(value: str) -> str:
    return str(Path(value)).replace("/", "\\").lower()


def compute_metrics(y_true: list[int], y_pred: list[int], y_score: list[float] | None = None) -> dict[str, float | None]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    roc_auc = None
    if y_score is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            roc_auc = None
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
    }


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
        metrics = compute_metrics(labels, preds, scores)
        comparison_key = (
            float(metrics[objective] or 0.0),
            float(metrics["accuracy"] or 0.0),
            float(metrics["f1"] or 0.0),
            float(metrics["precision"] or 0.0),
            -threshold,
        )
        if comparison_key > best_key:
            best_key = comparison_key
            best_threshold = threshold
            best_metrics = metrics

    return float(best_threshold), best_metrics


def evaluate_threshold(records: list[dict[str, object]], threshold: float) -> dict[str, object]:
    y_true = [int(record["true_label"]) for record in records]
    y_score = [float(record["decision_score"]) for record in records]
    y_pred = [1 if float(record["decision_score"]) >= threshold else 0 for record in records]
    metrics = compute_metrics(y_true, y_pred, y_score)
    return {
        "metrics": metrics,
        "rows_used": len(records),
        "positive_predictions": int(sum(y_pred)),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def load_manifest_rows(path: Path) -> list[dict]:
    maximize_csv_field_size()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def load_ocr_rows(path: Path) -> dict[str, dict]:
    maximize_csv_field_size()
    rows_by_path: dict[str, dict] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = str(row.get("path") or "").strip()
            if image_path:
                rows_by_path[normalize_path(image_path)] = row
    if not rows_by_path:
        raise ValueError(f"OCR CSV has no usable rows: {path}")
    return rows_by_path


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

    cfg = Config()
    if args.chinese_policy is not None:
        cfg.OCR_CHINESE_POLICY = args.chinese_policy
    if args.text_model:
        cfg.TEXT_PHISHING_MODEL_NAME = args.text_model
    if args.text_positive_class_index is not None:
        cfg.TEXT_MODEL_POSITIVE_CLASS_INDEX = args.text_positive_class_index

    processor = OCRTextProcessor(cfg)
    analyzer = TextRiskAnalyzer(cfg)
    manifest_rows = load_manifest_rows(manifest_csv)
    ocr_rows_by_path = load_ocr_rows(ocr_csv)

    split_records: dict[str, list[dict[str, object]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    skipped_rows: list[dict[str, object]] = []
    score_key = resolve_text_score_key(args.decision_source)

    for index, row in enumerate(manifest_rows):
        split = str(row.get("split") or "").strip().lower()
        label_name = str(row.get("label") or "").strip().lower()
        original_path = str(row.get("original_path") or "").strip()

        if split not in split_records:
            skipped_rows.append({"row_id": index, "reason": "invalid_split", "original_path": original_path})
            continue
        if label_name not in LABEL_TO_ID:
            skipped_rows.append({"row_id": index, "reason": "invalid_label", "original_path": original_path})
            continue

        ocr_row = ocr_rows_by_path.get(normalize_path(original_path))
        if ocr_row is None:
            skipped_rows.append({"row_id": index, "reason": "missing_ocr_row", "original_path": original_path})
            continue

        error_text = str(ocr_row.get("error") or "").strip()
        if error_text:
            skipped_rows.append(
                {
                    "row_id": index,
                    "reason": "ocr_error",
                    "original_path": original_path,
                    "ocr_error": error_text,
                }
            )
            continue

        raw_text = str(ocr_row.get("text") or "")
        processed = processor.process(raw_text)
        text = str(processed.get("text") or "").strip()
        if not text:
            skipped_rows.append(
                {
                    "row_id": index,
                    "reason": "empty_after_processing",
                    "original_path": original_path,
                }
            )
            continue

        analysis = analyzer.analyze(text)
        score = analysis.get(score_key)
        if score is None:
            skipped_rows.append(
                {
                    "row_id": index,
                    "reason": "missing_score",
                    "original_path": original_path,
                }
            )
            continue

        split_records[split].append(
            {
                "path": original_path,
                "true_label": LABEL_TO_ID[label_name],
                "decision_score": float(score),
                "combined_score": analysis.get("score"),
                "model_score": analysis.get("model_score"),
                "model_score_raw": analysis.get("model_score_raw"),
                "model_route": analysis.get("model_route"),
                "processed_text_preview": text[:200],
                "filtered_text_preview": str(analysis.get("filtered_text") or "")[:200],
            }
        )

    tune_records = split_records[args.tune_split]
    eval_records = split_records[args.eval_split]
    if not tune_records:
        raise ValueError(f"No usable rows found for tune split: {args.tune_split}")
    if not eval_records:
        raise ValueError(f"No usable rows found for eval split: {args.eval_split}")

    if args.auto_threshold:
        threshold, tune_metrics = pick_best_threshold(tune_records, args.auto_threshold_objective)
        threshold_source = f"auto-threshold ({args.auto_threshold_objective}) on {args.tune_split}"
    else:
        threshold = args.threshold if args.threshold is not None else cfg.MEDIUM_RISK_THRESHOLD
        tune_metrics = evaluate_threshold(tune_records, threshold)["metrics"]
        threshold_source = "command line" if args.threshold is not None else "config default"

    tune_eval = evaluate_threshold(tune_records, threshold)
    final_eval = evaluate_threshold(eval_records, threshold)

    print(f"OCR CSV: {ocr_csv}")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"Decision source: {args.decision_source}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Threshold source: {threshold_source}")
    print(f"Text model: {cfg.TEXT_PHISHING_MODEL_NAME}")
    print(f"Chinese policy: {cfg.OCR_CHINESE_POLICY}")
    print(f"Processor translator loaded: {processor.is_loaded}")
    print(f"Analyzer model loaded: {analyzer.model.is_loaded}")
    print()
    print(f"Tune split: {args.tune_split} | rows used: {len(tune_records)}")
    print(
        f"Tune metrics: accuracy={tune_eval['metrics']['accuracy']:.4f} "
        f"precision={tune_eval['metrics']['precision']:.4f} "
        f"recall={tune_eval['metrics']['recall']:.4f} "
        f"f1={tune_eval['metrics']['f1']:.4f} "
        f"auc={tune_eval['metrics']['roc_auc']:.4f}"
    )
    print()
    print(f"Eval split: {args.eval_split} | rows used: {len(eval_records)}")
    print(f"Accuracy: {final_eval['metrics']['accuracy']:.4f}")
    print(f"F1: {final_eval['metrics']['f1']:.4f}")
    print(f"ROC AUC: {final_eval['metrics']['roc_auc']:.4f}")
    print("Confusion matrix [ [tn, fp], [fn, tp] ]:")
    print(confusion_matrix(final_eval["y_true"], final_eval["y_pred"], labels=[0, 1]))
    print()
    print(classification_report(
        final_eval["y_true"],
        final_eval["y_pred"],
        labels=[0, 1],
        target_names=["non-phishing", "phishing"],
        digits=4,
        zero_division=0,
    ))

    summary = {
        "ocr_csv": str(ocr_csv),
        "manifest_csv": str(manifest_csv),
        "decision_source": args.decision_source,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_objective": args.auto_threshold_objective if args.auto_threshold else None,
        "tune_split": args.tune_split,
        "eval_split": args.eval_split,
        "tune_rows_used": len(tune_records),
        "eval_rows_used": len(eval_records),
        "tune_metrics": tune_eval["metrics"],
        "eval_metrics": final_eval["metrics"],
        "skipped_rows": len(skipped_rows),
        "text_model": cfg.TEXT_PHISHING_MODEL_NAME,
        "chinese_policy": cfg.OCR_CHINESE_POLICY,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to: {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
