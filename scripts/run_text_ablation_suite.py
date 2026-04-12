from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr.ocr_service import OCRService
from utils.config import Config
from utils.ocr_text_processor import OCRTextProcessor
from utils.text_risk_analyzer import TextRiskAnalyzer


PIPELINE_ROWS = [
    ("raw_ocr_model", "Raw distilbert with ocr"),
    ("ocr_preprocess_model", "Raw distilbert with ocr and preprocess"),
    ("heuristic_only", "Heuristic only with ocr and preprocess"),
    ("combined_heuristic", "Distilbert with ocr, preprocess and combined heuristic"),
    ("combined_heuristic_url", "Distilbert with ocr, preprocess, combined heuristic and url model"),
]

SUSPICIOUS_URL_TERMS = (
    "verify",
    "verification",
    "login",
    "secure",
    "account",
    "update",
    "payment",
    "claim",
    "reward",
    "refund",
    "delivery",
    "parcel",
    "customs",
    "bank",
    "rebate",
    "loan",
    "otp",
    "cpf",
    "iras",
    "dbs",
    "shopee",
    "paynow",
    "grab",
    "singpass",
)

BRANDISH_DOMAIN_TERMS = (
    "dbs",
    "ocbc",
    "uob",
    "shopee",
    "singpost",
    "singpass",
    "iras",
    "cpf",
    "paynow",
    "grab",
    "usps",
    "jt",
    "jnt",
    "ninja",
    "dhl",
    "fedex",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-shot text ablation experiments across multiple fine-tuned "
            "DistilBERT checkpoints and OCR/pipeline stages, then export table-ready summaries."
        )
    )
    parser.add_argument(
        "--val-phishing-dir",
        default="data/combined_split/val/fake",
        help="Validation phishing-image directory.",
    )
    parser.add_argument(
        "--val-non-phishing-dir",
        default="data/combined_split/val/real",
        help="Validation non-phishing-image directory.",
    )
    parser.add_argument(
        "--test-phishing-dir",
        default="data/combined_split/test/fake",
        help="Test phishing-image directory.",
    )
    parser.add_argument(
        "--test-non-phishing-dir",
        default="data/combined_split/test/real",
        help="Test non-phishing-image directory.",
    )
    parser.add_argument(
        "--raw-base-model-path",
        default="distilbert-base-uncased",
        help="Checkpoint for a raw DistilBERT baseline without task fine-tuning.",
    )
    parser.add_argument(
        "--raw-base-model-label",
        default="Raw distilbert-base-uncased (no fine-tuning)",
        help="Table label for the raw unfine-tuned DistilBERT baseline.",
    )
    parser.add_argument(
        "--image-model-path",
        default="artifacts/distilbert_route_pipeline/model",
        help="Checkpoint for 'Distilbert fine tuned on 231 image data'.",
    )
    parser.add_argument(
        "--public-text-model-path",
        default="artifacts/distilbert_df_sample10/model",
        help="Checkpoint for 'Distilbert fine tuned on public text data'.",
    )
    parser.add_argument(
        "--hybrid-model-path",
        default="artifacts/distilbert_df_then_image/model",
        help="Checkpoint for 'Distilbert fine tuned on public text data and 231 image data'.",
    )
    parser.add_argument(
        "--image-model-label",
        default="Distilbert fine tuned on 231 image data",
        help="Table label for the image-only fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--public-text-model-label",
        default="Distilbert fine tuned on public text data",
        help="Table label for the public-text fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--hybrid-model-label",
        default="Distilbert fine tuned on public text data and 231 image data",
        help="Table label for the public-text then image-fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/text_ablation_suite",
        help="Directory for CSV/JSON/Markdown ablation summaries.",
    )
    parser.add_argument(
        "--chinese-policy",
        default="route",
        choices=["strip", "skip", "translate", "route"],
        help="OCR Chinese handling policy for the suite.",
    )
    parser.add_argument(
        "--objective",
        default="accuracy",
        choices=["accuracy", "f1", "precision", "recall"],
        help="Validation metric used to choose thresholds.",
    )
    parser.add_argument(
        "--url-weight-grid",
        default="0.05,0.10,0.15,0.20,0.25,0.30",
        help="Comma-separated URL-risk fusion weights to try for the URL ablation row.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of files to evaluate per class for quick smoke tests.",
    )
    return parser.parse_args()


def collect_images(root: Path, limit: int | None) -> list[Path]:
    cfg = Config()
    allowed_exts = {ext.lower() for ext in cfg.SUPPORTED_IMAGE_EXTENSIONS}
    files = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_exts
    )
    if limit is not None:
        files = files[:limit]
    return files


def build_dataset_rows(
    split_name: str,
    phishing_dir: Path,
    non_phishing_dir: Path,
    ocr: OCRService,
    processor: OCRTextProcessor,
    limit: int | None,
) -> list[dict]:
    dataset_rows: list[dict] = []
    files = [(1, path) for path in collect_images(phishing_dir, limit)]
    files.extend((0, path) for path in collect_images(non_phishing_dir, limit))

    for label, image_path in files:
        raw_text = ocr.extract_text(str(image_path))
        processed = processor.process(raw_text)
        dataset_rows.append(
            {
                "split": split_name,
                "path": str(image_path),
                "label": int(label),
                "raw_text": raw_text,
                "processed_text": str(processed.get("text") or ""),
                "processed_action": str(processed.get("action") or ""),
                "contains_chinese": bool(processed.get("contains_chinese")),
            }
        )
    return dataset_rows


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
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

    return best_threshold, best_metrics


def evaluate_threshold(records: list[dict[str, object]], threshold: float) -> dict[str, object]:
    y_true = [int(record["true_label"]) for record in records]
    y_pred = [1 if float(record["decision_score"]) >= threshold else 0 for record in records]
    metrics = compute_binary_metrics(y_true, y_pred)
    return {
        "metrics": metrics,
        "rows_used": len(records),
        "positive_predictions": int(sum(y_pred)),
    }


def predict_model_raw_from_text(analyzer: TextRiskAnalyzer, text: str) -> tuple[float | None, str | None]:
    normalized = str(text or "").strip()
    if not normalized:
        return None, None

    chunks = analyzer._split_text_for_model(normalized)
    best_prob = None
    best_route = None
    for chunk in chunks:
        prob, route, _ = analyzer._predict_chunk_probability(chunk)
        if prob is None:
            continue
        if best_prob is None or prob > best_prob:
            best_prob = float(prob)
            best_route = route

    return best_prob, best_route


def compute_url_risk_score(analyzer: TextRiskAnalyzer, analysis: dict[str, object]) -> float | None:
    urls = list(analysis.get("urls") or [])
    trusted_urls = list(analysis.get("trusted_urls") or [])
    if trusted_urls and not urls:
        return 0.0
    if not urls:
        return None

    filtered_text = str(analysis.get("filtered_text") or "").lower()
    best_score = 0.0
    for url in urls:
        domain = analyzer._normalize_url_domain(url)
        candidate = str(url or "").lower()
        score = 0.55

        if domain:
            if any(char.isdigit() for char in domain):
                score += 0.07
            if "-" in domain:
                score += 0.10
            first_label = domain.split(".", 1)[0]
            if len(first_label) >= 18:
                score += 0.05
            if any(term in domain for term in BRANDISH_DOMAIN_TERMS):
                score += 0.08

        keyword_hits = sum(term in candidate or term in filtered_text for term in SUSPICIOUS_URL_TERMS)
        score += min(keyword_hits * 0.05, 0.20)

        if re.search(r"https?:/{0,1}[il|/]+", candidate):
            score += 0.05
        if "@" in candidate:
            score += 0.08

        best_score = max(best_score, min(score, 1.0))

    return best_score


def build_records_for_stage(stage_key: str, rows: list[dict], analyzer: TextRiskAnalyzer) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for row in rows:
        label = int(row["label"])
        path = str(row["path"])
        raw_text = str(row["raw_text"] or "").strip()
        processed_text = str(row["processed_text"] or "").strip()

        if stage_key == "raw_ocr_model":
            score, route = predict_model_raw_from_text(analyzer, raw_text)
            if score is None:
                continue
            records.append(
                {
                    "path": path,
                    "true_label": label,
                    "decision_score": float(score),
                    "model_route": route,
                    "source_text": raw_text,
                }
            )
            continue

        if not processed_text:
            continue

        analysis = analyzer.analyze(processed_text)

        if stage_key == "ocr_preprocess_model":
            score = analysis.get("model_score_raw")
            if score is None:
                continue
            records.append(
                {
                    "path": path,
                    "true_label": label,
                    "decision_score": float(score),
                    "model_route": analysis.get("model_route"),
                    "source_text": processed_text,
                }
            )
            continue

        if stage_key == "heuristic_only":
            score = analysis.get("rule_score")
            if score is None:
                continue
            records.append(
                {
                    "path": path,
                    "true_label": label,
                    "decision_score": float(score),
                    "model_route": analysis.get("model_route"),
                    "source_text": processed_text,
                }
            )
            continue

        if stage_key == "combined_heuristic":
            score = analysis.get("score")
            if score is None:
                continue
            records.append(
                {
                    "path": path,
                    "true_label": label,
                    "decision_score": float(score),
                    "model_route": analysis.get("model_route"),
                    "source_text": processed_text,
                }
            )
            continue

        if stage_key == "combined_heuristic_url":
            text_score = analysis.get("score")
            if text_score is None:
                continue
            records.append(
                {
                    "path": path,
                    "true_label": label,
                    "text_score": float(text_score),
                    "url_score": compute_url_risk_score(analyzer, analysis),
                    "model_route": analysis.get("model_route"),
                    "source_text": processed_text,
                }
            )
            continue

        raise ValueError(f"Unsupported stage key: {stage_key}")

    return records


def fuse_url_scores(records: list[dict[str, object]], url_weight: float) -> list[dict[str, object]]:
    fused_records: list[dict[str, object]] = []
    for record in records:
        url_score = record.get("url_score")
        text_score = float(record["text_score"])
        if url_score is None:
            decision_score = text_score
        else:
            decision_score = ((1.0 - url_weight) * text_score) + (url_weight * float(url_score))

        fused = dict(record)
        fused["decision_score"] = round(float(decision_score), 6)
        fused_records.append(fused)
    return fused_records


def parse_weight_grid(value: str) -> list[float]:
    weights = []
    for part in str(value).split(","):
        token = part.strip()
        if not token:
            continue
        weight = float(token)
        if not (0.0 < weight < 1.0):
            raise ValueError(f"URL weight must be in (0, 1), got {weight}")
        weights.append(weight)
    if not weights:
        raise ValueError("At least one URL weight must be provided.")
    return weights


def tune_url_stage(
    val_records: list[dict[str, object]],
    test_records: list[dict[str, object]],
    objective: str,
    weight_grid: list[float],
) -> dict[str, object]:
    best_result: dict[str, object] | None = None

    for url_weight in weight_grid:
        weighted_val = fuse_url_scores(val_records, url_weight)
        threshold, val_metrics = pick_best_threshold(weighted_val, objective)
        weighted_test = fuse_url_scores(test_records, url_weight)
        test_eval = evaluate_threshold(weighted_test, threshold)

        candidate = {
            "url_weight": url_weight,
            "threshold": threshold,
            "val_metrics": val_metrics,
            "test_metrics": test_eval["metrics"],
            "val_rows_used": len(weighted_val),
            "test_rows_used": len(weighted_test),
        }
        candidate_key = (
            val_metrics[objective],
            val_metrics["accuracy"],
            val_metrics["f1"],
            val_metrics["precision"],
            -threshold,
        )
        if best_result is None or candidate_key > best_result["selection_key"]:
            best_result = dict(candidate)
            best_result["selection_key"] = candidate_key

    assert best_result is not None
    best_result.pop("selection_key", None)
    return best_result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_markdown_table(summary_rows: list[dict]) -> str:
    lines = [
        "| Type | Model | Accuracy on test dataset |",
        "| --- | --- | --- |",
    ]
    for row in summary_rows:
        accuracy = row["test_accuracy"]
        rendered_accuracy = f"{accuracy:.4f}" if isinstance(accuracy, (float, int)) else str(accuracy)
        lines.append(f"| {row['type']} | {row['model']} | {rendered_accuracy} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    cfg.OCR_CHINESE_POLICY = args.chinese_policy
    ocr = OCRService(list(cfg.OCR_LANGUAGES), gpu=(cfg.DEVICE == "cuda"))
    processor = OCRTextProcessor(cfg)

    val_rows = build_dataset_rows(
        split_name="val",
        phishing_dir=Path(args.val_phishing_dir),
        non_phishing_dir=Path(args.val_non_phishing_dir),
        ocr=ocr,
        processor=processor,
        limit=args.limit,
    )
    test_rows = build_dataset_rows(
        split_name="test",
        phishing_dir=Path(args.test_phishing_dir),
        non_phishing_dir=Path(args.test_non_phishing_dir),
        ocr=ocr,
        processor=processor,
        limit=args.limit,
    )

    model_specs = [
        {
            "model_label": args.raw_base_model_label,
            "model_path": args.raw_base_model_path,
        },
        {
            "model_label": args.image_model_label,
            "model_path": args.image_model_path,
        },
        {
            "model_label": args.public_text_model_label,
            "model_path": args.public_text_model_path,
        },
        {
            "model_label": args.hybrid_model_label,
            "model_path": args.hybrid_model_path,
        },
    ]

    url_weight_grid = parse_weight_grid(args.url_weight_grid)
    summary_rows: list[dict] = []
    detailed_rows: list[dict] = []

    for spec in model_specs:
        cfg = Config()
        cfg.OCR_CHINESE_POLICY = args.chinese_policy
        cfg.TEXT_PHISHING_MODEL_NAME = spec["model_path"]
        analyzer = TextRiskAnalyzer(cfg)

        for stage_key, stage_label in PIPELINE_ROWS:
            try:
                val_records = build_records_for_stage(stage_key, val_rows, analyzer)
                test_records = build_records_for_stage(stage_key, test_rows, analyzer)
                if not val_records or not test_records:
                    raise ValueError("No usable rows for this stage/model combination.")

                if stage_key == "combined_heuristic_url":
                    tuned = tune_url_stage(
                        val_records=val_records,
                        test_records=test_records,
                        objective=args.objective,
                        weight_grid=url_weight_grid,
                    )
                    threshold = tuned["threshold"]
                    val_metrics = tuned["val_metrics"]
                    test_metrics = tuned["test_metrics"]
                    val_rows_used = tuned["val_rows_used"]
                    test_rows_used = tuned["test_rows_used"]
                    url_weight = tuned["url_weight"]
                    notes = "URL risk scorer uses domain/pattern heuristics; no separate external URL model exists in this repo."
                else:
                    threshold, val_metrics = pick_best_threshold(val_records, args.objective)
                    test_eval = evaluate_threshold(test_records, threshold)
                    test_metrics = test_eval["metrics"]
                    val_rows_used = len(val_records)
                    test_rows_used = len(test_records)
                    url_weight = None
                    notes = ""

                summary_row = {
                    "type": stage_label,
                    "model": spec["model_label"],
                    "accuracy_on_test_dataset": round(float(test_metrics["accuracy"]), 4),
                }
                summary_rows.append(
                    {
                        "type": stage_label,
                        "model": spec["model_label"],
                        "test_accuracy": round(float(test_metrics["accuracy"]), 4),
                    }
                )
                detailed_rows.append(
                    {
                        "type": stage_label,
                        "model": spec["model_label"],
                        "model_path": spec["model_path"],
                        "val_rows_used": val_rows_used,
                        "test_rows_used": test_rows_used,
                        "threshold": round(float(threshold), 6),
                        "url_weight": None if url_weight is None else round(float(url_weight), 6),
                        "val_accuracy": round(float(val_metrics["accuracy"]), 6),
                        "val_precision": round(float(val_metrics["precision"]), 6),
                        "val_recall": round(float(val_metrics["recall"]), 6),
                        "val_f1": round(float(val_metrics["f1"]), 6),
                        "test_accuracy": round(float(test_metrics["accuracy"]), 6),
                        "test_precision": round(float(test_metrics["precision"]), 6),
                        "test_recall": round(float(test_metrics["recall"]), 6),
                        "test_f1": round(float(test_metrics["f1"]), 6),
                        "notes": notes,
                    }
                )
                print(
                    f"{stage_label} | {spec['model_label']} | "
                    f"val_acc={val_metrics['accuracy']:.4f} | "
                    f"test_acc={test_metrics['accuracy']:.4f} | "
                    f"threshold={threshold:.4f}"
                    + (f" | url_weight={url_weight:.2f}" if url_weight is not None else "")
                )
            except Exception as exc:
                detailed_rows.append(
                    {
                        "type": stage_label,
                        "model": spec["model_label"],
                        "model_path": spec["model_path"],
                        "val_rows_used": 0,
                        "test_rows_used": 0,
                        "threshold": "",
                        "url_weight": "",
                        "val_accuracy": "",
                        "val_precision": "",
                        "val_recall": "",
                        "val_f1": "",
                        "test_accuracy": "",
                        "test_precision": "",
                        "test_recall": "",
                        "test_f1": "",
                        "notes": f"FAILED: {exc}",
                    }
                )
                summary_rows.append(
                    {
                        "type": stage_label,
                        "model": spec["model_label"],
                        "test_accuracy": "FAILED",
                    }
                )
                print(f"{stage_label} | {spec['model_label']} | FAILED: {exc}")

    write_csv(output_dir / "ablation_summary.csv", detailed_rows)
    write_json(
        output_dir / "ablation_summary.json",
        {
            "objective": args.objective,
            "chinese_policy": args.chinese_policy,
            "val_counts": {
                "total_images": len(val_rows),
                "usable_processed_rows": sum(1 for row in val_rows if str(row["processed_text"]).strip()),
            },
            "test_counts": {
                "total_images": len(test_rows),
                "usable_processed_rows": sum(1 for row in test_rows if str(row["processed_text"]).strip()),
            },
            "rows": detailed_rows,
        },
    )
    (output_dir / "ablation_table.md").write_text(build_markdown_table(summary_rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
