from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from utils.config import Config
from utils.ocr_text_processor import OCRTextProcessor
from utils.ocr_runtime import add_ocr_runtime_args, build_ocr_service
from utils.text_risk_analyzer import TextRiskAnalyzer


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Evaluate OCR + bilingual phishing-text accuracy on labeled image folders."
    )
    parser.add_argument(
        "--phishing-dir",
        default="data/phishing",
        help="Directory containing phishing images.",
    )
    parser.add_argument(
        "--non-phishing-dir",
        default="data/non-phishing",
        help="Directory containing non-phishing images.",
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
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for the combined text score. Defaults to MEDIUM_RISK_THRESHOLD.",
    )
    parser.add_argument(
        "--decision-source",
        choices=["combined", "model", "model_raw"],
        default="combined",
        help="Whether predictions should use the full text score, calibrated model score, or raw model probability.",
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        help="Search the evaluated scores and choose the threshold that maximizes the selected objective.",
    )
    parser.add_argument(
        "--auto-threshold-objective",
        choices=["accuracy", "f1", "precision", "recall"],
        default="accuracy",
        help="Metric used when --auto-threshold is enabled.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of files to evaluate per class.",
    )
    parser.add_argument(
        "--chinese-policy",
        choices=["strip", "skip", "translate", "route"],
        default=None,
        help="How to handle OCR text containing Chinese characters.",
    )
    parser.add_argument(
        "--show-chinese-samples",
        type=int,
        default=0,
        help="Print up to N Chinese-containing OCR samples after processing.",
    )
    parser.add_argument(
        "--show-used-images",
        action="store_true",
        help="Print all images that were included in the accuracy calculation.",
    )
    add_ocr_runtime_args(parser, cfg, default_timeout_seconds=150)
    return parser.parse_args()


def collect_images(root: Path, limit: int | None) -> list[Path]:
    cfg = Config()
    allowed_exts = {ext.lower() for ext in cfg.SUPPORTED_IMAGE_EXTENSIONS}
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_exts
    )
    if limit is not None:
        files = files[:limit]
    return files


def safe_console_text(text: str) -> str:
    normalized = str(text)
    return normalized.encode("cp1252", errors="backslashreplace").decode("cp1252")


def resolve_score_key(decision_source: str) -> str:
    return {
        "combined": "score",
        "model": "model_score",
        "model_raw": "model_score_raw",
    }[decision_source]


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
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
        metrics = compute_metrics(labels, preds)
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


def main() -> int:
    args = parse_args()
    cfg = Config()
    if args.chinese_policy is not None:
        cfg.OCR_CHINESE_POLICY = args.chinese_policy
    if args.text_model:
        cfg.TEXT_PHISHING_MODEL_NAME = args.text_model
    if args.text_positive_class_index is not None:
        cfg.TEXT_MODEL_POSITIVE_CLASS_INDEX = args.text_positive_class_index
    threshold = args.threshold if args.threshold is not None else cfg.MEDIUM_RISK_THRESHOLD
    threshold_source = "command line" if args.threshold is not None else "config default"

    phishing_dir = Path(args.phishing_dir)
    non_phishing_dir = Path(args.non_phishing_dir)

    if not phishing_dir.exists():
        raise FileNotFoundError(f"Missing phishing directory: {phishing_dir}")
    if not non_phishing_dir.exists():
        raise FileNotFoundError(f"Missing non-phishing directory: {non_phishing_dir}")

    phishing_files = collect_images(phishing_dir, args.limit)
    non_phishing_files = collect_images(non_phishing_dir, args.limit)

    if not phishing_files or not non_phishing_files:
        raise ValueError("Both directories must contain at least one supported image.")

    ocr = build_ocr_service(cfg, args)
    processor = OCRTextProcessor(cfg)
    analyzer = TextRiskAnalyzer(cfg)

    skipped_no_text: list[str] = []
    used_images: list[str] = []
    chinese_hits = 0
    translated_count = 0
    stripped_count = 0
    routed_count = 0
    skipped_chinese: list[str] = []
    chinese_samples: list[dict[str, str]] = []
    route_counts = {
        "english": 0,
        "english_stripped_fallback": 0,
        "chinese": 0,
    }
    scored_records: list[dict[str, object]] = []

    dataset = [(1, p) for p in phishing_files] + [(0, p) for p in non_phishing_files]

    for label, image_path in dataset:
        raw_text = ocr.extract_text(str(image_path))
        processed = processor.process(raw_text)
        text = processed["text"]
        if processed.get("contains_chinese"):
            chinese_hits += 1
            if len(chinese_samples) < args.show_chinese_samples:
                chinese_samples.append(
                    {
                        "path": str(image_path),
                        "action": str(processed.get("action")),
                        "raw_text": raw_text,
                        "processed_text": text,
                    }
                )
        if processed.get("action") == "translated":
            translated_count += 1
        if processed.get("action") == "stripped_chinese":
            stripped_count += 1
        if processed.get("action") == "routed_chinese":
            routed_count += 1
        if not text or not text.strip():
            if processed.get("action") in {"skipped_chinese", "translation_unavailable"}:
                skipped_chinese.append(str(image_path))
            skipped_no_text.append(str(image_path))
            continue

        result = analyzer.analyze(text)
        if result.get("model_route") in route_counts:
            route_counts[str(result.get("model_route"))] += 1
        score_key = resolve_score_key(args.decision_source)
        score = float(result.get(score_key) or 0.0)
        used_images.append(str(image_path))

        scored_records.append(
            {
                "path": str(image_path),
                "true_label": label,
                "decision_score": score,
                "combined_score": result.get("score"),
                "model_score": result.get("model_score"),
                "model_score_raw": result.get("model_score_raw"),
                "text_preview": (text[:160] + "...") if len(text) > 160 else text,
                "filtered_preview": ((result.get("filtered_text") or "")[:160] + "...") if len(result.get("filtered_text") or "") > 160 else (result.get("filtered_text") or ""),
                "kept_chunks": result.get("relevant_chunks") or [],
                "model_route": result.get("model_route"),
            }
        )

    if not scored_records:
        raise ValueError(
            "No images with OCR text were found to evaluate. "
            f"OCR backend={args.ocr_backend}. "
            f"OCR load warning={ocr.load_error!r}. "
            f"Skipped images={len(skipped_no_text)}."
        )

    if args.auto_threshold and args.threshold is not None:
        raise ValueError("Use either --threshold or --auto-threshold, not both.")

    if args.auto_threshold:
        threshold, auto_metrics = pick_best_threshold(scored_records, args.auto_threshold_objective)
        threshold_source = f"auto-threshold ({args.auto_threshold_objective})"
    else:
        auto_metrics = None

    y_true = [int(item["true_label"]) for item in scored_records]
    y_pred = [1 if float(item["decision_score"]) >= threshold else 0 for item in scored_records]
    mistakes: list[dict[str, object]] = []
    for item, pred in zip(scored_records, y_pred):
        if pred != int(item["true_label"]):
            mistake = dict(item)
            mistake["pred_label"] = pred
            mistakes.append(mistake)

    print(f"Scanned {len(dataset)} images")
    print(f"Evaluated with OCR text: {len(scored_records)}")
    print(f"Phishing images: {len(phishing_files)}")
    print(f"Non-phishing images: {len(non_phishing_files)}")
    print(f"Skipped with no OCR text: {len(skipped_no_text)}")
    print(f"Chinese OCR hits: {chinese_hits}")
    print(f"Chinese policy: {cfg.OCR_CHINESE_POLICY}")
    print(f"Translated Chinese texts: {translated_count}")
    print(f"Stripped Chinese texts: {stripped_count}")
    print(f"Routed Chinese texts: {routed_count}")
    print(f"OCR active languages: {ocr.active_languages}")
    print(f"OCR load warning: {ocr.load_error}")
    print(f"OCR backend: {args.ocr_backend}")
    for label, value in ocr.runtime_details().items():
        print(f"{label}: {value}")
    print(f"Decision source: {args.decision_source}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Threshold source: {threshold_source}")
    print(f"English model path: {cfg.TEXT_PHISHING_MODEL_NAME}")
    if auto_metrics is not None:
        print(
            "Auto-threshold metrics: "
            f"accuracy={auto_metrics['accuracy']:.4f} "
            f"precision={auto_metrics['precision']:.4f} "
            f"recall={auto_metrics['recall']:.4f} "
            f"f1={auto_metrics['f1']:.4f}"
        )
    print(f"English model loaded: {analyzer.model.is_loaded}")
    print(f"Chinese model loaded: {bool(analyzer.chinese_model and analyzer.chinese_model.is_loaded)}")
    print(f"Chinese model load warning: {None if analyzer.chinese_model is None else analyzer.chinese_model.load_error}")
    print(f"Model routes used: {route_counts}")
    print(f"Translator loaded: {processor.is_loaded}")
    print(f"Translator load warning: {processor.load_error}")
    print()
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Confusion matrix [ [tn, fp], [fn, tp] ]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print()
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["non-phishing", "phishing"],
        digits=4,
        zero_division=0,
    ))

    if mistakes:
        print("Sample misclassifications:")
        for item in mistakes[:10]:
            print(
                f"- {item['path']} | true={item['true_label']} pred={item['pred_label']} "
                f"| decision_score={item['decision_score']:.4f} combined={item['combined_score']} "
                f"| model={item['model_score']} raw={item['model_score_raw']} "
                f"| route={item['model_route']} | text={safe_console_text(item['text_preview'])}"
            )
            if item['filtered_preview']:
                print(f"  filtered={safe_console_text(item['filtered_preview'])}")
            kept_chunks = item.get('kept_chunks') or []
            if kept_chunks:
                print(f"  kept_chunks={safe_console_text(' | '.join(kept_chunks[:3]))}")

    if skipped_no_text:
        print()
        print("Sample skipped images with no OCR text:")
        for path in skipped_no_text[:10]:
            print(f"- {path}")

    if skipped_chinese:
        print()
        print("Sample skipped due to Chinese handling:")
        for path in skipped_chinese[:10]:
            print(f"- {path}")

    if chinese_samples:
        print()
        print("Chinese OCR sample previews:")
        for sample in chinese_samples:
            print(f"- {sample['path']} | action={sample['action']}")
            print(f"  raw: {safe_console_text(sample['raw_text'])}")
            print(f"  processed: {safe_console_text(sample['processed_text'])}")

    if args.show_used_images:
        print()
        print("Images used for accuracy:")
        for path in used_images:
            print(f"- {path}")

        if skipped_no_text:
            print()
            print("Images skipped from accuracy:")
            for path in skipped_no_text:
                print(f"- {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
