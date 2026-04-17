from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deepfake_model import DeepfakeModel
from preprocess.image_preprocessor import ImagePreprocessor
from utils.config import Config
from utils.inference_service import InferenceService
from utils.ocr_runtime import add_ocr_runtime_args, build_ocr_service
from utils.ocr_text_processor import OCRTextProcessor
from utils.risk_fusion_service import RiskFusionService
from utils.text_risk_analyzer import TextRiskAnalyzer
from utils.text_pipeline_runtime import run_text_pipeline_on_image


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the full multimodal phishing pipeline on labeled test directories. "
            "This runs image model + OCR + text pipeline + risk fusion."
        )
    )
    parser.add_argument(
        "--phishing-dir",
        default="data/combined_split/test/fake",
        help="Directory containing phishing images for evaluation.",
    )
    parser.add_argument(
        "--non-phishing-dir",
        default="data/combined_split/test/real",
        help="Directory containing non-phishing images for evaluation.",
    )
    parser.add_argument(
        "--val-phishing-dir",
        default="",
        help="Optional validation phishing directory used when auto-thresholding on validation.",
    )
    parser.add_argument(
        "--val-non-phishing-dir",
        default="",
        help="Optional validation non-phishing directory used when auto-thresholding on validation.",
    )
    parser.add_argument(
        "--text-model",
        default="",
        help="Optional local path or Hugging Face id override for the English text model.",
    )
    parser.add_argument(
        "--text-positive-class-index",
        type=int,
        default=None,
        help="Optional positive/spam class index override for the English text model.",
    )
    parser.add_argument(
        "--decision-mode",
        choices=["level", "score"],
        default="level",
        help=(
            "How to turn the fused output into a phishing/non-phishing prediction. "
            "'level' uses risk_level != low. 'score' uses --threshold."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for fused risk_score when --decision-mode score is used.",
    )
    parser.add_argument(
        "--auto-threshold-on-val",
        action="store_true",
        help="Tune the fused risk_score threshold on validation, then apply that threshold to the test split.",
    )
    parser.add_argument(
        "--auto-threshold-objective",
        choices=["accuracy", "f1", "precision", "recall"],
        default="accuracy",
        help="Metric used when --auto-threshold-on-val is enabled.",
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
        "--show-used-images",
        action="store_true",
        help="Print all images included in evaluation.",
    )
    parser.add_argument(
        "--show-misclassifications",
        type=int,
        default=10,
        help="Print up to N misclassified samples.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save a JSON summary.",
    )
    add_ocr_runtime_args(parser, cfg, default_timeout_seconds=150)
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


def safe_console_text(text: str) -> str:
    normalized = str(text)
    return normalized.encode("cp1252", errors="backslashreplace").decode("cp1252")


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
        raise ValueError("No scored validation rows available for threshold search.")

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

    return float(best_threshold), best_metrics


def infer_validation_dir(test_dir: Path) -> Path | None:
    parts = list(test_dir.parts)
    try:
        test_index = parts.index("test")
    except ValueError:
        return None

    val_parts = list(parts)
    val_parts[test_index] = "val"
    candidate = Path(*val_parts)
    return candidate if candidate.exists() else None


def resolve_validation_dirs(args: argparse.Namespace, phishing_dir: Path, non_phishing_dir: Path) -> tuple[Path, Path]:
    val_phishing_dir = Path(args.val_phishing_dir) if args.val_phishing_dir else infer_validation_dir(phishing_dir)
    val_non_phishing_dir = Path(args.val_non_phishing_dir) if args.val_non_phishing_dir else infer_validation_dir(non_phishing_dir)

    if val_phishing_dir is None or val_non_phishing_dir is None:
        raise ValueError(
            "Validation directories are required for --auto-threshold-on-val. "
            "Pass --val-phishing-dir and --val-non-phishing-dir explicitly."
        )
    if not val_phishing_dir.exists():
        raise FileNotFoundError(f"Missing validation phishing directory: {val_phishing_dir}")
    if not val_non_phishing_dir.exists():
        raise FileNotFoundError(f"Missing validation non-phishing directory: {val_non_phishing_dir}")
    return val_phishing_dir, val_non_phishing_dir


def build_dataset(phishing_files: list[Path], non_phishing_files: list[Path]) -> list[tuple[int, Path]]:
    return [(1, path) for path in phishing_files] + [(0, path) for path in non_phishing_files]


def run_pipeline_records(
    dataset: list[tuple[int, Path]],
    preprocessor: ImagePreprocessor,
    infer: InferenceService,
    ocr,
    ocr_text_processor: OCRTextProcessor,
    text_analyzer: TextRiskAnalyzer,
    risk_fusion: RiskFusionService,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    summary = {
        "total_images": len(dataset),
        "images_with_text": 0,
        "images_without_text": 0,
    }

    for true_label, image_path in dataset:
        image = preprocessor.load_image(str(image_path))
        image_result = infer.predict(image)
        text_runtime = run_text_pipeline_on_image(
            ocr,
            ocr_text_processor,
            text_analyzer,
            str(image_path),
        )
        raw_text = str(text_runtime["raw_text"] or "")
        processed = text_runtime["processed"]
        processed_text = str(text_runtime["processed_text"] or "")
        if processed_text.strip():
            summary["images_with_text"] += 1
        else:
            summary["images_without_text"] += 1
        text_result = text_runtime["text_result"]
        fused_result = risk_fusion.combine(image_result, text_result)
        risk_score = float(fused_result.get("risk_score") or 0.0)
        risk_level = str(fused_result.get("risk_level") or "").strip().lower()

        records.append(
            {
                "path": str(image_path),
                "true_label": int(true_label),
                "decision_score": round(risk_score, 6),
                "risk_level": risk_level,
                "risk_score": fused_result.get("risk_score"),
                "image_score": fused_result.get("image_score"),
                "text_score": fused_result.get("text_score"),
                "image_prediction": image_result.get("prediction"),
                "image_confidence": image_result.get("confidence"),
                "model_route": text_result.get("model_route"),
                "raw_text_preview": (raw_text[:160] + "...") if len(raw_text) > 160 else raw_text,
                "processed_text_preview": (
                    processed_text[:160] + "..."
                    if len(processed_text) > 160 else processed_text
                ),
                "reasons": list(fused_result.get("reasons") or []),
            }
        )

    return records, summary


def apply_decision_mode(
    records: list[dict[str, object]],
    decision_mode: str,
    threshold: float | None,
) -> tuple[list[int], list[int], list[dict[str, object]]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    enriched_records: list[dict[str, object]] = []

    for record in records:
        true_label = int(record["true_label"])
        risk_score = float(record["decision_score"])
        if decision_mode == "level":
            pred_label = 0 if str(record.get("risk_level") or "").lower() == "low" else 1
        else:
            assert threshold is not None
            pred_label = 1 if risk_score >= threshold else 0

        updated = dict(record)
        updated["pred_label"] = int(pred_label)
        y_true.append(true_label)
        y_pred.append(int(pred_label))
        enriched_records.append(updated)

    return y_true, y_pred, enriched_records


def main() -> int:
    args = parse_args()
    cfg = Config()
    if args.chinese_policy is not None:
        cfg.OCR_CHINESE_POLICY = args.chinese_policy
    if args.text_model:
        cfg.TEXT_PHISHING_MODEL_NAME = args.text_model
    if args.text_positive_class_index is not None:
        cfg.TEXT_MODEL_POSITIVE_CLASS_INDEX = args.text_positive_class_index
    if args.auto_threshold_on_val and args.decision_mode != "score":
        raise ValueError("--auto-threshold-on-val only works with --decision-mode score.")
    if args.auto_threshold_on_val and args.threshold is not None:
        raise ValueError("Use either --threshold or --auto-threshold-on-val, not both.")

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

    image_model = DeepfakeModel(cfg.DEEPFAKE_MODEL_NAME, cfg.DEVICE)
    infer = InferenceService(image_model)
    preprocessor = ImagePreprocessor()
    ocr = build_ocr_service(cfg, args)
    ocr_text_processor = OCRTextProcessor(cfg)
    text_analyzer = TextRiskAnalyzer(cfg)
    risk_fusion = RiskFusionService(cfg)

    test_dataset = build_dataset(phishing_files, non_phishing_files)
    test_records, test_summary = run_pipeline_records(
        test_dataset,
        preprocessor,
        infer,
        ocr,
        ocr_text_processor,
        text_analyzer,
        risk_fusion,
    )

    threshold = args.threshold if args.threshold is not None else cfg.MEDIUM_RISK_THRESHOLD
    threshold_source = "command line" if args.threshold is not None else "config default"
    validation_info = None
    if args.auto_threshold_on_val:
        val_phishing_dir, val_non_phishing_dir = resolve_validation_dirs(args, phishing_dir, non_phishing_dir)
        val_phishing_files = collect_images(val_phishing_dir, args.limit)
        val_non_phishing_files = collect_images(val_non_phishing_dir, args.limit)
        if not val_phishing_files or not val_non_phishing_files:
            raise ValueError("Validation directories must each contain at least one supported image.")

        val_dataset = build_dataset(val_phishing_files, val_non_phishing_files)
        val_records, val_summary = run_pipeline_records(
            val_dataset,
            preprocessor,
            infer,
            ocr,
            ocr_text_processor,
            text_analyzer,
            risk_fusion,
        )
        threshold, val_metrics = pick_best_threshold(val_records, args.auto_threshold_objective)
        threshold_source = f"validation auto-threshold ({args.auto_threshold_objective})"
        validation_info = {
            "phishing_dir": str(val_phishing_dir),
            "non_phishing_dir": str(val_non_phishing_dir),
            "summary": val_summary,
            "threshold": threshold,
            "metrics": val_metrics,
        }

    y_true, y_pred, records = apply_decision_mode(
        test_records,
        args.decision_mode,
        threshold if args.decision_mode == "score" else None,
    )
    metrics = compute_metrics(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    used_images = [str(record["path"]) for record in records]

    print(f"Scanned {len(test_dataset)} images")
    print(f"Phishing images: {len(phishing_files)}")
    print(f"Non-phishing images: {len(non_phishing_files)}")
    print(f"Images with OCR text: {test_summary['images_with_text']}")
    print(f"Images without OCR text: {test_summary['images_without_text']}")
    print(f"OCR backend: {args.ocr_backend}")
    print(f"OCR active languages: {ocr.active_languages}")
    print(f"OCR load warning: {ocr.load_error}")
    for label, value in ocr.runtime_details().items():
        print(f"{label}: {value}")
    print(f"Chinese policy: {cfg.OCR_CHINESE_POLICY}")
    print(f"Image model path: {cfg.DEEPFAKE_MODEL_NAME}")
    print(f"English model path: {cfg.TEXT_PHISHING_MODEL_NAME}")
    print(f"English model loaded: {text_analyzer.model.is_loaded}")
    print(f"Chinese model loaded: {bool(text_analyzer.chinese_model and text_analyzer.chinese_model.is_loaded)}")
    print(f"Chinese model load warning: {None if text_analyzer.chinese_model is None else text_analyzer.chinese_model.load_error}")
    print(f"Translator loaded: {ocr_text_processor.is_loaded}")
    print(f"Translator load warning: {ocr_text_processor.load_error}")
    print(f"Decision mode: {args.decision_mode}")
    if args.decision_mode == "score":
        print(f"Threshold: {threshold:.2f}")
        print(f"Threshold source: {threshold_source}")
    if validation_info is not None:
        print(f"Validation phishing dir: {validation_info['phishing_dir']}")
        print(f"Validation non-phishing dir: {validation_info['non_phishing_dir']}")
        print(f"Validation images with OCR text: {validation_info['summary']['images_with_text']}")
        print(f"Validation images without OCR text: {validation_info['summary']['images_without_text']}")
        print(
            "Validation threshold metrics: "
            f"accuracy={validation_info['metrics']['accuracy']:.4f} "
            f"precision={validation_info['metrics']['precision']:.4f} "
            f"recall={validation_info['metrics']['recall']:.4f} "
            f"f1={validation_info['metrics']['f1']:.4f}"
        )
    print()
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
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

    mistakes = [record for record in records if int(record["true_label"]) != int(record["pred_label"])]
    if mistakes and args.show_misclassifications > 0:
        print("Sample misclassifications:")
        for item in mistakes[:args.show_misclassifications]:
            print(
                f"- {item['path']} | true={item['true_label']} pred={item['pred_label']} "
                f"| risk_level={item['risk_level']} risk_score={item['risk_score']} "
                f"| image_score={item['image_score']} text_score={item['text_score']} "
                f"| image_pred={item['image_prediction']} "
                f"| text={safe_console_text(item['processed_text_preview'])}"
            )
            reasons = item.get("reasons") or []
            if reasons:
                print(f"  reasons={safe_console_text(' | '.join(reasons[:3]))}")

    if args.show_used_images:
        print()
        print("Images used for accuracy:")
        for path in used_images:
            print(f"- {path}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "phishing_dir": str(phishing_dir),
            "non_phishing_dir": str(non_phishing_dir),
            "ocr_backend": args.ocr_backend,
            "ocr_runtime_details": ocr.runtime_details(),
            "decision_mode": args.decision_mode,
            "threshold": None if args.decision_mode != "score" else threshold,
            "threshold_source": threshold_source,
            "validation": validation_info,
            "metrics": metrics,
            "confusion_matrix": confusion,
            "records": records,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"Saved JSON summary to: {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
