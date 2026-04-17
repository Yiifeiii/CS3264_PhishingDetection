from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deepfake_model import DeepfakeModel
from preprocess.image_preprocessor import ImagePreprocessor
from utils.config import Config
from utils.inference_service import InferenceService
from utils.ocr_runtime import build_ocr_service
from utils.ocr_text_processor import OCRTextProcessor
from utils.risk_fusion_service import RiskFusionService
from utils.text_risk_analyzer import TextRiskAnalyzer
from utils.text_pipeline_runtime import run_text_pipeline_on_image


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description=(
            "Run a full-pipeline OCR ablation on the English dataset. "
            "Compares EasyOCR, EasyOCR + Grounding DINO, and LLaMA via Ollama."
        )
    )
    parser.add_argument(
        "--val-phishing-dir",
        default="data/english_data_split/val/fake",
        help="Validation phishing-image directory.",
    )
    parser.add_argument(
        "--val-non-phishing-dir",
        default="data/english_data_split/val/real",
        help="Validation non-phishing-image directory.",
    )
    parser.add_argument(
        "--test-phishing-dir",
        default="data/english_data_split/test/fake",
        help="Test phishing-image directory.",
    )
    parser.add_argument(
        "--test-non-phishing-dir",
        default="data/english_data_split/test/real",
        help="Test non-phishing-image directory.",
    )
    parser.add_argument(
        "--backends",
        default="easyocr,easyocr_grounded,llama",
        help="Comma-separated OCR configs to compare: easyocr, easyocr_grounded, llama.",
    )
    parser.add_argument(
        "--objective",
        choices=["accuracy", "f1", "precision", "recall"],
        default="accuracy",
        help="Validation metric used to choose the best threshold per backend.",
    )
    parser.add_argument(
        "--text-model",
        default="",
        help="Optional shared text-model override used when no backend-specific model is provided.",
    )
    parser.add_argument(
        "--easyocr-text-model",
        default="",
        help="Optional text-model override for the EasyOCR backend.",
    )
    parser.add_argument(
        "--easyocr-grounded-text-model",
        default="",
        help="Optional text-model override for the EasyOCR + Grounding DINO backend.",
    )
    parser.add_argument(
        "--llama-text-model",
        default="",
        help="Optional text-model override for the LLaMA via Ollama backend.",
    )
    parser.add_argument(
        "--text-positive-class-index",
        type=int,
        default=None,
        help="Optional positive/spam class index override for the English text model.",
    )
    parser.add_argument(
        "--chinese-policy",
        choices=["strip", "skip", "translate", "route"],
        default=None,
        help="How to handle OCR text containing Chinese characters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of files to evaluate per class for quick smoke tests.",
    )
    parser.add_argument(
        "--show-misclassifications",
        type=int,
        default=3,
        help="Print up to N test misclassifications for each backend.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/english_ocr_ablation",
        help="Directory for CSV, JSON, and Markdown ablation summaries.",
    )
    parser.add_argument(
        "--ollama-model",
        default=getattr(cfg, "OCR_OLLAMA_MODEL", "llama3.2-vision"),
        help="Ollama vision model used for the llama backend.",
    )
    parser.add_argument(
        "--ollama-host",
        default=getattr(cfg, "OCR_OLLAMA_HOST", "http://localhost:11434"),
        help="Base URL for the Ollama server used by the llama backend.",
    )
    parser.add_argument(
        "--ollama-timeout-seconds",
        type=int,
        default=int(getattr(cfg, "OCR_OLLAMA_TIMEOUT_SECONDS", 150)),
        help="Per-image Ollama timeout for the llama backend.",
    )
    parser.add_argument(
        "--easyocr-grounding-dino-model",
        default=getattr(cfg, "OCR_EASYOCR_GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny"),
        help="Grounding DINO model used for easyocr_grounded.",
    )
    parser.add_argument(
        "--easyocr-grounding-dino-prompt",
        default=getattr(
            cfg,
            "OCR_EASYOCR_GROUNDING_DINO_PROMPT",
            "text. paragraph. text block. message. chat bubble. dialog.",
        ),
        help="Grounding DINO prompt used for easyocr_grounded.",
    )
    parser.add_argument(
        "--easyocr-grounding-box-threshold",
        type=float,
        default=float(getattr(cfg, "OCR_EASYOCR_GROUNDING_BOX_THRESHOLD", 0.25)),
        help="Grounding DINO box threshold used for easyocr_grounded.",
    )
    parser.add_argument(
        "--easyocr-grounding-text-threshold",
        type=float,
        default=float(getattr(cfg, "OCR_EASYOCR_GROUNDING_TEXT_THRESHOLD", 0.25)),
        help="Grounding DINO text threshold used for easyocr_grounded.",
    )
    parser.add_argument(
        "--easyocr-grounding-max-regions",
        type=int,
        default=int(getattr(cfg, "OCR_EASYOCR_GROUNDING_MAX_REGIONS", 6)),
        help="Maximum number of Grounding DINO regions to OCR for easyocr_grounded.",
    )
    parser.add_argument(
        "--easyocr-grounding-padding-ratio",
        type=float,
        default=float(getattr(cfg, "OCR_EASYOCR_GROUNDING_PADDING_RATIO", 0.03)),
        help="Extra padding ratio added around each Grounding DINO crop.",
    )
    parser.add_argument(
        "--easyocr-grounding-text-aggregation",
        choices=["concat", "max_model", "hybrid_max_model"],
        default=str(getattr(cfg, "OCR_EASYOCR_GROUNDING_TEXT_AGGREGATION", "concat")),
        help="How grounded OCR crop texts are aggregated before text-model scoring.",
    )
    return parser.parse_args()


def parse_backend_list(value: str) -> list[str]:
    aliases = {
        "ollama": "llama",
        "llama": "llama",
        "easyocr": "easyocr",
        "easyocr_grounded": "easyocr_grounded",
        "easyocr+dino": "easyocr_grounded",
        "easyocr_dino": "easyocr_grounded",
    }
    parsed = []
    for token in str(value).split(","):
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        backend = aliases.get(cleaned)
        if backend is None:
            raise ValueError(
                f"Unsupported backend '{token}'. Allowed backends: easyocr, easyocr_grounded, llama."
            )
        if backend not in parsed:
            parsed.append(backend)
    if not parsed:
        raise ValueError("At least one backend must be provided.")
    return parsed


def render_backend_label(backend: str, args: argparse.Namespace) -> str:
    if backend == "easyocr":
        return "EasyOCR"
    if backend == "easyocr_grounded":
        return "EasyOCR + Grounding DINO"
    if backend == "llama":
        return f"LLaMA via Ollama ({args.ollama_model})"
    raise ValueError(f"Unsupported backend: {backend}")


def resolve_backend_text_model(args: argparse.Namespace, backend: str) -> str:
    backend_specific = {
        "easyocr": args.easyocr_text_model,
        "easyocr_grounded": args.easyocr_grounded_text_model,
        "llama": args.llama_text_model,
    }.get(backend, "")
    if backend_specific:
        return str(backend_specific)
    if args.text_model:
        return str(args.text_model)

    defaults = {
        "easyocr": "artifacts/distilbert_route_pipeline/model",
        "easyocr_grounded": "artifacts/distilbert_easyocr_grounded_english/model",
        "llama": "artifacts/ollama_ft_raw_strict/model",
    }
    candidate = defaults.get(backend, "")
    if candidate and (PROJECT_ROOT / candidate).exists():
        return candidate
    return Config().TEXT_PHISHING_MODEL_NAME


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


def build_dataset(phishing_files: list[Path], non_phishing_files: list[Path]) -> list[tuple[int, Path]]:
    return [(1, path) for path in phishing_files] + [(0, path) for path in non_phishing_files]


def compute_metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict[str, float | None]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
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


def pick_best_threshold(records: list[dict[str, object]], objective: str) -> tuple[float, dict[str, float | None]]:
    scores = [float(record["decision_score"]) for record in records]
    labels = [int(record["true_label"]) for record in records]
    if not scores:
        raise ValueError("No scored validation rows available for threshold search.")

    candidate_thresholds = sorted(set(scores))
    candidate_thresholds = [0.0] + candidate_thresholds + [max(scores) + 1e-6]

    best_threshold = candidate_thresholds[0]
    best_metrics = {"accuracy": -1.0, "precision": -1.0, "recall": -1.0, "f1": -1.0, "roc_auc": None}
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


def build_ocr_args(args: argparse.Namespace, backend: str) -> argparse.Namespace:
    return argparse.Namespace(
        ocr_backend="ollama" if backend == "llama" else "easyocr",
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        ollama_timeout_seconds=args.ollama_timeout_seconds,
        ollama_disable_cleaning=False,
        easyocr_use_grounding_dino=(backend == "easyocr_grounded"),
        easyocr_grounding_dino_model=args.easyocr_grounding_dino_model,
        easyocr_grounding_dino_prompt=args.easyocr_grounding_dino_prompt,
        easyocr_grounding_box_threshold=args.easyocr_grounding_box_threshold,
        easyocr_grounding_text_threshold=args.easyocr_grounding_text_threshold,
        easyocr_grounding_max_regions=args.easyocr_grounding_max_regions,
        easyocr_grounding_padding_ratio=args.easyocr_grounding_padding_ratio,
        easyocr_grounding_text_aggregation=args.easyocr_grounding_text_aggregation,
        transformers_disable_cleaning=False,
    )


def run_pipeline_records(
    dataset: list[tuple[int, Path]],
    preprocessor: ImagePreprocessor,
    infer: InferenceService,
    ocr,
    ocr_text_processor: OCRTextProcessor,
    text_analyzer: TextRiskAnalyzer,
    risk_fusion: RiskFusionService,
) -> tuple[list[dict[str, object]], dict[str, int]]:
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
        processed = text_runtime["processed"]
        processed_text = str(text_runtime["processed_text"] or "")
        if processed_text.strip():
            summary["images_with_text"] += 1
        else:
            summary["images_without_text"] += 1
        text_result = text_runtime["text_result"]
        fused_result = risk_fusion.combine(image_result, text_result)
        risk_score = float(fused_result.get("risk_score") or 0.0)

        records.append(
            {
                "path": str(image_path),
                "true_label": int(true_label),
                "decision_score": risk_score,
                "pred_label": 0,
                "risk_level": str(fused_result.get("risk_level") or "").strip().lower(),
                "risk_score": fused_result.get("risk_score"),
                "image_score": fused_result.get("image_score"),
                "text_score": fused_result.get("text_score"),
                "image_prediction": image_result.get("prediction"),
                "processed_text_preview": (
                    processed_text[:160] + "..."
                    if len(processed_text) > 160 else processed_text
                ),
                "reasons": list(fused_result.get("reasons") or []),
            }
        )

    return records, summary


def evaluate_records(records: list[dict[str, object]], threshold: float) -> tuple[dict[str, float | None], list[dict[str, object]]]:
    y_true = [int(record["true_label"]) for record in records]
    y_score = [float(record["decision_score"]) for record in records]
    y_pred = [1 if score >= threshold else 0 for score in y_score]

    enriched_records: list[dict[str, object]] = []
    for record, pred in zip(records, y_pred):
        updated = dict(record)
        updated["pred_label"] = int(pred)
        enriched_records.append(updated)

    return compute_metrics(y_true, y_pred, y_score), enriched_records


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


def build_markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| OCR Backend | Threshold | Accuracy | F1 | ROC AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        auc_value = row["test_roc_auc"]
        rendered_auc = f"{auc_value:.4f}" if isinstance(auc_value, (float, int)) else "n/a"
        threshold_value = row["threshold"]
        rendered_threshold = f"{threshold_value:.4f}" if isinstance(threshold_value, (float, int)) else "n/a"
        accuracy_value = row["test_accuracy"]
        rendered_accuracy = f"{accuracy_value:.4f}" if isinstance(accuracy_value, (float, int)) else "FAILED"
        f1_value = row["test_f1"]
        rendered_f1 = f"{f1_value:.4f}" if isinstance(f1_value, (float, int)) else "FAILED"
        lines.append(
            f"| {row['ocr_backend']} | {rendered_threshold} | {rendered_accuracy} | "
            f"{rendered_f1} | {rendered_auc} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    backends = parse_backend_list(args.backends)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    if args.chinese_policy is not None:
        cfg.OCR_CHINESE_POLICY = args.chinese_policy
    if args.text_positive_class_index is not None:
        cfg.TEXT_MODEL_POSITIVE_CLASS_INDEX = args.text_positive_class_index

    val_phishing_dir = Path(args.val_phishing_dir)
    val_non_phishing_dir = Path(args.val_non_phishing_dir)
    test_phishing_dir = Path(args.test_phishing_dir)
    test_non_phishing_dir = Path(args.test_non_phishing_dir)
    for path in (val_phishing_dir, val_non_phishing_dir, test_phishing_dir, test_non_phishing_dir):
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset directory: {path}")

    val_phishing_files = collect_images(val_phishing_dir, args.limit)
    val_non_phishing_files = collect_images(val_non_phishing_dir, args.limit)
    test_phishing_files = collect_images(test_phishing_dir, args.limit)
    test_non_phishing_files = collect_images(test_non_phishing_dir, args.limit)
    if not val_phishing_files or not val_non_phishing_files:
        raise ValueError("Validation split must contain at least one supported image in each class.")
    if not test_phishing_files or not test_non_phishing_files:
        raise ValueError("Test split must contain at least one supported image in each class.")

    image_model = DeepfakeModel(cfg.DEEPFAKE_MODEL_NAME, cfg.DEVICE)
    infer = InferenceService(image_model)
    preprocessor = ImagePreprocessor()
    ocr_text_processor = OCRTextProcessor(cfg)
    risk_fusion = RiskFusionService(cfg)

    val_dataset = build_dataset(val_phishing_files, val_non_phishing_files)
    test_dataset = build_dataset(test_phishing_files, test_non_phishing_files)

    summary_rows: list[dict[str, object]] = []
    detailed_rows: list[dict[str, object]] = []
    experiment_rows: list[dict[str, object]] = []

    for backend in backends:
        backend_label = render_backend_label(backend, args)
        print(f"\n=== {backend_label} ===")

        try:
            backend_cfg = Config()
            if args.chinese_policy is not None:
                backend_cfg.OCR_CHINESE_POLICY = args.chinese_policy
            if args.text_positive_class_index is not None:
                backend_cfg.TEXT_MODEL_POSITIVE_CLASS_INDEX = args.text_positive_class_index
            backend_cfg.TEXT_PHISHING_MODEL_NAME = resolve_backend_text_model(args, backend)
            text_analyzer = TextRiskAnalyzer(backend_cfg)
            ocr = build_ocr_service(cfg, build_ocr_args(args, backend))
            val_records, val_summary = run_pipeline_records(
                val_dataset,
                preprocessor,
                infer,
                ocr,
                ocr_text_processor,
                text_analyzer,
                risk_fusion,
            )
            test_records, test_summary = run_pipeline_records(
                test_dataset,
                preprocessor,
                infer,
                ocr,
                ocr_text_processor,
                text_analyzer,
                risk_fusion,
            )
            threshold, val_metrics = pick_best_threshold(val_records, args.objective)
            test_metrics, enriched_test_records = evaluate_records(test_records, threshold)
            mistakes = [
                record for record in enriched_test_records
                if int(record["true_label"]) != int(record["pred_label"])
            ]

            summary_row = {
                "ocr_backend": backend_label,
                "backend_key": backend,
                "threshold_objective": args.objective,
                "text_model": backend_cfg.TEXT_PHISHING_MODEL_NAME,
                "threshold": round(threshold, 6),
                "val_accuracy": round(float(val_metrics["accuracy"] or 0.0), 6),
                "val_f1": round(float(val_metrics["f1"] or 0.0), 6),
                "val_roc_auc": None if val_metrics["roc_auc"] is None else round(float(val_metrics["roc_auc"]), 6),
                "test_accuracy": round(float(test_metrics["accuracy"] or 0.0), 6),
                "test_precision": round(float(test_metrics["precision"] or 0.0), 6),
                "test_recall": round(float(test_metrics["recall"] or 0.0), 6),
                "test_f1": round(float(test_metrics["f1"] or 0.0), 6),
                "test_roc_auc": None if test_metrics["roc_auc"] is None else round(float(test_metrics["roc_auc"]), 6),
                "val_images_with_text": val_summary["images_with_text"],
                "val_images_without_text": val_summary["images_without_text"],
                "test_images_with_text": test_summary["images_with_text"],
                "test_images_without_text": test_summary["images_without_text"],
                "ocr_load_warning": ocr.load_error,
            }
            summary_rows.append(summary_row)
            experiment_rows.append(
                {
                    "ocr_backend": backend_label,
                    "threshold": summary_row["threshold"],
                    "text_model": summary_row["text_model"],
                    "accuracy": summary_row["test_accuracy"],
                    "f1": summary_row["test_f1"],
                    "roc_auc": summary_row["test_roc_auc"],
                }
            )

            for record in enriched_test_records:
                detailed_rows.append(
                    {
                        "ocr_backend": backend_label,
                        "backend_key": backend,
                        "text_model": backend_cfg.TEXT_PHISHING_MODEL_NAME,
                        "path": record["path"],
                        "true_label": record["true_label"],
                        "pred_label": record["pred_label"],
                        "decision_score": round(float(record["decision_score"]), 6),
                        "threshold": round(threshold, 6),
                        "risk_level": record["risk_level"],
                        "risk_score": record["risk_score"],
                        "image_score": record["image_score"],
                        "text_score": record["text_score"],
                        "image_prediction": record["image_prediction"],
                        "processed_text_preview": record["processed_text_preview"],
                        "reasons": " | ".join(str(reason) for reason in (record.get("reasons") or [])),
                    }
                )

            auc_text = "n/a" if summary_row["test_roc_auc"] is None else f"{summary_row['test_roc_auc']:.4f}"
            print(
                f"threshold={summary_row['threshold']:.4f} | "
                f"accuracy={summary_row['test_accuracy']:.4f} | "
                f"f1={summary_row['test_f1']:.4f} | "
                f"auc={auc_text}"
            )
            print(f"text_model={summary_row['text_model']}")
            print(
                f"val_with_text={summary_row['val_images_with_text']} | "
                f"test_with_text={summary_row['test_images_with_text']}"
            )
            if mistakes and args.show_misclassifications > 0:
                print("Sample misclassifications:")
                for item in mistakes[:args.show_misclassifications]:
                    preview = str(item["processed_text_preview"])
                    short_preview = preview if len(preview) <= 140 else preview[:140] + "..."
                    print(
                        f"- {item['path']} | true={item['true_label']} pred={item['pred_label']} "
                        f"| risk_score={item['risk_score']} | text_score={item['text_score']} "
                        f"| text={short_preview}"
                    )
        except Exception as exc:
            failure_row = {
                "ocr_backend": backend_label,
                "backend_key": backend,
                "threshold_objective": args.objective,
                "text_model": resolve_backend_text_model(args, backend),
                "threshold": None,
                "val_accuracy": None,
                "val_f1": None,
                "val_roc_auc": None,
                "test_accuracy": None,
                "test_precision": None,
                "test_recall": None,
                "test_f1": None,
                "test_roc_auc": None,
                "val_images_with_text": 0,
                "val_images_without_text": 0,
                "test_images_with_text": 0,
                "test_images_without_text": 0,
                "ocr_load_warning": str(exc),
            }
            summary_rows.append(failure_row)
            experiment_rows.append(
                {
                    "ocr_backend": backend_label,
                    "threshold": "FAILED",
                    "text_model": resolve_backend_text_model(args, backend),
                    "accuracy": "FAILED",
                    "f1": "FAILED",
                    "roc_auc": "FAILED",
                }
            )
            print(f"FAILED: {exc}")

    write_csv(output_dir / "english_ocr_ablation_summary.csv", summary_rows)
    write_csv(output_dir / "english_ocr_ablation_detailed.csv", detailed_rows)
    (output_dir / "english_ocr_ablation_table.md").write_text(build_markdown_table(summary_rows), encoding="utf-8")
    (output_dir / "english_ocr_ablation_summary.json").write_text(
        json.dumps(
            {
                "val_phishing_dir": str(val_phishing_dir),
                "val_non_phishing_dir": str(val_non_phishing_dir),
                "test_phishing_dir": str(test_phishing_dir),
                "test_non_phishing_dir": str(test_non_phishing_dir),
                "objective": args.objective,
                "shared_text_model_override": args.text_model or "",
                "backend_text_models": {
                    backend: resolve_backend_text_model(args, backend)
                    for backend in backends
                },
                "backends": backends,
                "rows": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nFinal summary:")
    for row in experiment_rows:
        print(
            f"- {row['ocr_backend']} | threshold={row['threshold']} | "
            f"text_model={row['text_model']} | accuracy={row['accuracy']} | "
            f"f1={row['f1']} | auc={row['roc_auc']}"
        )
    print(f"\nSaved outputs to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
