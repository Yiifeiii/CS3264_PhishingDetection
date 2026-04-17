from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr.ocr_service import OCRService
from utils.config import Config
from utils.ocr_text_processor import OCRTextProcessor
from utils.text_risk_analyzer import TextRiskAnalyzer
from utils.text_pipeline_runtime import TEXT_DECISION_SOURCE_CHOICES, resolve_text_score_key


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description=(
            "Run OCR-backend ablations across multiple text models. "
            "Tunes threshold on validation and reports test performance."
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
        "--manifest-csv",
        default="data/english_data_split/manifest.csv",
        help="Manifest CSV produced by split_combined_dataset.py, used for saved OCR backends.",
    )
    parser.add_argument(
        "--ollama-ocr-csv",
        default="artifacts/ollama_ocr/english_data_ocr_strict_rerun_success.csv",
        help="Saved Ollama OCR CSV used by the ollama_csv backend.",
    )
    parser.add_argument(
        "--backends",
        default="easyocr,ollama_csv",
        help="Comma-separated OCR backends to compare: easyocr, ollama_csv, ollama, transformers.",
    )
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Repeatable model spec in the form Label=path. If omitted, sensible repo defaults are used.",
    )
    parser.add_argument(
        "--decision-source",
        choices=TEXT_DECISION_SOURCE_CHOICES,
        default=cfg.TEXT_DECISION_SOURCE,
        help="Which score to threshold for the text pipeline.",
    )
    parser.add_argument(
        "--objective",
        choices=["accuracy", "f1", "precision", "recall"],
        default="accuracy",
        help="Validation metric used to select the best threshold.",
    )
    parser.add_argument(
        "--chinese-policy",
        choices=["strip", "skip", "translate", "route"],
        default="route",
        help="OCR Chinese handling policy.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2-vision",
        help="Ollama vision model name for the live ollama backend.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama server base URL for the live ollama backend.",
    )
    parser.add_argument(
        "--ollama-timeout-seconds",
        type=int,
        default=150,
        help="Per-image timeout for the live ollama backend.",
    )
    parser.add_argument(
        "--transformers-model",
        default="florence-community/Florence-2-base-ft",
        help="Transformers vision OCR model name for the live transformers backend.",
    )
    parser.add_argument(
        "--transformers-task-prompt",
        default="<OCR>",
        help="Task prompt for the live transformers backend.",
    )
    parser.add_argument(
        "--transformers-max-new-tokens",
        type=int,
        default=1024,
        help="Max generated tokens for the live transformers backend.",
    )
    parser.add_argument(
        "--transformers-num-beams",
        type=int,
        default=3,
        help="Beam count for the live transformers backend.",
    )
    parser.add_argument(
        "--transformers-disable-cleaning",
        action="store_true",
        help="Disable OCR output cleaning for the live transformers backend.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of files to evaluate per class for quick smoke tests.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/ocr_backend_ablation",
        help="Directory for CSV/JSON/Markdown ablation summaries.",
    )
    return parser.parse_args()


def normalize_path(value: str) -> str:
    return str(Path(value)).replace("/", "\\").lower()


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

    return float(best_threshold), best_metrics


def evaluate_threshold(records: list[dict[str, object]], threshold: float) -> dict[str, object]:
    y_true = [int(record["true_label"]) for record in records]
    y_pred = [1 if float(record["decision_score"]) >= threshold else 0 for record in records]
    return {
        "metrics": compute_metrics(y_true, y_pred),
        "rows_used": len(records),
        "positive_predictions": int(sum(y_pred)),
    }


def parse_backend_list(value: str) -> list[str]:
    allowed = {"easyocr", "ollama_csv", "ollama", "transformers"}
    backends = [token.strip().lower() for token in str(value).split(",") if token.strip()]
    if not backends:
        raise ValueError("At least one backend must be provided.")
    invalid = [backend for backend in backends if backend not in allowed]
    if invalid:
        raise ValueError(f"Unsupported backend(s): {invalid}. Allowed: {sorted(allowed)}")
    return backends


def parse_model_specs(values: list[str]) -> list[dict[str, str]]:
    if values:
        specs = []
        for value in values:
            if "=" not in value:
                raise ValueError(f"Model spec must be Label=path, got: {value}")
            label, path = value.split("=", 1)
            specs.append({"label": label.strip(), "path": path.strip()})
        return specs

    defaults = [
        ("Distilbert fine tuned on 231 image data", "artifacts/distilbert_route_pipeline/model"),
        ("Distilbert fine tuned on public text data", "artifacts/distilbert_df_sample10/model"),
        ("Distilbert fine tuned on public text data and 231 image data", "artifacts/distilbert_df_then_image/model"),
        ("Distilbert fine tuned on strict Ollama OCR", "artifacts/ollama_ft_raw_strict/model"),
    ]
    specs = []
    for label, path in defaults:
        if (PROJECT_ROOT / path).exists():
            specs.append({"label": label, "path": path})
    if not specs:
        raise ValueError("No default model checkpoints found. Pass --model-spec Label=path.")
    return specs


def load_manifest_map(path: Path) -> dict[str, dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            normalize_path(str(row.get("split_path") or "")): row
            for row in reader
            if str(row.get("split_path") or "").strip()
        }


def load_ocr_csv_rows(path: Path) -> dict[str, dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            normalize_path(str(row.get("path") or "")): row
            for row in reader
            if str(row.get("path") or "").strip()
        }


def build_rows_from_backend(
    split_name: str,
    phishing_dir: Path,
    non_phishing_dir: Path,
    backend: str,
    processor: OCRTextProcessor,
    limit: int | None,
    manifest_map: dict[str, dict],
    ollama_csv_rows: dict[str, dict],
    easyocr_service: OCRService | None,
    ollama_service: OCRService | None,
    transformers_service: OCRService | None,
) -> tuple[list[dict], int]:
    dataset_rows: list[dict] = []
    skipped_rows = 0
    files = [(1, path) for path in collect_images(phishing_dir, limit)]
    files.extend((0, path) for path in collect_images(non_phishing_dir, limit))

    for label, image_path in files:
        raw_text = ""
        if backend == "easyocr":
            assert easyocr_service is not None
            raw_text = easyocr_service.extract_text(str(image_path))
        elif backend == "ollama":
            assert ollama_service is not None
            raw_text = ollama_service.extract_text(str(image_path))
        elif backend == "transformers":
            assert transformers_service is not None
            raw_text = transformers_service.extract_text(str(image_path))
        elif backend == "ollama_csv":
            manifest_row = manifest_map.get(normalize_path(str(image_path)))
            if manifest_row is None:
                skipped_rows += 1
                continue
            original_path = str(manifest_row.get("original_path") or "").strip()
            ocr_row = ollama_csv_rows.get(normalize_path(original_path))
            if ocr_row is None or str(ocr_row.get("error") or "").strip():
                skipped_rows += 1
                continue
            raw_text = str(ocr_row.get("text") or "")
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        processed = processor.process(raw_text)
        dataset_rows.append(
            {
                "split": split_name,
                "path": str(image_path),
                "label": int(label),
                "raw_text": raw_text,
                "processed_text": str(processed.get("text") or ""),
                "contains_chinese": bool(processed.get("contains_chinese")),
                "processed_action": str(processed.get("action") or ""),
            }
        )

    return dataset_rows, skipped_rows


def build_scored_records(rows: list[dict], analyzer: TextRiskAnalyzer, decision_source: str) -> list[dict[str, object]]:
    score_key = resolve_text_score_key(decision_source)
    records: list[dict[str, object]] = []

    for row in rows:
        text = str(row.get("processed_text") or "").strip()
        if not text:
            continue
        analysis = analyzer.analyze(text)
        score = analysis.get(score_key)
        if score is None:
            continue
        records.append(
            {
                "path": str(row["path"]),
                "true_label": int(row["label"]),
                "decision_score": float(score),
                "combined_score": analysis.get("score"),
                "model_score": analysis.get("model_score"),
                "model_score_raw": analysis.get("model_score_raw"),
                "model_route": analysis.get("model_route"),
            }
        )

    return records


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


def render_backend_label(backend: str) -> str:
    return {
        "easyocr": "EasyOCR",
        "ollama_csv": "Ollama OCR (saved text)",
        "ollama": "Ollama OCR (live)",
        "transformers": "Transformers OCR (live)",
    }[backend]


def build_markdown_table(rows: list[dict]) -> str:
    lines = [
        "| OCR Backend | Model | Accuracy on test dataset |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        accuracy = row["test_accuracy"]
        rendered_accuracy = f"{accuracy:.4f}" if isinstance(accuracy, (float, int)) else str(accuracy)
        lines.append(f"| {row['ocr_backend']} | {row['model']} | {rendered_accuracy} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backends = parse_backend_list(args.backends)
    model_specs = parse_model_specs(args.model_spec)

    cfg = Config()
    cfg.OCR_CHINESE_POLICY = args.chinese_policy
    processor = OCRTextProcessor(cfg)

    manifest_map = load_manifest_map(Path(args.manifest_csv)) if "ollama_csv" in backends else {}
    ollama_csv_rows = load_ocr_csv_rows(Path(args.ollama_ocr_csv)) if "ollama_csv" in backends else {}

    easyocr_service = None
    if "easyocr" in backends:
        easyocr_service = OCRService(list(cfg.OCR_LANGUAGES), gpu=(cfg.DEVICE == "cuda"), backend="easyocr")

    ollama_service = None
    if "ollama" in backends:
        ollama_service = OCRService(
            list(cfg.OCR_LANGUAGES),
            gpu=(cfg.DEVICE == "cuda"),
            backend="ollama",
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
            ollama_timeout_seconds=args.ollama_timeout_seconds,
            ollama_clean_output=True,
        )

    transformers_service = None
    if "transformers" in backends:
        transformers_service = OCRService(
            list(cfg.OCR_LANGUAGES),
            gpu=(cfg.DEVICE == "cuda"),
            backend="transformers",
            transformers_model=args.transformers_model,
            transformers_task_prompt=args.transformers_task_prompt,
            transformers_max_new_tokens=args.transformers_max_new_tokens,
            transformers_num_beams=args.transformers_num_beams,
            transformers_clean_output=not args.transformers_disable_cleaning,
        )

    backend_rows: dict[str, dict[str, object]] = {}
    for backend in backends:
        val_rows, val_skips = build_rows_from_backend(
            split_name="val",
            phishing_dir=Path(args.val_phishing_dir),
            non_phishing_dir=Path(args.val_non_phishing_dir),
            backend=backend,
            processor=processor,
            limit=args.limit,
            manifest_map=manifest_map,
            ollama_csv_rows=ollama_csv_rows,
            easyocr_service=easyocr_service,
            ollama_service=ollama_service,
            transformers_service=transformers_service,
        )
        test_rows, test_skips = build_rows_from_backend(
            split_name="test",
            phishing_dir=Path(args.test_phishing_dir),
            non_phishing_dir=Path(args.test_non_phishing_dir),
            backend=backend,
            processor=processor,
            limit=args.limit,
            manifest_map=manifest_map,
            ollama_csv_rows=ollama_csv_rows,
            easyocr_service=easyocr_service,
            ollama_service=ollama_service,
            transformers_service=transformers_service,
        )
        backend_rows[backend] = {
            "val_rows": val_rows,
            "test_rows": test_rows,
            "val_skips": val_skips,
            "test_skips": test_skips,
        }

    summary_rows: list[dict] = []
    detailed_rows: list[dict] = []

    for backend in backends:
        backend_label = render_backend_label(backend)
        rows_info = backend_rows[backend]

        for spec in model_specs:
            cfg = Config()
            cfg.OCR_CHINESE_POLICY = args.chinese_policy
            cfg.TEXT_PHISHING_MODEL_NAME = spec["path"]
            analyzer = TextRiskAnalyzer(cfg)

            try:
                val_records = build_scored_records(rows_info["val_rows"], analyzer, args.decision_source)
                test_records = build_scored_records(rows_info["test_rows"], analyzer, args.decision_source)
                if not val_records or not test_records:
                    raise ValueError("No usable scored rows for this backend/model combination.")

                threshold, val_metrics = pick_best_threshold(val_records, args.objective)
                test_eval = evaluate_threshold(test_records, threshold)
                test_metrics = test_eval["metrics"]

                summary_rows.append(
                    {
                        "ocr_backend": backend_label,
                        "model": spec["label"],
                        "test_accuracy": round(float(test_metrics["accuracy"]), 4),
                    }
                )
                detailed_rows.append(
                    {
                        "ocr_backend": backend_label,
                        "backend_key": backend,
                        "model": spec["label"],
                        "model_path": spec["path"],
                        "decision_source": args.decision_source,
                        "threshold": round(float(threshold), 6),
                        "val_rows_used": len(val_records),
                        "test_rows_used": len(test_records),
                        "val_skipped_before_scoring": rows_info["val_skips"],
                        "test_skipped_before_scoring": rows_info["test_skips"],
                        "val_accuracy": round(float(val_metrics["accuracy"]), 6),
                        "val_precision": round(float(val_metrics["precision"]), 6),
                        "val_recall": round(float(val_metrics["recall"]), 6),
                        "val_f1": round(float(val_metrics["f1"]), 6),
                        "test_accuracy": round(float(test_metrics["accuracy"]), 6),
                        "test_precision": round(float(test_metrics["precision"]), 6),
                        "test_recall": round(float(test_metrics["recall"]), 6),
                        "test_f1": round(float(test_metrics["f1"]), 6),
                    }
                )
                print(
                    f"{backend_label} | {spec['label']} | "
                    f"val_acc={val_metrics['accuracy']:.4f} | "
                    f"test_acc={test_metrics['accuracy']:.4f} | "
                    f"threshold={threshold:.4f}"
                )
            except Exception as exc:
                summary_rows.append(
                    {
                        "ocr_backend": backend_label,
                        "model": spec["label"],
                        "test_accuracy": "FAILED",
                    }
                )
                detailed_rows.append(
                    {
                        "ocr_backend": backend_label,
                        "backend_key": backend,
                        "model": spec["label"],
                        "model_path": spec["path"],
                        "decision_source": args.decision_source,
                        "threshold": "",
                        "val_rows_used": 0,
                        "test_rows_used": 0,
                        "val_skipped_before_scoring": rows_info["val_skips"],
                        "test_skipped_before_scoring": rows_info["test_skips"],
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
                print(f"{backend_label} | {spec['label']} | FAILED: {exc}")

    write_csv(output_dir / "ocr_backend_ablation_summary.csv", detailed_rows)
    write_json(
        output_dir / "ocr_backend_ablation_summary.json",
        {
            "decision_source": args.decision_source,
            "objective": args.objective,
            "chinese_policy": args.chinese_policy,
            "backends": backends,
            "models": model_specs,
            "rows": detailed_rows,
        },
    )
    (output_dir / "ocr_backend_ablation_table.md").write_text(build_markdown_table(summary_rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
