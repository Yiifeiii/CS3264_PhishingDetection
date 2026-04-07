from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ocr.ocr_service import OCRService
from utils.config import Config
from utils.ocr_text_processor import OCRTextProcessor
from utils.text_risk_analyzer import TextRiskAnalyzer


def parse_args() -> argparse.Namespace:
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
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for the combined text score. Defaults to MEDIUM_RISK_THRESHOLD.",
    )
    parser.add_argument(
        "--decision-source",
        choices=["combined", "model"],
        default="combined",
        help="Whether predictions should use the full text score or the model-only score.",
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


def main() -> int:
    args = parse_args()
    cfg = Config()
    if args.chinese_policy is not None:
        cfg.OCR_CHINESE_POLICY = args.chinese_policy
    threshold = args.threshold if args.threshold is not None else cfg.MEDIUM_RISK_THRESHOLD

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

    ocr = OCRService(list(cfg.OCR_LANGUAGES), gpu=(cfg.DEVICE == "cuda"))
    processor = OCRTextProcessor(cfg)
    analyzer = TextRiskAnalyzer(cfg)

    y_true: list[int] = []
    y_pred: list[int] = []
    mistakes: list[dict[str, object]] = []
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
        score_key = "score" if args.decision_source == "combined" else "model_score"
        score = float(result.get(score_key) or 0.0)
        pred = 1 if score >= threshold else 0
        used_images.append(str(image_path))

        y_true.append(label)
        y_pred.append(pred)

        if pred != label:
            mistakes.append(
                {
                    "path": str(image_path),
                    "true_label": label,
                    "pred_label": pred,
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

    if not y_true:
        raise ValueError("No images with OCR text were found to evaluate.")

    print(f"Scanned {len(dataset)} images")
    print(f"Evaluated with OCR text: {len(y_true)}")
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
    print(f"Decision source: {args.decision_source}")
    print(f"Threshold: {threshold:.2f}")
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