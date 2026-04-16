from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config
from utils.ocr_runtime import add_ocr_runtime_args, build_ocr_service
from utils.ocr_text_processor import OCRTextProcessor
from utils.text_risk_analyzer import TextRiskAnalyzer


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Run the OCR + text phishing pipeline on one image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
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
        "--chinese-policy",
        choices=["strip", "skip", "translate", "route"],
        default=None,
        help="How to handle OCR text containing Chinese characters.",
    )
    add_ocr_runtime_args(parser, cfg, default_timeout_seconds=150)
    return parser.parse_args()


def safe_console_text(text: str) -> str:
    normalized = str(text)
    return normalized.encode("cp1252", errors="backslashreplace").decode("cp1252")


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Expected an image file path, got: {image_path}")

    cfg = Config()
    if args.chinese_policy is not None:
        cfg.OCR_CHINESE_POLICY = args.chinese_policy
    if args.text_model:
        cfg.TEXT_PHISHING_MODEL_NAME = args.text_model
    if args.text_positive_class_index is not None:
        cfg.TEXT_MODEL_POSITIVE_CLASS_INDEX = args.text_positive_class_index

    ocr = build_ocr_service(cfg, args)
    processor = OCRTextProcessor(cfg)
    analyzer = TextRiskAnalyzer(cfg)

    raw_text = ocr.extract_text(str(image_path))
    processed = processor.process(raw_text)
    processed_text = str(processed.get("text") or "")
    result = analyzer.analyze(processed_text)
    if processed.get("warning"):
        result.setdefault("warnings", []).append(processed["warning"])

    print(f"Image: {image_path}")
    print(f"OCR Backend: {ocr.backend}")
    print(f"OCR Active Languages: {ocr.active_languages}")
    print(f"OCR Load Warning: {ocr.load_error}")
    for label, value in ocr.runtime_details().items():
        print(f"{label}: {value}")
    grounded_regions = list(getattr(ocr, "last_grounding_dino_regions", []) or [])
    if grounded_regions:
        print("Grounding DINO Regions:")
        for index, region in enumerate(grounded_regions, start=1):
            print(
                f"  {index}. box={region.get('box')} "
                f"score={float(region.get('score') or 0.0):.3f} "
                f"label={region.get('label')}"
            )
    print(f"Chinese Policy: {cfg.OCR_CHINESE_POLICY}")
    print(f"English Model Path: {cfg.TEXT_PHISHING_MODEL_NAME}")
    print(f"English Model Loaded: {analyzer.model.is_loaded}")
    print(f"Chinese Model Loaded: {bool(analyzer.chinese_model and analyzer.chinese_model.is_loaded)}")
    print(f"Chinese Model Warning: {None if analyzer.chinese_model is None else analyzer.chinese_model.load_error}")
    print(f"Translator Loaded: {processor.is_loaded}")
    print(f"Translator Load Warning: {processor.load_error}")
    print()
    print("OCR Raw Text:")
    print(safe_console_text(raw_text))
    print()
    print("OCR Processed Text:")
    print(safe_console_text(processed_text))
    print()
    print("Text Result:")
    print(safe_console_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
