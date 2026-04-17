"""Fill OCR/DistilBERT score columns into existing stage-1 score CSVs.

This script is for situations where image-branch scores already exist in
``train_scores.csv`` / ``test_scores.csv`` but the OCR score columns are blank.
It reuses a saved OCR CSV (e.g. Ollama output) and scores the matched text with
the repo's ``TextRiskAnalyzer``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config  # noqa: E402
from utils.ocr_text_processor import OCRTextProcessor  # noqa: E402
from utils.text_risk_analyzer import TextRiskAnalyzer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill saved OCR text scores into existing score CSVs.")
    parser.add_argument("--input-dir", required=True, help="Directory containing train_scores.csv and test_scores.csv.")
    parser.add_argument("--ocr-csv", required=True, help="Saved OCR CSV with text column.")
    parser.add_argument("--output-dir", required=True, help="Directory to write updated train/test score CSVs.")
    parser.add_argument("--text-model", default=None, help="Optional DistilBERT checkpoint path override.")
    parser.add_argument(
        "--overwrite-existing-text",
        action="store_true",
        help="Recompute OCR score columns even if they are already populated.",
    )
    return parser.parse_args()


def normalize_path_key(path_str: str) -> str:
    return str(Path(path_str)).replace("/", "\\").lower()


def load_saved_ocr(csv_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    by_path: dict[str, str] = {}
    by_name: dict[str, str] = {}

    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = str(row.get("text") or "").strip()
            error = str(row.get("error") or "").strip()
            if not text or error:
                continue

            candidates: list[str] = []
            for key in ("image_path", "path", "text_source_path", "live_image_path"):
                value = str(row.get(key) or "").strip()
                if value:
                    candidates.append(value)

            for candidate in candidates:
                by_path.setdefault(normalize_path_key(candidate), text)
                by_name.setdefault(Path(candidate).name.lower(), text)

            filename = str(row.get("filename") or "").strip()
            if filename:
                by_name.setdefault(Path(filename).name.lower(), text)

    return by_path, by_name


def should_fill_row(row: dict[str, str], overwrite_existing_text: bool) -> bool:
    if overwrite_existing_text:
        return True
    return not any(
        str(row.get(col) or "").strip()
        for col in (
            "ocr_distilbert_combined",
            "ocr_distilbert_heuristic",
            "ocr_distilbert_model",
        )
    )


def round_or_blank(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def fill_split(
    input_csv: Path,
    output_csv: Path,
    ocr_by_path: dict[str, str],
    ocr_by_name: dict[str, str],
    processor: OCRTextProcessor,
    analyzer: TextRiskAnalyzer,
    overwrite_existing_text: bool,
) -> None:
    with input_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    matched = 0
    missing = 0
    preserved = 0

    for row in rows:
        if not should_fill_row(row, overwrite_existing_text):
            preserved += 1
            continue

        image_path = str(row.get("image_path") or "").strip()
        filename = Path(image_path).name.lower()
        raw_text = ocr_by_path.get(normalize_path_key(image_path)) or ocr_by_name.get(filename, "")

        if not raw_text:
            row["ocr_distilbert_combined"] = ""
            row["ocr_distilbert_heuristic"] = ""
            row["ocr_distilbert_model"] = ""
            missing += 1
            continue

        matched += 1
        processed = processor.process(raw_text)
        processed_text = str(processed.get("text") or "").strip()
        if not processed_text:
            row["ocr_distilbert_combined"] = ""
            row["ocr_distilbert_heuristic"] = ""
            row["ocr_distilbert_model"] = ""
            continue

        analysis = analyzer.analyze(processed_text)
        row["ocr_distilbert_combined"] = round_or_blank(analysis.get("score"))
        row["ocr_distilbert_heuristic"] = round_or_blank(analysis.get("rule_score"))
        row["ocr_distilbert_model"] = round_or_blank(analysis.get("model_score_raw"))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{input_csv.name}: matched_saved={matched}/{len(rows)}, "
        f"missing_saved={missing}/{len(rows)}, preserved_existing={preserved}"
    )
    print(f"  saved -> {output_csv}")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    train_csv = input_dir / "train_scores.csv"
    test_csv = input_dir / "test_scores.csv"
    for path in (train_csv, test_csv):
        if not path.exists():
            print(f"ERROR: missing required file {path}")
            return 1

    cfg = Config()
    if args.text_model:
        cfg.TEXT_PHISHING_MODEL_NAME = args.text_model

    ocr_by_path, ocr_by_name = load_saved_ocr(Path(args.ocr_csv))
    print(
        f"Loaded OCR cache: path_keys={len(ocr_by_path)}, "
        f"filename_keys={len(ocr_by_name)}"
    )

    processor = OCRTextProcessor(cfg)
    analyzer = TextRiskAnalyzer(cfg)

    fill_split(
        train_csv,
        output_dir / "train_scores.csv",
        ocr_by_path,
        ocr_by_name,
        processor,
        analyzer,
        overwrite_existing_text=args.overwrite_existing_text,
    )
    fill_split(
        test_csv,
        output_dir / "test_scores.csv",
        ocr_by_path,
        ocr_by_name,
        processor,
        analyzer,
        overwrite_existing_text=args.overwrite_existing_text,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
