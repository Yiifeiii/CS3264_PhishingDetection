"""Extract sub-model scores from SigLIP and text pipelines into a unified CSV.

Uses ``data/sglip/{val,test}`` as the canonical split so that both the
pre-extracted SigLIP embeddings and the OCR + DistilBERT text pipeline
operate on exactly the same images.

Output
------
artifacts/ensemble/val_scores.csv
artifacts/ensemble/test_scores.csv

Each row contains:
    filename, label (0=real 1=fake),
    siglip_logreg_prob,
    text_raw_model_score, text_preprocess_model_score,
    text_heuristic_score, text_combined_score
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# SigLIP helpers
sys.path.insert(0, str(PROJECT_ROOT / "siglip"))
from config import (
    VAL_EMBED_FILE,
    TEST_EMBED_FILE,
    LOGREG_MODEL_FILE,
    SIGLIP_DATA_DIR,
)

# Text pipeline helpers
from ocr.ocr_service import OCRService
from utils.config import Config
from utils.ocr_text_processor import OCRTextProcessor
from utils.text_risk_analyzer import TextRiskAnalyzer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract ensemble feature scores.")
    p.add_argument(
        "--text-model-path",
        default=None,
        help="DistilBERT checkpoint for text scoring (default: Config default).",
    )
    p.add_argument(
        "--output-dir",
        default="artifacts/ensemble",
        help="Directory for output CSVs.",
    )
    p.add_argument(
        "--siglip-classifier",
        default=str(LOGREG_MODEL_FILE),
        help="Path to the trained SigLIP classifier .joblib.",
    )
    return p.parse_args()


# ── SigLIP score helpers ────────────────────────────────────────────

def load_siglip_scores(embed_file: Path, clf_path: Path) -> dict[str, tuple[int, float]]:
    """Return {filename: (label, phishing_prob)} from pre-extracted embeddings."""
    data = np.load(embed_file, allow_pickle=True)
    X, y, paths = data["X"], data["y"], data["paths"]
    clf = joblib.load(clf_path)
    probs = clf.predict_proba(X)[:, 1]

    result: dict[str, tuple[int, float]] = {}
    for path_str, label, prob in zip(paths, y, probs):
        fname = Path(str(path_str)).name
        result[fname] = (int(label), float(prob))
    return result


# ── Text score helpers ──────────────────────────────────────────────

def collect_images(root: Path) -> list[tuple[int, Path]]:
    """Collect (label, path) pairs from real/ and fake/ sub-dirs."""
    cfg = Config()
    allowed = {ext.lower() for ext in cfg.SUPPORTED_IMAGE_EXTENSIONS}
    pairs: list[tuple[int, Path]] = []
    for label, cls_name in [(0, "real"), (1, "fake")]:
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in allowed:
                pairs.append((label, f))
    return pairs


def extract_text_scores(
    image_pairs: list[tuple[int, Path]],
    ocr: OCRService,
    processor: OCRTextProcessor,
    analyzer: TextRiskAnalyzer,
) -> dict[str, dict]:
    """Return {filename: {label, raw_model, preprocess_model, heuristic, combined}}."""
    result: dict[str, dict] = {}

    for label, img_path in image_pairs:
        fname = img_path.name
        raw_text = ocr.extract_text(str(img_path))
        processed = processor.process(raw_text)
        processed_text = str(processed.get("text") or "").strip()

        # Stage 1: raw OCR -> model (no preprocessing)
        raw_model_score = _predict_raw(analyzer, raw_text)

        # Stage 2: preprocessed OCR -> model
        preprocess_model_score = None
        heuristic_score = None
        combined_score = None

        if processed_text:
            analysis = analyzer.analyze(processed_text)
            preprocess_model_score = analysis.get("model_score_raw")
            heuristic_score = analysis.get("rule_score")
            combined_score = analysis.get("score")

        result[fname] = {
            "label": label,
            "text_raw_model_score": raw_model_score,
            "text_preprocess_model_score": preprocess_model_score,
            "text_heuristic_score": heuristic_score,
            "text_combined_score": combined_score,
        }

    return result


def _predict_raw(analyzer: TextRiskAnalyzer, text: str) -> float | None:
    """Run model on raw text without preprocessing (Stage 1)."""
    normalized = (text or "").strip()
    if not normalized:
        return None
    chunks = analyzer._split_text_for_model(normalized)
    best = None
    for chunk in chunks:
        prob, _, _ = analyzer._predict_chunk_probability(chunk)
        if prob is not None and (best is None or prob > best):
            best = prob
    return best


# ── Main ────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def build_split(
    split_name: str,
    embed_file: Path,
    data_dir: Path,
    clf_path: Path,
    ocr: OCRService,
    processor: OCRTextProcessor,
    analyzer: TextRiskAnalyzer,
) -> list[dict]:
    print(f"\n=== Processing {split_name} split ===")

    # SigLIP scores
    siglip = load_siglip_scores(embed_file, clf_path)
    print(f"  SigLIP: {len(siglip)} images from embeddings")

    # Text scores
    image_pairs = collect_images(data_dir)
    print(f"  Text:   {len(image_pairs)} images from {data_dir}")
    text_scores = extract_text_scores(image_pairs, ocr, processor, analyzer)

    # Merge on filename
    all_fnames = set(siglip.keys()) | set(text_scores.keys())
    rows: list[dict] = []
    matched = 0
    for fname in sorted(all_fnames):
        sig = siglip.get(fname)
        txt = text_scores.get(fname)

        if sig is None or txt is None:
            continue  # skip images not in both pipelines

        matched += 1
        label = sig[0]  # both should agree; use SigLIP label
        rows.append({
            "filename": fname,
            "label": label,
            "siglip_logreg_prob": round(sig[1], 6),
            "text_raw_model_score": _round_or_na(txt["text_raw_model_score"]),
            "text_preprocess_model_score": _round_or_na(txt["text_preprocess_model_score"]),
            "text_heuristic_score": _round_or_na(txt["text_heuristic_score"]),
            "text_combined_score": _round_or_na(txt["text_combined_score"]),
        })

    print(f"  Matched: {matched} images in both pipelines")
    skipped = len(all_fnames) - matched
    if skipped:
        print(f"  Skipped: {skipped} images (in one pipeline only)")

    return rows


def _round_or_na(val):
    if val is None:
        return ""
    return round(float(val), 6)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    # Init text pipeline
    cfg = Config()
    if args.text_model_path:
        cfg.TEXT_PHISHING_MODEL_NAME = args.text_model_path
    ocr = OCRService(list(cfg.OCR_LANGUAGES), gpu=(cfg.DEVICE == "cuda"))
    processor = OCRTextProcessor(cfg)
    analyzer = TextRiskAnalyzer(cfg)

    clf_path = Path(args.siglip_classifier)
    if not clf_path.exists():
        print(f"ERROR: SigLIP classifier not found at {clf_path}")
        return 1

    for split_name, embed_file, data_subdir in [
        ("val", VAL_EMBED_FILE, SIGLIP_DATA_DIR / "val"),
        ("test", TEST_EMBED_FILE, SIGLIP_DATA_DIR / "test"),
    ]:
        if not embed_file.exists():
            print(f"WARNING: {embed_file} not found, skipping {split_name}")
            continue
        rows = build_split(split_name, embed_file, data_subdir, clf_path, ocr, processor, analyzer)
        out_path = output_dir / f"{split_name}_scores.csv"
        write_csv(out_path, rows)
        print(f"  Saved: {out_path} ({len(rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
