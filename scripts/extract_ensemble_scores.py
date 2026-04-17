"""Extract sub-model scores for the two Bayesian ensemble inputs.

Uses the Grounding DINO fusion holdout split
(``outputs/chat_social_grounding_dino_fusion/split/{train,test}_samples.csv``)
as the canonical split so that both pipelines operate on exactly the same images.

Model A — fuse_siglip_DINO:
    Loads pre-computed embeddings from the ``fusion_concat_siglip`` stream
    (global SigLIP + Grounding DINO crop SigLIP concatenated) and runs the
    trained LightGBM classifier to get ``fuse_siglip_dino_prob``.

    Also loads the crop-only stream (``grounding_dino_crop_siglip``) LightGBM
    classifier to emit ``crop_siglip_dino_prob``. This is an auxiliary signal
    used by the Step 4 safety guardrail at inference time (not a feature of
    the fused meta-classifier).

Model B — ocr_ollama_distilbert:
    For each image:
      1. Runs a local Ollama vision model (e.g. ``llama3.2-vision``) as OCR
         to transcribe all visible text on the full image in one shot.
      2. Preprocesses the transcription with ``OCRTextProcessor``.
      3. Scores with ``TextRiskAnalyzer`` (DistilBERT + heuristics) to get
         ``ocr_distilbert_combined``, ``ocr_distilbert_heuristic``, and
         ``ocr_distilbert_model``.

Output
------
{output_dir}/train_scores.csv
{output_dir}/test_scores.csv

Each row contains:
    image_path, filename, label,
    fuse_siglip_dino_prob, crop_siglip_dino_prob,
    ocr_distilbert_combined, ocr_distilbert_heuristic, ocr_distilbert_model
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

import torch  # noqa: E402

from ocr.ocr_service import OCRService  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.ocr_text_processor import OCRTextProcessor  # noqa: E402
from utils.text_risk_analyzer import TextRiskAnalyzer  # noqa: E402

FUSION_ROOT = PROJECT_ROOT / "outputs" / "chat_social_grounding_dino_fusion"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract ensemble feature scores.")
    p.add_argument(
        "--fusion-root",
        default=str(FUSION_ROOT),
        help="Root of the Grounding DINO fusion holdout output.",
    )
    p.add_argument(
        "--ocr-csv",
        default=None,
        help="Optional saved OCR CSV to reuse instead of live OCR where possible.",
    )
    p.add_argument(
        "--allow-partial-ocr",
        action="store_true",
        help="Allow rows with no saved OCR match to remain blank.",
    )
    p.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama vision model name (default: Config.OCR_OLLAMA_MODEL).",
    )
    p.add_argument(
        "--ollama-host",
        default=None,
        help="Ollama server URL (default: Config.OCR_OLLAMA_HOST).",
    )
    p.add_argument(
        "--ollama-timeout-seconds",
        type=int,
        default=None,
        help="Per-image Ollama timeout (default: Config.OCR_OLLAMA_TIMEOUT_SECONDS).",
    )
    p.add_argument(
        "--text-model-path",
        default=None,
        help="DistilBERT checkpoint path (default: Config.TEXT_PHISHING_MODEL_NAME).",
    )
    p.add_argument(
        "--text-model",
        default=None,
        help="Alias for --text-model-path.",
    )
    p.add_argument(
        "--output-dir",
        default="artifacts/ensemble",
        help="Directory for output CSVs.",
    )
    return p.parse_args()


# ── Model A: fuse_siglip_DINO score ─────────────────────────────────

def load_fusion_scores(
    embed_file: Path, clf_path: Path
) -> dict[str, tuple[int, float]]:
    """Return {image_path: (label, fake_prob)} from fusion embeddings + classifier."""
    data = np.load(embed_file, allow_pickle=True)
    X, y, paths = data["X"], data["y"], data["paths"]
    clf = joblib.load(clf_path)
    probs = clf.predict_proba(X)[:, 1]

    result: dict[str, tuple[int, float]] = {}
    for path_str, label, prob in zip(paths, y, probs):
        result[str(path_str)] = (int(label), float(prob))
    return result


# ── Split CSV loader ────────────────────────────────────────────────

def load_split_csv(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _normalize_path_key(path_str: str) -> str:
    return str(Path(path_str)).replace("/", "\\").lower()


def load_saved_ocr(csv_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return OCR maps keyed by normalized path and lowercased basename."""
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
                by_path.setdefault(_normalize_path_key(candidate), text)
                by_name.setdefault(Path(candidate).name.lower(), text)

            filename = str(row.get("filename") or "").strip()
            if filename:
                by_name.setdefault(Path(filename).name.lower(), text)

    return by_path, by_name


# ── Model B: Ollama OCR + DistilBERT ────────────────────────────────

@torch.no_grad()
def extract_ocr_distilbert_scores(
    image_paths: list[str],
    ocr: OCRService,
    processor: OCRTextProcessor,
    analyzer: TextRiskAnalyzer,
    saved_ocr_by_path: dict[str, str] | None = None,
    saved_ocr_by_name: dict[str, str] | None = None,
    allow_partial_ocr: bool = False,
) -> dict[str, dict]:
    """Return {image_path: {ocr_distilbert_combined, _heuristic, _model}}."""
    result: dict[str, dict] = {}
    total = len(image_paths)
    matched_saved = 0
    missing_saved = 0
    live_used = 0

    for i, img_path in enumerate(image_paths, 1):
        fname = Path(img_path).name
        print(f"    [{i}/{total}] OCR: {fname}", flush=True)

        raw_text = ""
        if saved_ocr_by_path is not None or saved_ocr_by_name is not None:
            raw_text = (
                (saved_ocr_by_path or {}).get(_normalize_path_key(img_path))
                or (saved_ocr_by_name or {}).get(fname.lower(), "")
            )
            if raw_text:
                matched_saved += 1
                print("      source: saved OCR", flush=True)
            else:
                missing_saved += 1
                if not allow_partial_ocr:
                    raise RuntimeError(
                        f"no saved OCR row matched '{img_path}'. "
                        "Use --allow-partial-ocr to keep text blank."
                    )
                print("      WARNING: no saved OCR match; leaving text blank", flush=True)
        else:
            live_used += 1
            try:
                raw_text = ocr.extract_text(img_path)
            except Exception as exc:
                print(f"      WARNING: OCR failed ({exc}); using empty text")
                raw_text = ""

        processed = processor.process(raw_text or "")
        processed_text = str(processed.get("text") or "").strip()

        combined = heuristic = model_score = None
        if processed_text:
            analysis = analyzer.analyze(processed_text)
            model_score = analysis.get("model_score_raw")
            heuristic = analysis.get("rule_score")
            combined = analysis.get("score")

        result[img_path] = {
            "ocr_distilbert_combined": combined,
            "ocr_distilbert_heuristic": heuristic,
            "ocr_distilbert_model": model_score,
        }

    if saved_ocr_by_path is not None or saved_ocr_by_name is not None:
        print(
            f"  OCR summary: matched_saved={matched_saved}/{total}, "
            f"missing_saved={missing_saved}/{total}, live_used={live_used}",
            flush=True,
        )

    return result


# ── Utilities ───────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _round_or_na(val) -> object:
    if val is None:
        return ""
    return round(float(val), 6)


# ── Main ────────────────────────────────────────────────────────────

def build_split(
    split_name: str,
    split_rows: list[dict],
    fusion_scores: dict[str, tuple[int, float]],
    crop_scores: dict[str, tuple[int, float]],
    ocr: OCRService,
    processor: OCRTextProcessor,
    analyzer: TextRiskAnalyzer,
    saved_ocr_by_path: dict[str, str] | None = None,
    saved_ocr_by_name: dict[str, str] | None = None,
    allow_partial_ocr: bool = False,
) -> list[dict]:
    print(f"\n=== Processing {split_name} split ({len(split_rows)} images) ===")

    image_paths = [row["image_path"] for row in split_rows]
    text_scores = extract_ocr_distilbert_scores(
        image_paths=image_paths,
        ocr=ocr,
        processor=processor,
        analyzer=analyzer,
        saved_ocr_by_path=saved_ocr_by_path,
        saved_ocr_by_name=saved_ocr_by_name,
        allow_partial_ocr=allow_partial_ocr,
    )

    out_rows: list[dict] = []
    matched = 0
    for row in split_rows:
        img_path = row["image_path"]
        label = int(row["label_id"])
        fname = Path(img_path).name

        fusion = fusion_scores.get(img_path)
        fusion_prob = fusion[1] if fusion is not None else None
        if fusion_prob is None:
            print(f"  WARNING: no fusion score for {fname}, skipping")
            continue

        crop = crop_scores.get(img_path)
        crop_prob = crop[1] if crop is not None else None

        txt = text_scores.get(img_path, {})
        matched += 1
        out_rows.append({
            "image_path": img_path,
            "filename": fname,
            "label": label,
            "fuse_siglip_dino_prob": _round_or_na(fusion_prob),
            "crop_siglip_dino_prob": _round_or_na(crop_prob),
            "ocr_distilbert_combined": _round_or_na(txt.get("ocr_distilbert_combined")),
            "ocr_distilbert_heuristic": _round_or_na(txt.get("ocr_distilbert_heuristic")),
            "ocr_distilbert_model": _round_or_na(txt.get("ocr_distilbert_model")),
        })

    print(f"  Matched: {matched}/{len(split_rows)} images with fusion scores")
    return out_rows


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fusion_root = Path(args.fusion_root)

    fused_embed_dir = fusion_root / "fusion_concat_siglip" / "embeddings"
    fused_model_dir = fusion_root / "fusion_concat_siglip" / "models"
    crop_embed_dir = fusion_root / "grounding_dino_crop_siglip" / "embeddings"
    crop_model_dir = fusion_root / "grounding_dino_crop_siglip" / "models"
    split_dir = fusion_root / "split"

    fused_clf_path = fused_model_dir / "lightgbm.joblib"
    crop_clf_path = crop_model_dir / "lightgbm.joblib"
    for p in (fused_clf_path, crop_clf_path):
        if not p.exists():
            print(f"ERROR: Required classifier not found at {p}")
            return 1

    cfg = Config()
    text_model_path = args.text_model or args.text_model_path
    if text_model_path:
        cfg.TEXT_PHISHING_MODEL_NAME = text_model_path

    ollama_model = args.ollama_model or cfg.OCR_OLLAMA_MODEL
    ollama_host = args.ollama_host or cfg.OCR_OLLAMA_HOST
    ollama_timeout = (
        args.ollama_timeout_seconds
        if args.ollama_timeout_seconds is not None
        else cfg.OCR_OLLAMA_TIMEOUT_SECONDS
    )

    print(f"Model A (fuse_siglip_DINO): fusion_concat_siglip + lightgbm")
    print(f"  + guardrail signal: grounding_dino_crop_siglip + lightgbm (crop_siglip_dino_prob)")
    if args.ocr_csv:
        print(f"Model B (ocr_ollama_distilbert): saved OCR CSV + DistilBERT")
    else:
        print(f"Model B (ocr_ollama_distilbert): Ollama '{ollama_model}' @ {ollama_host} + DistilBERT")

    ocr = OCRService(
        languages=list(cfg.OCR_LANGUAGES),
        gpu=(cfg.DEVICE == "cuda"),
        backend="ollama",
        ollama_model=ollama_model,
        ollama_host=ollama_host,
        ollama_timeout_seconds=ollama_timeout,
        ollama_clean_output=cfg.OCR_OLLAMA_CLEAN_OUTPUT,
    )
    processor = OCRTextProcessor(cfg)
    analyzer = TextRiskAnalyzer(cfg)
    saved_ocr_by_path: dict[str, str] | None = None
    saved_ocr_by_name: dict[str, str] | None = None
    if args.ocr_csv:
        ocr_csv = Path(args.ocr_csv)
        if not ocr_csv.exists():
            print(f"ERROR: OCR CSV not found at {ocr_csv}")
            return 1
        saved_ocr_by_path, saved_ocr_by_name = load_saved_ocr(ocr_csv)
        print(
            f"  Loaded saved OCR cache: path_keys={len(saved_ocr_by_path)}, "
            f"filename_keys={len(saved_ocr_by_name)}"
        )

    for split_name in ["train", "test"]:
        split_csv = split_dir / f"{split_name}_samples.csv"
        fused_embed_file = fused_embed_dir / f"{split_name}_embeddings.npz"
        crop_embed_file = crop_embed_dir / f"{split_name}_embeddings.npz"

        if not split_csv.exists():
            print(f"WARNING: {split_csv} not found, skipping {split_name}")
            continue
        missing = [p for p in (fused_embed_file, crop_embed_file) if not p.exists()]
        if missing:
            print(f"WARNING: missing embeddings for {split_name}: {missing}, skipping")
            continue

        split_rows = load_split_csv(split_csv)
        fusion_scores = load_fusion_scores(fused_embed_file, fused_clf_path)
        crop_scores = load_fusion_scores(crop_embed_file, crop_clf_path)
        rows = build_split(
            split_name=split_name,
            split_rows=split_rows,
            fusion_scores=fusion_scores,
            crop_scores=crop_scores,
            ocr=ocr,
            processor=processor,
            analyzer=analyzer,
            saved_ocr_by_path=saved_ocr_by_path,
            saved_ocr_by_name=saved_ocr_by_name,
            allow_partial_ocr=args.allow_partial_ocr,
        )

        out_path = output_dir / f"{split_name}_scores.csv"
        write_csv(out_path, rows)
        print(f"  Saved: {out_path} ({len(rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
