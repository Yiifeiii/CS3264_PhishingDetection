"""Extract sub-model scores for the Bayesian ensemble inputs.

This script supports two image-branch artifact layouts:

1. The legacy Grounding DINO fusion layout:
   ``outputs/chat_social_grounding_dino_fusion/...``
2. The current lightweight layout in this repo:
   ``outputs/embeddings`` + ``outputs/models``

For the text branch, you can either:

1. Re-run live Ollama OCR on each image, or
2. Reuse a saved OCR CSV via ``--ocr-csv`` plus ``--manifest-csv``.

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
from pathlib import Path, PurePosixPath

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

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
LEGACY_FUSION_ROOT = PROJECT_ROOT / "outputs" / "chat_social_grounding_dino_fusion"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract ensemble feature scores.")
    p.add_argument(
        "--fusion-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=(
            "Root containing the image-branch embeddings/models. Supports both the "
            "legacy chat_social_grounding_dino_fusion layout and the current outputs layout."
        ),
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
        "--text-model",
        dest="text_model_path",
        default=None,
        help="DistilBERT checkpoint path (default: Config.TEXT_PHISHING_MODEL_NAME).",
    )
    p.add_argument(
        "--ocr-csv",
        default=None,
        help=(
            "Optional saved Ollama OCR CSV with path/model/text/error columns. "
            "When provided, step 1 reuses these OCR texts instead of calling Ollama live."
        ),
    )
    p.add_argument(
        "--manifest-csv",
        default=None,
        help=(
            "Optional split manifest with split,label,source,name,original_path columns. "
            "Required with --ocr-csv, and also used to map outputs/embeddings rows "
            "back to original image paths."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="artifacts/ensemble",
        help="Directory for output CSVs.",
    )
    return p.parse_args()


def normalize_path_key(value: str) -> str:
    return str(value or "").replace("/", "\\").strip().lower()


def basename_key(value: str) -> str:
    normalized = str(value or "").replace("\\", "/")
    return PurePosixPath(normalized).name.lower()


def manifest_slug(row: dict) -> str:
    source = str(row.get("source") or "").strip()
    label = str(row.get("label") or "").strip()
    name = str(row.get("name") or "").strip()
    if source and label and name:
        return f"{source}__{label}__{name}".lower()
    return basename_key(name)


def load_manifest_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def load_ocr_rows(csv_path: Path) -> dict[str, dict]:
    rows_by_path: dict[str, dict] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = str(row.get("path") or "").strip()
            if image_path:
                rows_by_path[normalize_path_key(image_path)] = row
    return rows_by_path


def load_fusion_scores(embed_file: Path, clf_path: Path) -> dict[str, tuple[int, float]]:
    """Return {stored_image_path: (label, fake_prob)} from embeddings + classifier."""
    data = np.load(embed_file, allow_pickle=True)
    X, y, paths = data["X"], data["y"], data["paths"]
    clf = joblib.load(clf_path)
    probs = clf.predict_proba(X)[:, 1]

    result: dict[str, tuple[int, float]] = {}
    for path_str, label, prob in zip(paths, y, probs):
        result[str(path_str)] = (int(label), float(prob))
    return result


def load_split_csv(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def build_rows_from_legacy_split(split_csv: Path) -> list[dict]:
    rows = []
    for row in load_split_csv(split_csv):
        image_path = str(row["image_path"])
        rows.append(
            {
                "fusion_lookup_key": image_path,
                "image_path": image_path,
                "text_source_path": image_path,
                "label_id": int(row["label_id"]),
                "filename": PurePosixPath(image_path.replace("\\", "/")).name,
            }
        )
    return rows


def build_rows_from_manifest_and_embeddings(
    split_name: str,
    embed_file: Path,
    manifest_rows: list[dict],
) -> tuple[list[dict], int]:
    data = np.load(embed_file, allow_pickle=True)
    paths = data["paths"]
    labels = data["y"]

    manifest_by_slug = {
        manifest_slug(row): row
        for row in manifest_rows
        if str(row.get("split") or "").strip().lower() == split_name
    }

    rows: list[dict] = []
    matched = 0
    unmatched = 0
    for stored_path, label in zip(paths, labels):
        stored_path_str = str(stored_path)
        filename = PurePosixPath(stored_path_str.replace("\\", "/")).name
        manifest_row = manifest_by_slug.get(filename.lower())
        if manifest_row is not None:
            matched += 1
            original_path = str(manifest_row.get("original_path") or stored_path_str)
        else:
            unmatched += 1
            original_path = stored_path_str

        rows.append(
            {
                "fusion_lookup_key": stored_path_str,
                "image_path": original_path,
                "text_source_path": original_path,
                "label_id": int(label),
                "filename": filename,
            }
        )

    if unmatched:
        print(
            f"  WARNING: {unmatched} {split_name} embedding row(s) could not be mapped "
            "back to manifest original_path values."
        )

    return rows, matched


def analyze_raw_text(raw_text: str, processor: OCRTextProcessor, analyzer: TextRiskAnalyzer) -> dict:
    processed = processor.process(raw_text or "")
    processed_text = str(processed.get("text") or "").strip()

    combined = heuristic = model_score = None
    if processed_text:
        analysis = analyzer.analyze(processed_text)
        model_score = analysis.get("model_score_raw")
        heuristic = analysis.get("rule_score")
        combined = analysis.get("score")

    return {
        "ocr_distilbert_combined": combined,
        "ocr_distilbert_heuristic": heuristic,
        "ocr_distilbert_model": model_score,
    }


@torch.no_grad()
def extract_live_ocr_distilbert_scores(
    rows: list[dict],
    ocr: OCRService,
    processor: OCRTextProcessor,
    analyzer: TextRiskAnalyzer,
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    total = len(rows)

    for i, row in enumerate(rows, 1):
        image_key = str(row["image_path"])
        text_source_path = str(row["text_source_path"])
        fname = str(row["filename"])
        print(f"    [{i}/{total}] Ollama OCR: {fname}", flush=True)

        try:
            raw_text = ocr.extract_text(text_source_path)
        except Exception as exc:
            print(f"      WARNING: OCR failed ({exc}); using empty text")
            raw_text = ""

        result[image_key] = analyze_raw_text(str(raw_text or ""), processor, analyzer)

    return result


@torch.no_grad()
def extract_saved_ocr_distilbert_scores(
    rows: list[dict],
    ocr_rows_by_path: dict[str, dict],
    processor: OCRTextProcessor,
    analyzer: TextRiskAnalyzer,
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    total = len(rows)

    for i, row in enumerate(rows, 1):
        image_key = str(row["image_path"])
        text_source_path = str(row["text_source_path"])
        fname = str(row["filename"])
        print(f"    [{i}/{total}] Saved OCR: {fname}", flush=True)

        ocr_row = ocr_rows_by_path.get(normalize_path_key(text_source_path))
        raw_text = ""
        if ocr_row is None:
            print(f"      WARNING: no OCR CSV row for {text_source_path}")
        else:
            error_text = str(ocr_row.get("error") or "").strip()
            if error_text:
                print(f"      WARNING: OCR CSV row has error for {fname}: {error_text}")
            else:
                raw_text = str(ocr_row.get("text") or "")

        result[image_key] = analyze_raw_text(raw_text, processor, analyzer)

    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round_or_na(val) -> object:
    if val is None:
        return ""
    return round(float(val), 6)


def build_split(
    split_name: str,
    split_rows: list[dict],
    fusion_scores: dict[str, tuple[int, float]],
    crop_scores: dict[str, tuple[int, float]],
    text_scores: dict[str, dict],
) -> list[dict]:
    print(f"\n=== Processing {split_name} split ({len(split_rows)} images) ===")

    out_rows: list[dict] = []
    matched = 0
    for row in split_rows:
        fusion_lookup_key = str(row["fusion_lookup_key"])
        image_path = str(row["image_path"])
        label = int(row["label_id"])
        fname = str(row["filename"])

        fusion = fusion_scores.get(fusion_lookup_key)
        fusion_prob = fusion[1] if fusion is not None else None
        if fusion_prob is None:
            print(f"  WARNING: no fusion score for {fname}, skipping")
            continue

        crop = crop_scores.get(fusion_lookup_key)
        crop_prob = crop[1] if crop is not None else None

        txt = text_scores.get(image_path, {})
        matched += 1
        out_rows.append(
            {
                "image_path": image_path,
                "filename": fname,
                "label": label,
                "fuse_siglip_dino_prob": _round_or_na(fusion_prob),
                "crop_siglip_dino_prob": _round_or_na(crop_prob),
                "ocr_distilbert_combined": _round_or_na(txt.get("ocr_distilbert_combined")),
                "ocr_distilbert_heuristic": _round_or_na(txt.get("ocr_distilbert_heuristic")),
                "ocr_distilbert_model": _round_or_na(txt.get("ocr_distilbert_model")),
            }
        )

    print(f"  Matched: {matched}/{len(split_rows)} images with fusion scores")
    return out_rows


def resolve_image_layout(fusion_root: Path) -> dict[str, object]:
    legacy_layout = {
        "layout": "legacy",
        "root": fusion_root,
        "split_dir": fusion_root / "split",
        "fused_embed_dir": fusion_root / "fusion_concat_siglip" / "embeddings",
        "fused_clf_path": fusion_root / "fusion_concat_siglip" / "models" / "lightgbm.joblib",
        "crop_embed_dir": fusion_root / "grounding_dino_crop_siglip" / "embeddings",
        "crop_clf_path": fusion_root / "grounding_dino_crop_siglip" / "models" / "lightgbm.joblib",
    }
    if (
        legacy_layout["fused_embed_dir"].exists()
        and Path(legacy_layout["fused_clf_path"]).exists()
    ):
        return legacy_layout

    simple_layout = {
        "layout": "simple",
        "root": fusion_root,
        "split_dir": None,
        "fused_embed_dir": fusion_root / "embeddings",
        "fused_clf_path": fusion_root / "models" / "lightgbm.joblib",
        "crop_embed_dir": None,
        "crop_clf_path": None,
    }
    if (
        Path(simple_layout["fused_embed_dir"]).exists()
        and Path(simple_layout["fused_clf_path"]).exists()
    ):
        return simple_layout

    if fusion_root != LEGACY_FUSION_ROOT and LEGACY_FUSION_ROOT.exists():
        return resolve_image_layout(LEGACY_FUSION_ROOT)

    raise FileNotFoundError(
        "Could not find supported image-branch artifacts. "
        f"Tried '{fusion_root}' and legacy fallback '{LEGACY_FUSION_ROOT}'."
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fusion_root = Path(args.fusion_root)

    try:
        layout = resolve_image_layout(fusion_root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    fused_clf_path = Path(layout["fused_clf_path"])
    if not fused_clf_path.exists():
        print(f"ERROR: Required classifier not found at {fused_clf_path}")
        return 1

    cfg = Config()
    if args.text_model_path:
        cfg.TEXT_PHISHING_MODEL_NAME = args.text_model_path

    manifest_rows = None
    if args.manifest_csv:
        manifest_csv = Path(args.manifest_csv)
        if not manifest_csv.exists():
            print(f"ERROR: manifest CSV not found at {manifest_csv}")
            return 1
        manifest_rows = load_manifest_rows(manifest_csv)

    ocr_rows_by_path = None
    if args.ocr_csv:
        if manifest_rows is None:
            print("ERROR: --ocr-csv requires --manifest-csv so OCR rows can be mapped to splits.")
            return 1
        ocr_csv = Path(args.ocr_csv)
        if not ocr_csv.exists():
            print(f"ERROR: OCR CSV not found at {ocr_csv}")
            return 1
        ocr_rows_by_path = load_ocr_rows(ocr_csv)

    if layout["layout"] == "simple" and manifest_rows is None:
        print(
            "ERROR: the current outputs/embeddings layout requires --manifest-csv so train/test "
            "embedding rows can be mapped back to original image paths."
        )
        return 1

    ollama_model = args.ollama_model or cfg.OCR_OLLAMA_MODEL
    ollama_host = args.ollama_host or cfg.OCR_OLLAMA_HOST
    ollama_timeout = (
        args.ollama_timeout_seconds
        if args.ollama_timeout_seconds is not None
        else cfg.OCR_OLLAMA_TIMEOUT_SECONDS
    )

    print(f"Model A (fuse_siglip_DINO): {layout['layout']} layout @ {layout['root']}")
    crop_clf_path = layout["crop_clf_path"]
    if crop_clf_path and Path(crop_clf_path).exists():
        print("  + guardrail signal: grounding_dino_crop_siglip + lightgbm (crop_siglip_dino_prob)")
    else:
        print("  + guardrail signal unavailable in this layout; crop_siglip_dino_prob will be blank")

    if ocr_rows_by_path is not None:
        print(
            "Model B (ocr_ollama_distilbert): saved OCR CSV + DistilBERT "
            f"(text model: {cfg.TEXT_PHISHING_MODEL_NAME})"
        )
    else:
        print(
            "Model B (ocr_ollama_distilbert): "
            f"Ollama '{ollama_model}' @ {ollama_host} + DistilBERT"
        )

    processor = OCRTextProcessor(cfg)
    analyzer = TextRiskAnalyzer(cfg)

    ocr = None
    if ocr_rows_by_path is None:
        ocr = OCRService(
            languages=list(cfg.OCR_LANGUAGES),
            gpu=(cfg.DEVICE == "cuda"),
            backend="ollama",
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            ollama_timeout_seconds=ollama_timeout,
            ollama_clean_output=cfg.OCR_OLLAMA_CLEAN_OUTPUT,
        )

    fused_embed_dir = Path(layout["fused_embed_dir"])
    crop_embed_dir = Path(layout["crop_embed_dir"]) if layout["crop_embed_dir"] else None

    for split_name in ["train", "test"]:
        fused_embed_file = fused_embed_dir / f"{split_name}_embeddings.npz"
        if not fused_embed_file.exists():
            print(f"WARNING: {fused_embed_file} not found, skipping {split_name}")
            continue

        if manifest_rows is not None:
            split_rows, manifest_matches = build_rows_from_manifest_and_embeddings(
                split_name=split_name,
                embed_file=fused_embed_file,
                manifest_rows=manifest_rows,
            )
            coverage = manifest_matches / max(len(split_rows), 1)
            print(
                f"  Manifest mapping coverage for {split_name}: "
                f"{manifest_matches}/{len(split_rows)} ({coverage:.1%})"
            )
            if ocr_rows_by_path is not None and coverage < 0.95:
                print(
                    "ERROR: manifest mapping coverage is too low for saved OCR reuse. "
                    "The image-branch split does not match the provided manifest. "
                    "Regenerate matching image-branch artifacts or use a manifest/OCR CSV "
                    "from the same split as the embeddings."
                )
                return 1
        else:
            split_dir = Path(layout["split_dir"])
            split_csv = split_dir / f"{split_name}_samples.csv"
            if not split_csv.exists():
                print(f"WARNING: {split_csv} not found, skipping {split_name}")
                continue
            split_rows = build_rows_from_legacy_split(split_csv)

        fusion_scores = load_fusion_scores(fused_embed_file, fused_clf_path)
        crop_scores: dict[str, tuple[int, float]] = {}
        if crop_embed_dir is not None and crop_clf_path and Path(crop_clf_path).exists():
            crop_embed_file = crop_embed_dir / f"{split_name}_embeddings.npz"
            if crop_embed_file.exists():
                crop_scores = load_fusion_scores(crop_embed_file, Path(crop_clf_path))
            else:
                print(f"  WARNING: crop embeddings missing for {split_name}; leaving crop scores blank")

        if ocr_rows_by_path is not None:
            text_scores = extract_saved_ocr_distilbert_scores(
                rows=split_rows,
                ocr_rows_by_path=ocr_rows_by_path,
                processor=processor,
                analyzer=analyzer,
            )
        else:
            assert ocr is not None
            text_scores = extract_live_ocr_distilbert_scores(
                rows=split_rows,
                ocr=ocr,
                processor=processor,
                analyzer=analyzer,
            )

        rows = build_split(
            split_name=split_name,
            split_rows=split_rows,
            fusion_scores=fusion_scores,
            crop_scores=crop_scores,
            text_scores=text_scores,
        )

        out_path = output_dir / f"{split_name}_scores.csv"
        write_csv(out_path, rows)
        print(f"  Saved: {out_path} ({len(rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
