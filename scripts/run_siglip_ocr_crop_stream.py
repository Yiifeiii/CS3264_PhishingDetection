import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGLIP_ROOT = PROJECT_ROOT / "siglip"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SIGLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(SIGLIP_ROOT))

from ocr.ocr_service import OCRService  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.image_loading import load_image_rgb  # noqa: E402
from classifier_utils import compute_metrics, predict_positive_scores  # noqa: E402
from feature_utils import get_normalized_image_features  # noqa: E402
from hf_utils import load_siglip_processor_and_model  # noqa: E402


CLASS_NAMES = ("real", "fake")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
URL_PATTERN = re.compile(r"(https?://|www\.|[A-Za-z0-9-]+\.[A-Za-z]{2,})", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_PATTERN = re.compile(r"(?:\+\d[\d\s-]{5,}\d|\b\d[\d\s-]{5,}\d\b)")
MONEY_PATTERN = re.compile(r"(\$\s?\d+|\b\d+\s?(?:usd|sgd|rm)\b)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run an OCR-crop SigLIP inference stream over one or more labeled source roots. "
            "Every OCR crop is saved locally, scored, and aggregated per image by max fake probability."
        )
    )
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        help="Labeled source dataset root. Repeat to include multiple sources.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Run OCR-crop inference on a single image path.",
    )
    parser.add_argument(
        "--existing-crops-root",
        default=None,
        help=(
            "Reuse an existing OCR crop directory that already contains per-image "
            "metadata.json and crop_*.png files."
        ),
    )
    parser.add_argument(
        "--classifier-path",
        default="outputs/fixed_split_cv_crop/final_models/xgboost.joblib",
        help="Path to the trained cropped-image classifier checkpoint.",
    )
    parser.add_argument(
        "--model-name",
        default="artifacts/siglip2-base-patch16-224",
        help="SigLIP model name or local artifact path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device override: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--languages",
        default=",".join(Config.OCR_LANGUAGES),
        help="Comma-separated EasyOCR languages.",
    )
    parser.add_argument(
        "--ocr-passes",
        choices=("original", "both"),
        default="original",
        help=(
            "OCR detection passes to use. `original` is much faster. `both` also runs OCR on "
            "the preprocessed image and usually produces more duplicate crops."
        ),
    )
    parser.add_argument(
        "--crop-output-root",
        default="outputs/ocr_crop_stream/crops",
        help="Directory where OCR crop images and metadata will be written.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/ocr_crop_stream/inference",
        help="Directory where inference CSVs and reports will be written.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum OCR confidence for keeping a raw text box.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.05,
        help="Padding ratio applied around each crop candidate.",
    )
    parser.add_argument(
        "--max-crops-per-image",
        type=int,
        default=0,
        help="Optional cap on saved/scored crop candidates per image. Use 0 for no cap.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("blocks_preferred", "blocks_only", "all"),
        default="blocks_preferred",
        help=(
            "How to mix grouped OCR blocks and single OCR detections. "
            "`blocks_preferred` uses grouped blocks when available and falls back to singles only "
            "when no block can be formed."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for SigLIP crop embedding.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the crop and inference output roots before running.",
    )
    parser.add_argument(
        "--save-overview",
        action="store_true",
        help="Save an overview image with all OCR crop candidates drawn.",
    )
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_languages(raw_languages: str):
    languages = [part.strip() for part in raw_languages.split(",") if part.strip()]
    return languages or list(Config.OCR_LANGUAGES)


def collect_class_files(source_root: Path, class_name: str):
    class_dir = source_root / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Expected class directory not found: {class_dir}")

    files = [
        path.resolve()
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def collect_existing_crop_entries(existing_crops_root: Path):
    metadata_files = sorted(existing_crops_root.rglob("metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(
            f"No metadata.json files were found under {existing_crops_root}"
        )

    entries = []
    for metadata_path in metadata_files:
        with open(metadata_path) as handle:
            metadata = json.load(handle)

        candidates = []
        for candidate in metadata.get("candidates", []):
            crop_path_value = candidate.get("crop_path")
            if not crop_path_value:
                continue
            crop_path = Path(crop_path_value)
            if not crop_path.is_absolute():
                crop_path = (metadata_path.parent / crop_path).resolve()
            if not crop_path.exists():
                continue

            candidate_copy = dict(candidate)
            candidate_copy["crop_path"] = str(crop_path)
            candidates.append(candidate_copy)

        if not candidates:
            continue

        source_name = metadata.get("source_name") or metadata_path.parents[2].name
        class_name = metadata.get("class_name") or metadata_path.parents[1].name
        true_label_id = metadata.get("true_label_id")
        if true_label_id is None:
            true_label_id = CLASS_TO_ID.get(class_name)

        entries.append(
            {
                "source_name": source_name,
                "class_name": class_name,
                "label_id": int(true_label_id) if true_label_id is not None else -1,
                "image_path": Path(metadata.get("image_path", metadata_path.parent.name)),
                "image_slug": metadata_path.parent.name,
                "metadata_path": metadata_path,
                "candidates": candidates,
                "regions": metadata.get("regions", []),
                "image_width": metadata.get("image_width"),
                "image_height": metadata.get("image_height"),
            }
        )

    if not entries:
        raise ValueError(
            f"No usable crop candidates were found under {existing_crops_root}"
        )

    return entries


def safe_slug(path: Path):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    if not stem:
        stem = "image"
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{stem[:48]}__{digest}"


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def interval_overlap(a1, a2, b1, b2):
    return max(0, min(a2, b2) - max(a1, b1))


def interval_gap(a1, a2, b1, b2):
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0


def bbox_iou(a_bbox, b_bbox):
    ax1, ay1, ax2, ay2 = a_bbox
    bx1, by1, bx2, by2 = b_bbox
    inter_w = interval_overlap(ax1, ax2, bx1, bx2)
    inter_h = interval_overlap(ay1, ay2, by1, by2)
    inter = inter_w * inter_h
    union = bbox_area(a_bbox) + bbox_area(b_bbox) - inter
    if union <= 0:
        return 0.0
    return inter / union


def normalize_text(text: str):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def texts_equivalent(a_text: str, b_text: str):
    a_norm = normalize_text(a_text)
    b_norm = normalize_text(b_text)
    if not a_norm or not b_norm:
        return False
    return a_norm == b_norm or a_norm in b_norm or b_norm in a_norm


def clamp_and_pad_bbox(bbox, image_width: int, image_height: int, padding_ratio: float):
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    pad_x = max(int(round(width * padding_ratio)), 6)
    pad_y = max(int(round(height * padding_ratio)), 6)
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    ]


def text_signal_flags(text: str):
    normalized = normalize_text(text)
    compact = re.sub(r"\s+", "", normalized)
    word_count = len([part for part in normalized.split(" ") if part])
    return {
        "normalized": normalized,
        "char_count": len(compact),
        "word_count": word_count,
        "has_url": bool(URL_PATTERN.search(normalized)),
        "has_email": bool(EMAIL_PATTERN.search(normalized)),
        "has_phone": bool(PHONE_PATTERN.search(normalized)),
        "has_money": bool(MONEY_PATTERN.search(normalized)),
    }


def should_keep_single_region(region, image_width: int, image_height: int):
    flags = text_signal_flags(region["text"])
    area_ratio = region["area"] / max(image_width * image_height, 1)
    has_strong_signal = (
        flags["has_url"]
        or flags["has_email"]
        or flags["has_phone"]
        or flags["has_money"]
    )
    has_meaningful_span = (
        flags["char_count"] >= 18
        or flags["word_count"] >= 4
        or (flags["char_count"] >= 10 and region["confidence"] >= 0.55)
    )
    has_visible_size = area_ratio >= 0.0012 or region["height"] >= 20 or region["width"] >= 100
    return has_strong_signal or (has_meaningful_span and has_visible_size)


def should_keep_block_candidate(candidate, image_width: int, image_height: int):
    merged_text = " ".join(candidate.get("texts", []))
    flags = text_signal_flags(merged_text)
    area_ratio = bbox_area(candidate["bbox"]) / max(image_width * image_height, 1)
    has_strong_signal = (
        flags["has_url"]
        or flags["has_email"]
        or flags["has_phone"]
        or flags["has_money"]
    )
    has_meaningful_span = (
        flags["char_count"] >= 14
        or flags["word_count"] >= 3
        or (
            candidate.get("ocr_region_count", 0) >= 4
            and flags["char_count"] >= 8
        )
    )
    has_visible_size = area_ratio >= 0.002 or candidate.get("ocr_region_count", 0) >= 4
    return has_strong_signal or (has_meaningful_span and has_visible_size)


def dedupe_regions(regions, iou_threshold: float = 0.75):
    ordered = sorted(
        regions,
        key=lambda region: (
            region["confidence"],
            len(region["text"]),
            region["area"],
        ),
        reverse=True,
    )
    kept = []
    for region in ordered:
        duplicate = False
        for existing in kept:
            overlap = bbox_iou(region["bbox"], existing["bbox"])
            if overlap >= iou_threshold and texts_equivalent(region["text"], existing["text"]):
                duplicate = True
                break
        if not duplicate:
            kept.append(region)
    return kept


def boxes_connected(a_region, b_region, image_width: int, image_height: int):
    ax1, ay1, ax2, ay2 = a_region["bbox"]
    bx1, by1, bx2, by2 = b_region["bbox"]

    vertical_gap = interval_gap(ay1, ay2, by1, by2)
    horizontal_gap = interval_gap(ax1, ax2, bx1, bx2)
    horizontal_overlap = interval_overlap(ax1, ax2, bx1, bx2)
    vertical_overlap = interval_overlap(ay1, ay2, by1, by2)

    min_width = max(1, min(ax2 - ax1, bx2 - bx1))
    min_height = max(1, min(ay2 - ay1, by2 - by1))
    horizontal_overlap_ratio = horizontal_overlap / min_width
    vertical_overlap_ratio = vertical_overlap / min_height

    left_aligned = abs(ax1 - bx1) <= max(20, int(round(image_width * 0.08)))
    right_aligned = abs(ax2 - bx2) <= max(20, int(round(image_width * 0.08)))
    center_aligned = abs(((ax1 + ax2) / 2) - ((bx1 + bx2) / 2)) <= max(
        24,
        int(round(image_width * 0.1)),
    )

    stacked_block = (
        vertical_gap <= max(18, int(round(image_height * 0.04)))
        and (horizontal_overlap_ratio >= 0.25 or left_aligned or right_aligned or center_aligned)
    )
    same_line = (
        horizontal_gap <= max(18, int(round(image_width * 0.04)))
        and vertical_overlap_ratio >= 0.55
    )
    return stacked_block or same_line


def build_block_candidates(regions, image_width: int, image_height: int):
    if len(regions) < 2:
        return []

    adjacency = {idx: set() for idx in range(len(regions))}
    for left_idx in range(len(regions)):
        for right_idx in range(left_idx + 1, len(regions)):
            if boxes_connected(regions[left_idx], regions[right_idx], image_width, image_height):
                adjacency[left_idx].add(right_idx)
                adjacency[right_idx].add(left_idx)

    visited = set()
    block_candidates = []
    for start_idx in range(len(regions)):
        if start_idx in visited:
            continue

        stack = [start_idx]
        component = []
        while stack:
            idx = stack.pop()
            if idx in visited:
                continue
            visited.add(idx)
            component.append(regions[idx])
            stack.extend(adjacency[idx] - visited)

        if len(component) < 2:
            continue

        x1 = min(region["bbox"][0] for region in component)
        y1 = min(region["bbox"][1] for region in component)
        x2 = max(region["bbox"][2] for region in component)
        y2 = max(region["bbox"][3] for region in component)
        text_parts = [region["text"] for region in component if region["text"]]
        confidences = [region["confidence"] for region in component]
        total_text_len = sum(len(text) for text in text_parts)
        area_ratio = bbox_area([x1, y1, x2, y2]) / max(image_width * image_height, 1)

        block_candidates.append(
            {
                "candidate_type": "block",
                "bbox": [x1, y1, x2, y2],
                "texts": text_parts,
                "ocr_confidence_mean": float(np.mean(confidences)),
                "ocr_confidence_max": float(np.max(confidences)),
                "ocr_region_count": len(component),
                "score_hint": float(
                    np.sum(confidences) + min(total_text_len / 40.0, 4.0) + area_ratio * 3.0
                ),
                "sources": sorted({region["source"] for region in component}),
            }
        )

    return block_candidates


def candidates_connected(a_candidate, b_candidate, image_width: int, image_height: int):
    ax1, ay1, ax2, ay2 = a_candidate["bbox"]
    bx1, by1, bx2, by2 = b_candidate["bbox"]

    vertical_gap = interval_gap(ay1, ay2, by1, by2)
    horizontal_overlap = interval_overlap(ax1, ax2, bx1, bx2)
    min_width = max(1, min(ax2 - ax1, bx2 - bx1))
    overlap_ratio = horizontal_overlap / min_width

    left_aligned = abs(ax1 - bx1) <= max(24, int(round(image_width * 0.08)))
    right_aligned = abs(ax2 - bx2) <= max(24, int(round(image_width * 0.08)))
    center_aligned = abs(((ax1 + ax2) / 2) - ((bx1 + bx2) / 2)) <= max(
        28,
        int(round(image_width * 0.1)),
    )

    text_chars_a = sum(len(text) for text in a_candidate.get("texts", []))
    text_chars_b = sum(len(text) for text in b_candidate.get("texts", []))
    if max(text_chars_a, text_chars_b) < 12:
        return False

    return (
        vertical_gap <= max(36, int(round(image_height * 0.05)))
        and (overlap_ratio >= 0.45 or left_aligned or right_aligned or center_aligned)
    )


def merge_block_candidates(block_candidates, image_width: int, image_height: int):
    if len(block_candidates) < 2:
        return block_candidates

    adjacency = {idx: set() for idx in range(len(block_candidates))}
    for left_idx in range(len(block_candidates)):
        for right_idx in range(left_idx + 1, len(block_candidates)):
            if candidates_connected(
                block_candidates[left_idx],
                block_candidates[right_idx],
                image_width=image_width,
                image_height=image_height,
            ):
                adjacency[left_idx].add(right_idx)
                adjacency[right_idx].add(left_idx)

    visited = set()
    merged = []
    for start_idx in range(len(block_candidates)):
        if start_idx in visited:
            continue

        stack = [start_idx]
        component = []
        while stack:
            idx = stack.pop()
            if idx in visited:
                continue
            visited.add(idx)
            component.append(block_candidates[idx])
            stack.extend(adjacency[idx] - visited)

        if len(component) == 1:
            merged.append(component[0])
            continue

        x1 = min(candidate["bbox"][0] for candidate in component)
        y1 = min(candidate["bbox"][1] for candidate in component)
        x2 = max(candidate["bbox"][2] for candidate in component)
        y2 = max(candidate["bbox"][3] for candidate in component)

        texts = []
        confidences = []
        sources = set()
        region_count = 0
        score_hint = 0.0
        for candidate in component:
            texts.extend(candidate.get("texts", []))
            confidences.extend([
                candidate.get("ocr_confidence_mean", 0.0),
                candidate.get("ocr_confidence_max", 0.0),
            ])
            sources.update(candidate.get("sources", []))
            region_count += candidate.get("ocr_region_count", 0)
            score_hint += candidate.get("score_hint", 0.0)

        merged.append(
            {
                "candidate_type": "block",
                "bbox": [x1, y1, x2, y2],
                "texts": texts,
                "ocr_confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
                "ocr_confidence_max": float(np.max(confidences)) if confidences else 0.0,
                "ocr_region_count": region_count,
                "score_hint": score_hint + 1.5,
                "sources": sorted(sources),
            }
        )

    return merged


def dedupe_candidates(candidates, iou_threshold: float = 0.92):
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate["score_hint"],
            candidate["ocr_region_count"],
            bbox_area(candidate["bbox"]),
        ),
        reverse=True,
    )
    kept = []
    for candidate in ordered:
        duplicate = False
        for existing in kept:
            if bbox_iou(candidate["bbox"], existing["bbox"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def generate_crop_candidates(
    regions,
    image_width: int,
    image_height: int,
    padding_ratio: float,
    candidate_mode: str,
):
    unique_regions = dedupe_regions(regions)
    single_candidates = []
    for region in unique_regions:
        if not should_keep_single_region(region, image_width=image_width, image_height=image_height):
            continue
        area_ratio = region["area"] / max(image_width * image_height, 1)
        padded_bbox = clamp_and_pad_bbox(region["bbox"], image_width, image_height, padding_ratio)
        single_candidates.append(
            {
                "candidate_type": "single",
                "bbox": padded_bbox,
                "texts": [region["text"]],
                "ocr_confidence_mean": region["confidence"],
                "ocr_confidence_max": region["confidence"],
                "ocr_region_count": 1,
                "score_hint": float(
                    region["confidence"] + min(len(region["text"]) / 30.0, 3.0) + area_ratio * 2.0
                ),
                "sources": [region["source"]],
            }
        )

    block_candidates = build_block_candidates(unique_regions, image_width, image_height)
    block_candidates = [
        candidate
        for candidate in block_candidates
        if should_keep_block_candidate(candidate, image_width=image_width, image_height=image_height)
    ]
    block_candidates = merge_block_candidates(
        block_candidates,
        image_width=image_width,
        image_height=image_height,
    )
    block_candidates = [
        candidate
        for candidate in block_candidates
        if should_keep_block_candidate(candidate, image_width=image_width, image_height=image_height)
    ]
    for candidate in block_candidates:
        candidate["bbox"] = clamp_and_pad_bbox(
            candidate["bbox"],
            image_width,
            image_height,
            padding_ratio * 1.25,
        )

    if candidate_mode == "all":
        candidates = dedupe_candidates(block_candidates + single_candidates)
    elif candidate_mode == "blocks_only":
        candidates = dedupe_candidates(block_candidates)
        if not candidates:
            candidates = dedupe_candidates(single_candidates)
    else:
        candidates = dedupe_candidates(block_candidates)
        if not candidates:
            candidates = dedupe_candidates(single_candidates)

    if not candidates:
        candidates = [
            {
                "candidate_type": "fallback_full_image",
                "bbox": [0, 0, image_width, image_height],
                "texts": [],
                "ocr_confidence_mean": 0.0,
                "ocr_confidence_max": 0.0,
                "ocr_region_count": 0,
                "score_hint": 0.0,
                "sources": [],
            }
        ]
    return candidates, unique_regions


@torch.no_grad()
def embed_pil_images(images, processor, model, device: str, batch_size: int):
    if not images:
        raise ValueError("No crop images were provided for embedding.")

    embeddings = []
    for start_idx in range(0, len(images), batch_size):
        batch_images = images[start_idx:start_idx + batch_size]
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        features = get_normalized_image_features(model, inputs)
        embeddings.append(features.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def score_candidates_for_entry(entry, candidates, processor, model, clf, device: str, batch_size: int):
    crop_images = []
    for candidate in candidates:
        crop_image = Image.open(candidate["crop_path"]).convert("RGB")
        crop_images.append(crop_image)

    try:
        embeddings = embed_pil_images(
            crop_images,
            processor=processor,
            model=model,
            device=device,
            batch_size=batch_size,
        )
    finally:
        for crop_image in crop_images:
            crop_image.close()

    crop_scores = predict_positive_scores(clf, embeddings)
    crop_scores = np.asarray(crop_scores, dtype=float)
    selected_index = int(np.argmax(crop_scores))

    per_crop_rows = []
    for idx, candidate in enumerate(candidates):
        candidate["fake_probability"] = float(crop_scores[idx])
        candidate["pred_label_id"] = int(crop_scores[idx] >= 0.5)
        per_crop_rows.append(
            {
                "source_name": entry["source_name"],
                "class_name": entry["class_name"],
                "true_label_id": entry["label_id"],
                "image_path": str(entry["image_path"]),
                "image_slug": entry["image_slug"],
                "crop_index": idx + 1,
                "candidate_type": candidate.get("candidate_type", "unknown"),
                "bbox_x1": candidate["bbox"][0],
                "bbox_y1": candidate["bbox"][1],
                "bbox_x2": candidate["bbox"][2],
                "bbox_y2": candidate["bbox"][3],
                "ocr_region_count": candidate.get("ocr_region_count", 0),
                "ocr_confidence_mean": candidate.get("ocr_confidence_mean", 0.0),
                "ocr_confidence_max": candidate.get("ocr_confidence_max", 0.0),
                "fake_probability": candidate["fake_probability"],
                "pred_label_id": candidate["pred_label_id"],
                "crop_path": candidate["crop_path"],
                "texts": " || ".join(candidate.get("texts", [])),
            }
        )

    max_prob = float(crop_scores[selected_index])
    pred_label_id = int(max_prob >= 0.5)
    selected_candidate = candidates[selected_index]
    per_image_row = {
        "source_name": entry["source_name"],
        "class_name": entry["class_name"],
        "true_label_id": entry["label_id"],
        "pred_label_id": pred_label_id,
        "correct": int(entry["label_id"] >= 0 and pred_label_id == entry["label_id"]),
        "image_path": str(entry["image_path"]),
        "image_slug": entry["image_slug"],
        "crop_count": len(candidates),
        "max_fake_probability": max_prob,
        "selected_crop_index": selected_index + 1,
        "selected_crop_type": selected_candidate.get("candidate_type", "unknown"),
        "selected_crop_path": selected_candidate["crop_path"],
        "metadata_path": str(entry["metadata_path"]) if entry.get("metadata_path") else "",
    }
    return per_crop_rows, per_image_row, selected_index, selected_candidate, candidates


def save_overview_image(image: Image.Image, candidates, overview_path: Path, selected_index: int):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for idx, candidate in enumerate(candidates):
        color = "#00cc66" if idx == selected_index else "#ff4d4d"
        draw.rectangle(candidate["bbox"], outline=color, width=3 if idx == selected_index else 2)
    canvas.save(overview_path)


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, classifier_path: Path, overall_metrics, per_source_rows, image_rows):
    total_images = len(image_rows)
    total_sources = len({row["source_name"] for row in image_rows})
    total_positive = int(sum(row["true_label_id"] == 1 for row in image_rows))
    predicted_positive = int(sum(row["pred_label_id"] == 1 for row in image_rows))

    lines = [
        "# OCR Crop Stream Report",
        "",
        "## Setup",
        f"- Classifier: `{classifier_path}`",
        f"- Sources: {total_sources}",
        f"- Images scored: {total_images}",
        f"- True fake images: {total_positive}",
        f"- Predicted fake images: {predicted_positive}",
        "",
        "## Overall Metrics",
    ]

    if overall_metrics:
        for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            value = overall_metrics.get(metric_name)
            if value is not None:
                lines.append(f"- {metric_name}: {value:.4f}")
    else:
        lines.append("- No labeled images were available for evaluation.")

    lines.extend(
        [
            "",
            "## Per-Source Metrics",
            "| Source | Images | Accuracy | Precision | Recall | F1 | ROC AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in per_source_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["source_name"],
                    str(row["images"]),
                    f"{row['accuracy']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1']:.4f}",
                    f"{row['roc_auc']:.4f}" if row.get("roc_auc") is not None else "n/a",
                ]
            )
            + " |"
        )

    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()

    source_roots = [resolve_path(raw_path) for raw_path in args.source_root]
    single_image_path = resolve_path(args.image) if args.image else None
    existing_crops_root = resolve_path(args.existing_crops_root) if args.existing_crops_root else None
    classifier_path = resolve_path(args.classifier_path)
    crop_output_root = resolve_path(args.crop_output_root)
    output_root = resolve_path(args.output_root)
    device = detect_device() if args.device == "auto" else args.device
    languages = parse_languages(args.languages)

    active_inputs = sum(
        [
            1 if source_roots else 0,
            1 if existing_crops_root is not None else 0,
            1 if single_image_path is not None else 0,
        ]
    )
    if active_inputs != 1:
        raise ValueError(
            "Specify exactly one input mode: --source-root, --existing-crops-root, or --image."
        )

    for source_root in source_roots:
        if not source_root.exists():
            raise FileNotFoundError(f"Source root does not exist: {source_root}")
        for class_name in CLASS_NAMES:
            if not (source_root / class_name).exists():
                raise FileNotFoundError(
                    f"Expected class directory not found: {source_root / class_name}"
                )
    if existing_crops_root is not None and not existing_crops_root.exists():
        raise FileNotFoundError(f"Existing crops root does not exist: {existing_crops_root}")
    if single_image_path is not None and not single_image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {single_image_path}")

    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")

    if args.clear_output:
        roots_to_clear = [output_root]
        if existing_crops_root is None:
            roots_to_clear.insert(0, crop_output_root)
        for root in roots_to_clear:
            if root.exists():
                shutil.rmtree(root)

    if existing_crops_root is None:
        crop_output_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print("OCR crop stream plan:")
    if single_image_path is not None:
        print(f"  mode: single_image")
        print(f"  image: {single_image_path}")
        print(f"  crop output root: {crop_output_root}")
    elif existing_crops_root is None:
        print(f"  mode: source_roots")
        print(f"  sources: {', '.join(root.name for root in source_roots)}")
        print(f"  crop output root: {crop_output_root}")
    else:
        print(f"  mode: existing_crops")
        print(f"  existing crops root: {existing_crops_root}")
    print(f"  classifier: {classifier_path}")
    print(f"  inference output root: {output_root}")
    if existing_crops_root is None:
        print(f"  ocr languages: {languages}")
        print(f"  ocr passes: {args.ocr_passes}")
        print(f"  candidate mode: {args.candidate_mode}")
    print(f"  max crops per image: {'unlimited' if args.max_crops_per_image <= 0 else args.max_crops_per_image}")

    ocr = OCRService(languages=languages, gpu=(device == "cuda")) if existing_crops_root is None else None
    processor, model = load_siglip_processor_and_model(args.model_name, device)
    clf = joblib.load(classifier_path)

    if single_image_path is not None:
        image_entries = [
            {
                "source_name": "single",
                "class_name": "unlabeled",
                "label_id": -1,
                "image_path": single_image_path,
            }
        ]
    elif existing_crops_root is None:
        image_entries = []
        for source_root in source_roots:
            for class_name in CLASS_NAMES:
                for image_path in collect_class_files(source_root, class_name):
                    image_entries.append(
                        {
                            "source_name": source_root.name,
                            "class_name": class_name,
                            "label_id": CLASS_TO_ID[class_name],
                            "image_path": image_path,
                        }
                    )
    else:
        image_entries = collect_existing_crop_entries(existing_crops_root)

    per_crop_rows = []
    per_image_rows = []

    for entry in tqdm(image_entries, desc="Running OCR crop stream"):
        if existing_crops_root is None:
            image_path = entry["image_path"]
            image = load_image_rgb(image_path)
            image_width, image_height = image.size

            regions = ocr.detect_text_regions(
                str(image_path),
                min_confidence=args.min_confidence,
                include_processed=(args.ocr_passes == "both"),
            )
            candidates, unique_regions = generate_crop_candidates(
                regions,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=args.padding_ratio,
                candidate_mode=args.candidate_mode,
            )

            if args.max_crops_per_image > 0:
                candidates = candidates[:args.max_crops_per_image]

            image_slug = safe_slug(image_path)
            image_output_dir = (
                crop_output_root
                / entry["source_name"]
                / entry["class_name"]
                / image_slug
            )
            image_output_dir.mkdir(parents=True, exist_ok=True)

            for idx, candidate in enumerate(candidates, start=1):
                bbox = tuple(candidate["bbox"])
                crop_image = image.crop(bbox)
                crop_path = image_output_dir / f"crop_{idx:03d}_{candidate['candidate_type']}.png"
                crop_image.save(crop_path)
                crop_image.close()
                candidate["crop_path"] = str(crop_path)

            entry["image_slug"] = image_slug
            entry["metadata_path"] = image_output_dir / "metadata.json"
        else:
            candidates = [dict(candidate) for candidate in entry["candidates"]]
            unique_regions = entry.get("regions", [])
            image = None
            image_output_dir = None

        crop_rows, image_row, selected_index, selected_candidate, scored_candidates = score_candidates_for_entry(
            entry,
            candidates,
            processor=processor,
            model=model,
            clf=clf,
            device=device,
            batch_size=args.batch_size,
        )
        per_crop_rows.extend(crop_rows)
        per_image_rows.append(image_row)

        if existing_crops_root is None and args.save_overview:
            overview_path = image_output_dir / "overview_boxes.png"
            save_overview_image(image, scored_candidates, overview_path, selected_index=selected_index)

        if existing_crops_root is None:
            metadata = {
                "source_name": entry["source_name"],
                "class_name": entry["class_name"],
                "true_label_id": entry["label_id"],
                "image_path": str(entry["image_path"]),
                "image_width": image_width,
                "image_height": image_height,
                "ocr_reader_languages": list(ocr.active_languages),
                "ocr_reader_warning": ocr.load_error,
                "regions": unique_regions,
                "candidates": scored_candidates,
                "selected_crop_index": selected_index + 1,
                "selected_crop_path": selected_candidate["crop_path"],
                "selected_crop_fake_probability": image_row["max_fake_probability"],
            }
            with open(entry["metadata_path"], "w") as handle:
                json.dump(metadata, handle, indent=2)
            image.close()

    labeled_rows = [row for row in per_image_rows if row["true_label_id"] in (0, 1)]
    if labeled_rows:
        y_true = np.asarray([row["true_label_id"] for row in labeled_rows], dtype=int)
        y_pred = np.asarray([row["pred_label_id"] for row in labeled_rows], dtype=int)
        y_prob = np.asarray([row["max_fake_probability"] for row in labeled_rows], dtype=float)
        overall_metrics = compute_metrics(y_true, y_pred, probs=y_prob)
    else:
        overall_metrics = {}

    per_source_rows = []
    for source_name in sorted({row["source_name"] for row in labeled_rows}):
        source_rows = [row for row in per_image_rows if row["source_name"] == source_name]
        source_true = np.asarray([row["true_label_id"] for row in source_rows], dtype=int)
        source_pred = np.asarray([row["pred_label_id"] for row in source_rows], dtype=int)
        source_prob = np.asarray([row["max_fake_probability"] for row in source_rows], dtype=float)
        source_metrics = compute_metrics(source_true, source_pred, probs=source_prob)
        per_source_rows.append(
            {
                "source_name": source_name,
                "images": len(source_rows),
                **source_metrics,
            }
        )

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        summary_dir / "per_crop_predictions.csv",
        per_crop_rows,
        fieldnames=[
            "source_name",
            "class_name",
            "true_label_id",
            "image_path",
            "image_slug",
            "crop_index",
            "candidate_type",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "ocr_region_count",
            "ocr_confidence_mean",
            "ocr_confidence_max",
            "fake_probability",
            "pred_label_id",
            "crop_path",
            "texts",
        ],
    )
    write_csv(
        summary_dir / "per_image_predictions.csv",
        per_image_rows,
        fieldnames=[
            "source_name",
            "class_name",
            "true_label_id",
            "pred_label_id",
            "correct",
            "image_path",
            "image_slug",
            "crop_count",
            "max_fake_probability",
            "selected_crop_index",
            "selected_crop_type",
            "selected_crop_path",
            "metadata_path",
        ],
    )
    write_csv(
        summary_dir / "per_source_metrics.csv",
        per_source_rows,
        fieldnames=[
            "source_name",
            "images",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ],
    )

    metrics_path = summary_dir / "overall_metrics.json"
    with open(metrics_path, "w") as handle:
        json.dump(overall_metrics, handle, indent=2)

    report_path = summary_dir / "final_report.md"
    write_report(
        report_path,
        classifier_path=classifier_path,
        overall_metrics=overall_metrics,
        per_source_rows=per_source_rows,
        image_rows=per_image_rows,
    )

    print("\nFinished OCR crop stream inference.")
    print(f"Per-crop predictions: {summary_dir / 'per_crop_predictions.csv'}")
    print(f"Per-image predictions: {summary_dir / 'per_image_predictions.csv'}")
    print(f"Per-source metrics: {summary_dir / 'per_source_metrics.csv'}")
    print(f"Overall metrics: {metrics_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
