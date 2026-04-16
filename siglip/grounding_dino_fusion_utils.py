import csv
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImagePathDataset
from feature_utils import get_normalized_image_features


CLASS_NAMES = ("real", "fake")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
DEFAULT_PROMPT_LABELS = (
    "message bubble",
    "chat message",
    "text message",
    "message block",
    "social media post",
    "post card",
    "caption text",
    "text overlay",
    "headline",
    "logo",
    "profile picture",
    "person",
    "face",
    "notification banner",
    "button",
    "qr code",
)
CHAT_PROMPT_LABELS = (
    "message bubble",
    "chat message",
    "text message",
    "sms",
    "message block",
    "conversation text",
    "notification banner",
    "link preview",
    "button",
    "qr code",
)
SOCIAL_PROMPT_LABELS = (
    "social media post",
    "post card",
    "caption text",
    "text overlay",
    "headline",
    "news card",
    "logo",
    "profile picture",
    "person",
    "face",
    "button",
    "qr code",
)
DEFAULT_SIGNAL_LABEL_WEIGHTS = {
    "message bubble": 3.8,
    "chat message": 3.7,
    "text message": 3.7,
    "sms": 3.5,
    "message block": 3.6,
    "conversation text": 3.4,
    "social media post": 2.8,
    "post card": 2.8,
    "caption text": 3.4,
    "text overlay": 3.2,
    "headline": 3.0,
    "news card": 2.7,
    "notification banner": 2.4,
    "link preview": 2.3,
    "qr code": 2.8,
    "button": 1.2,
    "logo": 1.8,
    "profile picture": 1.5,
    "person": 1.1,
    "face": 1.2,
}
CHAT_SIGNAL_LABEL_WEIGHTS = {
    **DEFAULT_SIGNAL_LABEL_WEIGHTS,
    "message bubble": 4.2,
    "chat message": 4.1,
    "text message": 4.1,
    "sms": 3.9,
    "message block": 4.0,
    "conversation text": 3.8,
    "notification banner": 2.6,
    "link preview": 2.6,
    "profile picture": 0.8,
    "person": 0.7,
    "face": 0.7,
}
SOCIAL_SIGNAL_LABEL_WEIGHTS = {
    **DEFAULT_SIGNAL_LABEL_WEIGHTS,
    "caption text": 3.8,
    "text overlay": 3.6,
    "headline": 3.4,
    "social media post": 3.2,
    "post card": 3.1,
    "news card": 3.0,
    "logo": 2.2,
    "profile picture": 1.8,
    "person": 1.3,
    "face": 1.3,
    "message bubble": 1.2,
    "chat message": 1.2,
    "text message": 1.2,
    "sms": 1.0,
    "message block": 1.1,
}
GROUNDING_SIGNAL_LABEL_REASONS = {
    "message bubble": "A message bubble usually contains the main scam text in chat screenshots.",
    "chat message": "A chat message usually contains the main scam text in chat screenshots.",
    "text message": "A text message usually contains the main scam text in chat screenshots.",
    "sms": "An SMS block usually contains the main scam text in chat screenshots.",
    "message block": "A message block concentrates the main scam text in chat screenshots.",
    "conversation text": "Conversation text concentrates the message content that drives the scam.",
    "social media post": "A social post card captures the combined caption and media context.",
    "post card": "A post card captures the combined caption and media context.",
    "caption text": "Caption text often carries the misleading claim or lure on social screenshots.",
    "text overlay": "Text overlay often carries the misleading claim directly on the media.",
    "headline": "A headline-like text block often carries the main claim being shared.",
    "news card": "A news-style card can capture the combined logo, media, and claim context.",
    "notification banner": "A notification-style banner can carry urgency or a fake platform alert.",
    "link preview": "A link-preview region can expose the lure destination in chat screenshots.",
    "qr code": "A QR code may redirect the victim off-platform to a malicious flow.",
    "button": "A button can be the phishing call-to-action, but is weak without text context.",
    "logo": "A logo can support impersonation or source framing in social screenshots.",
    "profile picture": "A profile picture helps identify the account framing the social post.",
    "person": "A person crop preserves the main subject of the social post media.",
    "face": "A face crop preserves the main subject of the social post media.",
}


def detect_device(preferred: str = "auto"):
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_holdout_per_bucket(total, per_bucket, bucket_count, split_name, default_per_bucket):
    if total is not None and per_bucket is not None:
        raise ValueError(
            f"Specify only one of --{split_name}-total or --{split_name}-per-bucket."
        )

    if total is not None:
        if total <= 0:
            raise ValueError(f"--{split_name}-total must be positive, got {total}.")
        if total % bucket_count != 0:
            raise ValueError(
                f"--{split_name}-total={total} is not divisible by {bucket_count} buckets."
            )
        return total // bucket_count

    if per_bucket is not None:
        if per_bucket <= 0:
            raise ValueError(
                f"--{split_name}-per-bucket must be positive, got {per_bucket}."
            )
        return per_bucket

    return default_per_bucket


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


def build_fixed_holdout_split(source_roots, seed: int, test_total=None, test_per_bucket=None):
    bucket_count = len(source_roots) * len(CLASS_NAMES)
    if bucket_count == 0:
        raise ValueError("At least one source root is required.")

    holdout_per_bucket = resolve_holdout_per_bucket(
        total=test_total,
        per_bucket=test_per_bucket,
        bucket_count=bucket_count,
        split_name="test",
        default_per_bucket=20,
    )

    rng = random.Random(seed)
    train_rows = []
    test_rows = []
    manifest = {
        "sources": [str(path) for path in source_roots],
        "seed": seed,
        "test_per_bucket": holdout_per_bucket,
        "test_total": holdout_per_bucket * bucket_count,
        "buckets": [],
    }

    for source_root in source_roots:
        source_name = source_root.name
        for class_name in CLASS_NAMES:
            files = collect_class_files(source_root, class_name)
            if len(files) <= holdout_per_bucket:
                raise ValueError(
                    f"{source_name}/{class_name} has only {len(files)} images, not enough "
                    f"to reserve {holdout_per_bucket} final test images."
                )

            shuffled = files[:]
            rng.shuffle(shuffled)
            test_files = shuffled[:holdout_per_bucket]
            train_files = shuffled[holdout_per_bucket:]

            manifest["buckets"].append(
                {
                    "source_name": source_name,
                    "class_name": class_name,
                    "total_count": len(files),
                    "train_count": len(train_files),
                    "test_count": len(test_files),
                    "train_files": [str(path) for path in train_files],
                    "test_files": [str(path) for path in test_files],
                }
            )

            label_id = CLASS_TO_ID[class_name]
            for split_name, split_files, target in (
                ("train", train_files, train_rows),
                ("test", test_files, test_rows),
            ):
                for image_path in split_files:
                    target.append(
                        {
                            "split": split_name,
                            "source_name": source_name,
                            "class_name": class_name,
                            "label_id": label_id,
                            "image_path": str(image_path),
                        }
                    )

    return manifest, train_rows, test_rows


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def collate_fn(batch):
    images, labels, paths = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(paths)


@torch.no_grad()
def extract_embeddings_for_rows(rows, processor, model, device: str, batch_size: int, num_workers: int, desc: str):
    if not rows:
        raise ValueError(f"No rows were provided for {desc}.")

    dataset = ImagePathDataset([(row["image_path"], row["label_id"]) for row in rows])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    all_embeddings = []
    all_labels = []
    all_paths = []

    for images, labels, paths in tqdm(loader, desc=desc):
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        image_features = get_normalized_image_features(model, inputs)
        all_embeddings.append(image_features.detach().cpu().numpy())
        all_labels.append(labels.numpy())
        all_paths.extend(paths)

    X = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0).astype(np.int64)
    paths_arr = np.asarray(all_paths)
    return X, y, paths_arr


def save_embeddings(path: Path, X, y, paths):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, paths=np.asarray(paths))


def load_embeddings(path: Path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["paths"]


def align_and_concatenate_embeddings(
    global_X,
    global_y,
    global_paths,
    crop_X,
    crop_y,
    crop_paths,
    l2_normalize_concat: bool = False,
):
    global_rows = {
        str(path): (global_X[idx], int(global_y[idx]))
        for idx, path in enumerate(global_paths.tolist())
    }
    crop_rows = {
        str(path): (crop_X[idx], int(crop_y[idx]))
        for idx, path in enumerate(crop_paths.tolist())
    }

    shared_paths = sorted(set(global_rows) & set(crop_rows))
    if not shared_paths:
        raise ValueError("No shared image paths found between global and crop embeddings.")

    fused_X = []
    fused_y = []
    for path in shared_paths:
        global_features, global_label = global_rows[path]
        crop_features, crop_label = crop_rows[path]
        if global_label != crop_label:
            raise ValueError(
                f"Label mismatch for {path}: global={global_label}, crop={crop_label}"
            )
        fused_X.append(np.concatenate([global_features, crop_features], axis=0))
        fused_y.append(global_label)

    fused_X = np.stack(fused_X, axis=0).astype(np.float32)
    if l2_normalize_concat:
        norms = np.linalg.norm(fused_X, axis=1, keepdims=True)
        fused_X = fused_X / np.maximum(norms, 1e-12)
    fused_y = np.asarray(fused_y, dtype=np.int64)
    fused_paths = np.asarray(shared_paths)
    return fused_X, fused_y, fused_paths


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


def bbox_iou(a_bbox, b_bbox):
    inter_w = interval_overlap(a_bbox[0], a_bbox[2], b_bbox[0], b_bbox[2])
    inter_h = interval_overlap(a_bbox[1], a_bbox[3], b_bbox[1], b_bbox[3])
    inter = inter_w * inter_h
    union = bbox_area(a_bbox) + bbox_area(b_bbox) - inter
    if union <= 0:
        return 0.0
    return inter / union


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


def parse_prompt_labels(raw_labels: str):
    labels = [part.strip().lower() for part in raw_labels.split(",") if part.strip()]
    return labels or list(DEFAULT_PROMPT_LABELS)


def prompt_labels_for_source(
    source_name: str,
    fallback_prompt_labels,
    chat_prompt_labels=None,
    social_prompt_labels=None,
):
    normalized_source = str(source_name or "").strip().lower()
    if "chat" in normalized_source:
        return list(chat_prompt_labels or CHAT_PROMPT_LABELS)
    if "social" in normalized_source:
        return list(social_prompt_labels or SOCIAL_PROMPT_LABELS)
    return list(fallback_prompt_labels or DEFAULT_PROMPT_LABELS)


def build_grounding_prompt(prompt_labels):
    return " . ".join(prompt_labels) + " ."


def _load_component_local_first(component_cls, model_name: str, component_label: str):
    try:
        return component_cls.from_pretrained(model_name, local_files_only=True)
    except Exception:
        try:
            return component_cls.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load the GroundingDINO {component_label} for {model_name}. "
                "The files are not fully cached locally, and downloading from Hugging Face failed."
            ) from exc


def load_grounding_dino_processor_and_model(model_name: str, device: str):
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

    processor = _load_component_local_first(GroundingDinoProcessor, model_name, "processor")
    model = _load_component_local_first(GroundingDinoForObjectDetection, model_name, "model").to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def detect_grounding_dino_regions(
    image: Image.Image,
    processor,
    model,
    device: str,
    prompt_labels,
    box_threshold: float,
    text_threshold: float,
):
    prompt = build_grounding_prompt(prompt_labels)
    encoded = processor(images=image, text=prompt, return_tensors="pt")
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    outputs = model(**model_inputs)
    target_sizes = [image.size[::-1]]
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=model_inputs.get("input_ids"),
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )[0]

    detections = []
    for score, box, label in zip(
        results.get("scores", []),
        results.get("boxes", []),
        results.get("text_labels", []),
    ):
        bbox = [int(round(coord)) for coord in box.tolist()]
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 0)
        height = max(y2 - y1, 0)
        if width == 0 or height == 0:
            continue
        detections.append(
            {
                "bbox": bbox,
                "text": str(label or "").strip(),
                "confidence": float(score),
                "source": "grounding_dino",
                "width": width,
                "height": height,
                "area": width * height,
            }
        )

    return detections


def build_grounding_dino_candidates(
    detections,
    image_width: int,
    image_height: int,
    padding_ratio: float,
):
    candidates = []
    image_area = max(image_width * image_height, 1)
    for detection in detections:
        label = detection["text"] or "detected_region"
        area_ratio = detection["area"] / image_area
        padded_bbox = clamp_and_pad_bbox(
            detection["bbox"],
            image_width=image_width,
            image_height=image_height,
            padding_ratio=padding_ratio,
        )
        candidates.append(
            {
                "candidate_type": "grounding_dino",
                "bbox": padded_bbox,
                "texts": [label],
                "ocr_confidence_mean": detection["confidence"],
                "ocr_confidence_max": detection["confidence"],
                "ocr_region_count": 1,
                "score_hint": float(detection["confidence"] + area_ratio * 2.0),
                "sources": [detection["source"]],
            }
        )

    candidates = dedupe_candidates(candidates, iou_threshold=0.9)
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
    return candidates


@torch.no_grad()
def embed_pil_images(images, processor, model, device: str, batch_size: int):
    if not images:
        raise ValueError("At least one crop image is required for embedding.")

    features = []
    for start_idx in range(0, len(images), batch_size):
        batch_images = images[start_idx:start_idx + batch_size]
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        image_features = get_normalized_image_features(model, inputs)
        features.append(image_features.detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def pool_crop_features(crop_features, crop_weights, pooling: str):
    if pooling not in {"avg", "max"}:
        raise ValueError(f"Unsupported crop pooling mode: {pooling}")

    if pooling == "max":
        pooled_feature = np.max(crop_features, axis=0)
    else:
        weights = np.asarray(crop_weights, dtype=np.float32)
        pooled_feature = np.average(crop_features, axis=0, weights=weights)

    pooled_feature = pooled_feature / max(np.linalg.norm(pooled_feature), 1e-12)
    return pooled_feature.astype(np.float32)


def _normalize_label(text: str):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _signal_weights_for_source(source_name: str):
    normalized_source = str(source_name or "").strip().lower()
    if "chat" in normalized_source:
        return CHAT_SIGNAL_LABEL_WEIGHTS
    if "social" in normalized_source:
        return SOCIAL_SIGNAL_LABEL_WEIGHTS
    return DEFAULT_SIGNAL_LABEL_WEIGHTS


def resolve_grounding_signal_profile(texts, source_name: str = ""):
    normalized_texts = [_normalize_label(text) for text in texts if _normalize_label(text)]
    best_label = "detected_region"
    best_weight = 1.0
    best_reason = "A detected UI region may contain the phishing prompt or CTA."
    signal_weights = _signal_weights_for_source(source_name)

    for text in normalized_texts:
        for known_label, weight in signal_weights.items():
            if text == known_label or text in known_label or known_label in text:
                if weight > best_weight:
                    best_label = known_label
                    best_weight = weight
                    best_reason = GROUNDING_SIGNAL_LABEL_REASONS[known_label]
        if best_label == "detected_region" and text:
            best_label = text

    return best_label, float(best_weight), best_reason


def rank_grounding_candidates(candidates, source_name: str, image_width: int, image_height: int):
    image_area = max(image_width * image_height, 1)

    def sort_key(candidate):
        _, label_weight, _ = resolve_grounding_signal_profile(
            candidate.get("texts", []),
            source_name=source_name,
        )
        area_ratio = bbox_area(candidate["bbox"]) / image_area
        return (
            label_weight + float(candidate.get("score_hint", 0.0)) + area_ratio,
            label_weight,
            float(candidate.get("score_hint", 0.0)),
            bbox_area(candidate["bbox"]),
        )

    return sorted(candidates, key=sort_key, reverse=True)


def build_grounding_dino_embeddings(
    rows,
    detector_processor,
    detector_model,
    siglip_processor,
    siglip_model,
    device: str,
    batch_size: int,
    prompt_labels,
    chat_prompt_labels,
    social_prompt_labels,
    box_threshold: float,
    text_threshold: float,
    padding_ratio: float,
    max_crops_per_image: int,
    crop_output_root: Path,
    pooling: str = "avg",
    log_every: int = 0,
    logger=None,
):
    embeddings = []
    labels = []
    image_paths = []
    per_image_rows = []
    per_crop_rows = []
    total = len(rows)
    last_logged = 0

    for idx, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        image_slug = safe_slug(image_path)
        image_output_dir = (
            crop_output_root
            / row["split"]
            / row["source_name"]
            / row["class_name"]
            / image_slug
        )
        image_output_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            image_width, image_height = image.size
            active_prompt_labels = prompt_labels_for_source(
                row["source_name"],
                fallback_prompt_labels=prompt_labels,
                chat_prompt_labels=chat_prompt_labels,
                social_prompt_labels=social_prompt_labels,
            )

            try:
                detections = detect_grounding_dino_regions(
                    image=image,
                    processor=detector_processor,
                    model=detector_model,
                    device=device,
                    prompt_labels=active_prompt_labels,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                candidates = build_grounding_dino_candidates(
                    detections,
                    image_width=image_width,
                    image_height=image_height,
                    padding_ratio=padding_ratio,
                )
            except Exception:
                detections = []
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

            candidates = rank_grounding_candidates(
                candidates,
                source_name=row["source_name"],
                image_width=image_width,
                image_height=image_height,
            )
            if max_crops_per_image > 0:
                candidates = candidates[:max_crops_per_image]

            crop_images = []
            crop_weights = []
            signal_scores = []
            candidate_metadata = []

            try:
                for crop_idx, candidate in enumerate(candidates, start=1):
                    bbox = tuple(candidate["bbox"])
                    crop_image = image.crop(bbox).copy()
                    crop_images.append(crop_image)

                    crop_path = image_output_dir / f"crop_{crop_idx:03d}_{candidate['candidate_type']}.png"
                    crop_image.save(crop_path)

                    signal_label, label_weight, signal_reason = resolve_grounding_signal_profile(
                        candidate.get("texts", []),
                        source_name=row["source_name"],
                    )
                    area_ratio = bbox_area(candidate["bbox"]) / max(image_width * image_height, 1)
                    detector_confidence = float(candidate.get("ocr_confidence_mean", 0.0))
                    signal_score = label_weight + detector_confidence + area_ratio * 2.0
                    pooling_weight = max(
                        float(candidate.get("score_hint", 0.0))
                        + label_weight
                        + detector_confidence
                        + area_ratio,
                        0.05,
                    )

                    crop_weights.append(pooling_weight)
                    signal_scores.append(signal_score)
                    candidate_copy = dict(candidate)
                    candidate_copy.update(
                        {
                            "crop_index": crop_idx,
                            "crop_path": str(crop_path),
                            "signal_label": signal_label,
                            "signal_reason": signal_reason,
                            "signal_score": float(signal_score),
                            "pooling_weight": float(pooling_weight),
                            "prompt_labels_used": list(active_prompt_labels),
                        }
                    )
                    candidate_metadata.append(candidate_copy)
            finally:
                pass

            crop_features = embed_pil_images(
                crop_images,
                processor=siglip_processor,
                model=siglip_model,
                device=device,
                batch_size=batch_size,
            )
            pooled_embedding = pool_crop_features(
                crop_features,
                crop_weights=crop_weights,
                pooling=pooling,
            )
            selected_signal_index = int(np.argmax(np.asarray(signal_scores, dtype=np.float32)))

            metadata_path = image_output_dir / "metadata.json"
            save_json(
                metadata_path,
                {
                    "split": row["split"],
                    "source_name": row["source_name"],
                    "class_name": row["class_name"],
                    "true_label_id": row["label_id"],
                    "image_path": str(image_path),
                    "image_width": image_width,
                    "image_height": image_height,
                    "prompt_labels": list(active_prompt_labels),
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                    "padding_ratio": padding_ratio,
                    "pooling": pooling,
                    "detections": detections,
                    "candidates": candidate_metadata,
                    "selected_signal_index": selected_signal_index + 1,
                },
            )

            selected_candidate = candidate_metadata[selected_signal_index]
            per_image_rows.append(
                {
                    "split": row["split"],
                    "source_name": row["source_name"],
                    "class_name": row["class_name"],
                    "label_id": row["label_id"],
                    "image_path": str(image_path),
                    "image_slug": image_slug,
                    "crop_count": len(candidate_metadata),
                    "selected_signal_index": selected_signal_index + 1,
                    "selected_signal_label": selected_candidate["signal_label"],
                    "selected_signal_reason": selected_candidate["signal_reason"],
                    "selected_signal_score": selected_candidate["signal_score"],
                    "selected_signal_bbox_x1": selected_candidate["bbox"][0],
                    "selected_signal_bbox_y1": selected_candidate["bbox"][1],
                    "selected_signal_bbox_x2": selected_candidate["bbox"][2],
                    "selected_signal_bbox_y2": selected_candidate["bbox"][3],
                    "selected_signal_crop_path": selected_candidate["crop_path"],
                    "metadata_path": str(metadata_path),
                    "candidate_types": "|".join(
                        candidate["candidate_type"] for candidate in candidate_metadata
                    ),
                }
            )

            for candidate in candidate_metadata:
                per_crop_rows.append(
                    {
                        "split": row["split"],
                        "source_name": row["source_name"],
                        "class_name": row["class_name"],
                        "label_id": row["label_id"],
                        "image_path": str(image_path),
                        "image_slug": image_slug,
                        "crop_index": candidate["crop_index"],
                        "candidate_type": candidate["candidate_type"],
                        "bbox_x1": candidate["bbox"][0],
                        "bbox_y1": candidate["bbox"][1],
                        "bbox_x2": candidate["bbox"][2],
                        "bbox_y2": candidate["bbox"][3],
                        "detector_confidence": candidate.get("ocr_confidence_mean", 0.0),
                        "score_hint": candidate.get("score_hint", 0.0),
                        "pooling_weight": candidate["pooling_weight"],
                        "signal_label": candidate["signal_label"],
                        "signal_reason": candidate["signal_reason"],
                        "signal_score": candidate["signal_score"],
                        "prompt_labels_used": " | ".join(candidate.get("prompt_labels_used", [])),
                        "texts": " | ".join(candidate.get("texts", [])),
                        "crop_path": candidate["crop_path"],
                    }
                )

            for crop_image in crop_images:
                crop_image.close()

        embeddings.append(pooled_embedding)
        labels.append(int(row["label_id"]))
        image_paths.append(str(image_path))

        if logger and log_every > 0 and (idx - last_logged >= log_every or idx == total):
            logger(f"grounding_dino {row['split']} embeddings: processed {idx}/{total} images")
            last_logged = idx

    X = np.stack(embeddings, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    paths = np.asarray(image_paths)
    return X, y, paths, per_image_rows, per_crop_rows
