import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGLIP_ROOT = PROJECT_ROOT / "siglip"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SIGLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(SIGLIP_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from classifier_utils import compute_metrics  # noqa: E402
from run_siglip_ocr_crop_stream import (  # noqa: E402
    CLASS_NAMES,
    CLASS_TO_ID,
    SUPPORTED_EXTENSIONS,
    bbox_area,
    clamp_and_pad_bbox,
    collect_class_files,
    dedupe_candidates,
    detect_device,
    resolve_path,
    safe_slug,
    save_overview_image,
    score_candidates_for_entry,
    write_csv,
)
from hf_utils import load_siglip_processor_and_model  # noqa: E402
from utils.image_loading import load_image_rgb  # noqa: E402


DEFAULT_PROMPT_LABELS = (
    "logo",
    "button",
    "login form",
    "sign in form",
    "text field",
    "password field",
    "dialog box",
    "pop-up",
    "notification",
    "qr code",
    "barcode",
    "card",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a GroundingDINO-crop SigLIP inference stream over one or more labeled source roots. "
            "Detected proposal crops are scored with the same cropped-image SigLIP classifier."
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
        help="Run GroundingDINO-crop inference on a single image path.",
    )
    parser.add_argument(
        "--classifier-path",
        default="outputs/fixed_split_cv_crop/final_models/xgboost.joblib",
        help="Path to the trained cropped-image classifier checkpoint.",
    )
    parser.add_argument(
        "--siglip-model-name",
        default="artifacts/siglip2-base-patch16-224",
        help="SigLIP model name or local artifact path.",
    )
    parser.add_argument(
        "--detector-model-name",
        default="IDEA-Research/grounding-dino-tiny",
        help="GroundingDINO model name or local artifact path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device override: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--prompt-labels",
        default=",".join(DEFAULT_PROMPT_LABELS),
        help="Comma-separated GroundingDINO prompt labels.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Minimum GroundingDINO detection score for keeping a proposal.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="GroundingDINO text threshold for phrase extraction.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.05,
        help="Padding ratio applied around each detected proposal.",
    )
    parser.add_argument(
        "--max-crops-per-image",
        type=int,
        default=0,
        help="Optional cap on saved/scored proposal crops per image. Use 0 for no cap.",
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
        help="Save an overview image with all GroundingDINO boxes drawn.",
    )
    parser.add_argument(
        "--crop-output-root",
        default="outputs/grounding_dino_crop_stream/crops",
        help="Directory where GroundingDINO crop images and metadata will be written.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/grounding_dino_crop_stream/inference",
        help="Directory where inference CSVs and reports will be written.",
    )
    return parser.parse_args()


def parse_prompt_labels(raw_labels: str):
    labels = [part.strip().lower() for part in raw_labels.split(",") if part.strip()]
    return labels or list(DEFAULT_PROMPT_LABELS)


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
    for detection in detections:
        label = detection["text"] or "detected_region"
        area_ratio = detection["area"] / max(image_width * image_height, 1)
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


def write_report(path: Path, classifier_path: Path, detector_model_name: str, overall_metrics, per_source_rows, image_rows):
    total_images = len(image_rows)
    total_sources = len({row["source_name"] for row in image_rows})
    total_positive = int(sum(row["true_label_id"] == 1 for row in image_rows))
    predicted_positive = int(sum(row["pred_label_id"] == 1 for row in image_rows))

    lines = [
        "# GroundingDINO Crop Stream Report",
        "",
        "## Setup",
        f"- Classifier: `{classifier_path}`",
        f"- Detector: `{detector_model_name}`",
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
    classifier_path = resolve_path(args.classifier_path)
    crop_output_root = resolve_path(args.crop_output_root)
    output_root = resolve_path(args.output_root)
    device = detect_device() if args.device == "auto" else args.device
    prompt_labels = parse_prompt_labels(args.prompt_labels)

    active_inputs = sum(
        [
            1 if source_roots else 0,
            1 if single_image_path is not None else 0,
        ]
    )
    if active_inputs != 1:
        raise ValueError("Specify exactly one input mode: --source-root or --image.")

    for source_root in source_roots:
        if not source_root.exists():
            raise FileNotFoundError(f"Source root does not exist: {source_root}")
        for class_name in CLASS_NAMES:
            if not (source_root / class_name).exists():
                raise FileNotFoundError(
                    f"Expected class directory not found: {source_root / class_name}"
                )
    if single_image_path is not None and not single_image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {single_image_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")

    if args.clear_output:
        for root in (crop_output_root, output_root):
            if root.exists():
                shutil.rmtree(root)

    crop_output_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print("GroundingDINO crop stream plan:")
    if single_image_path is not None:
        print("  mode: single_image")
        print(f"  image: {single_image_path}")
    else:
        print("  mode: source_roots")
        print(f"  sources: {', '.join(root.name for root in source_roots)}")
    print(f"  detector model: {args.detector_model_name}")
    print(f"  prompt labels: {prompt_labels}")
    print(f"  classifier: {classifier_path}")
    print(f"  crop output root: {crop_output_root}")
    print(f"  inference output root: {output_root}")
    print(f"  max crops per image: {'unlimited' if args.max_crops_per_image <= 0 else args.max_crops_per_image}")

    detector_processor, detector_model = load_grounding_dino_processor_and_model(
        args.detector_model_name,
        device=device,
    )
    siglip_processor, siglip_model = load_siglip_processor_and_model(args.siglip_model_name, device)
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
    else:
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

    per_crop_rows = []
    per_image_rows = []

    for entry in tqdm(image_entries, desc="Running GroundingDINO crop stream"):
        image_path = entry["image_path"]
        image = load_image_rgb(image_path)
        image_width, image_height = image.size

        detections = detect_grounding_dino_regions(
            image=image,
            processor=detector_processor,
            model=detector_model,
            device=device,
            prompt_labels=prompt_labels,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        candidates = build_grounding_dino_candidates(
            detections,
            image_width=image_width,
            image_height=image_height,
            padding_ratio=args.padding_ratio,
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
            crop_path = image_output_dir / f"crop_{idx:03d}_{candidate['candidate_type']}.png"
            image.crop(tuple(candidate["bbox"])).save(crop_path)
            candidate["crop_path"] = str(crop_path)

        entry["image_slug"] = image_slug
        entry["metadata_path"] = image_output_dir / "metadata.json"

        crop_rows, image_row, selected_index, selected_candidate, scored_candidates = score_candidates_for_entry(
            entry,
            candidates,
            processor=siglip_processor,
            model=siglip_model,
            clf=clf,
            device=device,
            batch_size=args.batch_size,
        )
        per_crop_rows.extend(crop_rows)
        per_image_rows.append(image_row)

        if args.save_overview:
            overview_path = image_output_dir / "overview_boxes.png"
            save_overview_image(image, scored_candidates, overview_path, selected_index=selected_index)

        metadata = {
            "source_name": entry["source_name"],
            "class_name": entry["class_name"],
            "true_label_id": entry["label_id"],
            "image_path": str(entry["image_path"]),
            "image_width": image_width,
            "image_height": image_height,
            "detector_model_name": args.detector_model_name,
            "prompt_labels": prompt_labels,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "detections": detections,
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
        source_rows = [row for row in labeled_rows if row["source_name"] == source_name]
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
        detector_model_name=args.detector_model_name,
        overall_metrics=overall_metrics,
        per_source_rows=per_source_rows,
        image_rows=per_image_rows,
    )

    print("\nFinished GroundingDINO crop stream inference.")
    print(f"Per-crop predictions: {summary_dir / 'per_crop_predictions.csv'}")
    print(f"Per-image predictions: {summary_dir / 'per_image_predictions.csv'}")
    print(f"Per-source metrics: {summary_dir / 'per_source_metrics.csv'}")
    print(f"Overall metrics: {metrics_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
