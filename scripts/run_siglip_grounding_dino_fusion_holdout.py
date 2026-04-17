import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGLIP_ROOT = PROJECT_ROOT / "siglip"
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".matplotlib-cache").resolve()))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SIGLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(SIGLIP_ROOT))

from classifier_utils import (  # noqa: E402
    MODEL_CHOICES,
    build_classifier,
    format_report,
    predict_positive_scores,
)
from grounding_dino_fusion_utils import (  # noqa: E402
    CHAT_PROMPT_LABELS,
    CLASS_TO_ID,
    DEFAULT_PROMPT_LABELS,
    SOCIAL_PROMPT_LABELS,
    align_and_concatenate_embeddings,
    build_fixed_holdout_split,
    build_grounding_dino_embeddings,
    detect_device,
    extract_embeddings_for_rows,
    load_grounding_dino_processor_and_model,
    parse_prompt_labels,
    save_embeddings,
    save_json,
    write_csv,
)
from hf_utils import load_siglip_processor_and_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed-holdout SigLIP experiment with three streams: global screenshot, "
            "GroundingDINO crop pooling, and global+crop concatenation."
        )
    )
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        help="Labeled source dataset root. Repeat to include multiple sources such as data/chat and data/social.",
    )
    parser.add_argument(
        "--test-total",
        type=int,
        default=80,
        help="Total fixed final test images across all (source, class) buckets.",
    )
    parser.add_argument(
        "--test-per-bucket",
        type=int,
        default=None,
        help="Optional fixed final test images per (source, class) bucket. Overrides --test-total.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/siglip_grounding_dino_fusion_holdout",
        help="Directory where splits, crops, embeddings, models, and reports will be written.",
    )
    parser.add_argument(
        "--model-name",
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
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for SigLIP global and crop embedding extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for global-image embedding extraction.",
    )
    parser.add_argument(
        "--prompt-labels",
        default=",".join(DEFAULT_PROMPT_LABELS),
        help="Comma-separated fallback GroundingDINO prompt labels for sources without a dedicated profile.",
    )
    parser.add_argument(
        "--chat-prompt-labels",
        default=",".join(CHAT_PROMPT_LABELS),
        help="Comma-separated GroundingDINO prompt labels used for the chat source.",
    )
    parser.add_argument(
        "--social-prompt-labels",
        default=",".join(SOCIAL_PROMPT_LABELS),
        help="Comma-separated GroundingDINO prompt labels used for the social source.",
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
        help="Padding ratio applied around each detected crop proposal.",
    )
    parser.add_argument(
        "--max-crops-per-image",
        type=int,
        default=4,
        help="Maximum GroundingDINO crop proposals kept per image. Use 0 for no cap.",
    )
    parser.add_argument(
        "--crop-pooling",
        choices=("avg", "max"),
        default="avg",
        help="How multiple GroundingDINO crop embeddings are fused into one image-level embedding.",
    )
    parser.add_argument(
        "--l2-normalize-concat",
        action="store_true",
        help="L2-normalize the concatenated global+crop feature vectors before training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the fixed test split.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the output root before running.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Emit a progress log every N images for long-running stages. Use 0 to disable.",
    )
    return parser.parse_args()


def resolve_path(raw_path: str):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def log(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def rows_to_csv_rows(rows):
    return [
        {
            "split": row["split"],
            "source_name": row["source_name"],
            "class_name": row["class_name"],
            "label_id": row["label_id"],
            "image_path": row["image_path"],
        }
        for row in rows
    ]


def rows_by_image_path(rows):
    return {str(row["image_path"]): row for row in rows}


def save_predictions(path: Path, base_rows_by_path, paths, y_true, preds, probs):
    prediction_rows = []
    for image_path, true_label, pred_label, prob in zip(paths.tolist(), y_true.tolist(), preds.tolist(), probs.tolist()):
        base_row = dict(base_rows_by_path[str(image_path)])
        base_row.update(
            {
                "pred_label_id": int(pred_label),
                "fake_probability": float(prob),
                "correct": int(int(pred_label) == int(true_label)),
            }
        )
        prediction_rows.append(base_row)

    fieldnames = sorted({key for row in prediction_rows for key in row.keys()})
    write_csv(path, prediction_rows, fieldnames=fieldnames)


def train_and_evaluate_stream(
    stream_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    test_paths,
    output_root: Path,
    test_rows_by_path,
):
    stream_dir = output_root / stream_name
    models_dir = stream_dir / "models"
    reports_dir = stream_dir / "reports"
    predictions_dir = stream_dir / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for model_name in MODEL_CHOICES:
        log(f"training {stream_name} -> {model_name}")
        clf = build_classifier(model_name, y_train)
        clf.fit(X_train, y_train)

        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(clf, model_path)

        preds = clf.predict(X_test)
        probs = predict_positive_scores(clf, X_test)
        metrics, report_text = format_report(
            y_test,
            preds,
            probs=probs,
            include_confusion=True,
        )
        report_path = reports_dir / f"test_metrics_{model_name}.txt"
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(report_text)

        predictions_path = predictions_dir / f"test_predictions_{model_name}.csv"
        save_predictions(
            predictions_path,
            base_rows_by_path=test_rows_by_path,
            paths=test_paths,
            y_true=y_test,
            preds=preds,
            probs=probs,
        )

        log(
            f"{stream_name} {model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics.get('roc_auc', float('nan')):.4f}"
        )

        summary_rows.append(
            {
                "stream": stream_name,
                "model": model_name,
                "train_size_total": int(len(y_train)),
                "train_real_count": int(np.sum(y_train == CLASS_TO_ID["real"])),
                "train_fake_count": int(np.sum(y_train == CLASS_TO_ID["fake"])),
                "test_size_total": int(len(y_test)),
                "test_real_count": int(np.sum(y_test == CLASS_TO_ID["real"])),
                "test_fake_count": int(np.sum(y_test == CLASS_TO_ID["fake"])),
                "embedding_dim": int(X_train.shape[1]),
                "model_path": str(model_path),
                "report_path": str(report_path),
                "predictions_path": str(predictions_path),
                **metrics,
            }
        )

    return summary_rows


def build_final_report(summary_rows, output_path: Path, manifest, output_root: Path):
    stream_order = ["global_siglip", "grounding_dino_crop_siglip", "fusion_concat_siglip"]
    lines = [
        "# SigLIP GroundingDINO Fusion Holdout Report",
        "",
        "## Setup",
        f"- Sources: {', '.join(Path(path).name for path in manifest['sources'])}",
        f"- Fixed final test total: {manifest['test_total']}",
        f"- Fixed final test per bucket: {manifest['test_per_bucket']}",
        f"- Output root: `{output_root}`",
        "",
        "## Final Test Metrics",
        "| Stream | Model | Accuracy | Precision | Recall | F1 | ROC AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (stream_order.index(row["stream"]), MODEL_CHOICES.index(row["model"])),
    )
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["stream"],
                    row["model"],
                    f"{row['accuracy']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1']:.4f}",
                    f"{row.get('roc_auc', float('nan')):.4f}",
                ]
            )
            + " |"
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    output_root = resolve_path(args.output_root)
    source_roots = [resolve_path(raw_path) for raw_path in args.source_root]
    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for source_root in source_roots:
        if not source_root.exists():
            raise FileNotFoundError(f"Source root does not exist: {source_root}")

    device = detect_device(args.device)
    prompt_labels = parse_prompt_labels(args.prompt_labels)
    chat_prompt_labels = parse_prompt_labels(args.chat_prompt_labels)
    social_prompt_labels = parse_prompt_labels(args.social_prompt_labels)
    log(
        "starting fixed-holdout GroundingDINO fusion run: "
        f"sources={[root.name for root in source_roots]}, "
        f"test_total={args.test_total if args.test_per_bucket is None else 'custom per bucket'}, "
        f"device={device}"
    )
    log(
        f"prompt profile: chat={chat_prompt_labels}, social={social_prompt_labels}, "
        f"fallback={prompt_labels}"
    )

    manifest, train_rows, test_rows = build_fixed_holdout_split(
        source_roots=source_roots,
        seed=args.seed,
        test_total=args.test_total,
        test_per_bucket=args.test_per_bucket,
    )
    save_json(output_root / "manifest.json", manifest)
    write_csv(
        output_root / "split" / "train_samples.csv",
        rows_to_csv_rows(train_rows),
        fieldnames=["split", "source_name", "class_name", "label_id", "image_path"],
    )
    write_csv(
        output_root / "split" / "test_samples.csv",
        rows_to_csv_rows(test_rows),
        fieldnames=["split", "source_name", "class_name", "label_id", "image_path"],
    )
    log(
        f"saved split manifest with {len(train_rows)} train images and {len(test_rows)} test images "
        f"to {output_root / 'manifest.json'}"
    )

    log("loading SigLIP model")
    siglip_processor, siglip_model = load_siglip_processor_and_model(args.model_name, device)

    log("extracting global train embeddings")
    X_global_train, y_global_train, global_train_paths = extract_embeddings_for_rows(
        train_rows,
        processor=siglip_processor,
        model=siglip_model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        desc="Global train embeddings",
    )
    log("extracting global test embeddings")
    X_global_test, y_global_test, global_test_paths = extract_embeddings_for_rows(
        test_rows,
        processor=siglip_processor,
        model=siglip_model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        desc="Global test embeddings",
    )
    global_dir = output_root / "global_siglip"
    save_embeddings(global_dir / "embeddings" / "train_embeddings.npz", X_global_train, y_global_train, global_train_paths)
    save_embeddings(global_dir / "embeddings" / "test_embeddings.npz", X_global_test, y_global_test, global_test_paths)
    log(f"saved global embeddings under {global_dir / 'embeddings'}")

    log("loading GroundingDINO model")
    detector_processor, detector_model = load_grounding_dino_processor_and_model(
        args.detector_model_name,
        device=device,
    )

    crop_dir = output_root / "grounding_dino_crop_siglip"
    log("extracting GroundingDINO crop train embeddings")
    X_crop_train, y_crop_train, crop_train_paths, crop_train_image_rows, crop_train_rows = build_grounding_dino_embeddings(
        train_rows,
        detector_processor=detector_processor,
        detector_model=detector_model,
        siglip_processor=siglip_processor,
        siglip_model=siglip_model,
        device=device,
        batch_size=args.batch_size,
        prompt_labels=prompt_labels,
        chat_prompt_labels=chat_prompt_labels,
        social_prompt_labels=social_prompt_labels,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        padding_ratio=args.padding_ratio,
        max_crops_per_image=args.max_crops_per_image,
        crop_output_root=crop_dir / "crops",
        pooling=args.crop_pooling,
        log_every=args.log_every,
        logger=log,
    )
    log("extracting GroundingDINO crop test embeddings")
    X_crop_test, y_crop_test, crop_test_paths, crop_test_image_rows, crop_test_rows = build_grounding_dino_embeddings(
        test_rows,
        detector_processor=detector_processor,
        detector_model=detector_model,
        siglip_processor=siglip_processor,
        siglip_model=siglip_model,
        device=device,
        batch_size=args.batch_size,
        prompt_labels=prompt_labels,
        chat_prompt_labels=chat_prompt_labels,
        social_prompt_labels=social_prompt_labels,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        padding_ratio=args.padding_ratio,
        max_crops_per_image=args.max_crops_per_image,
        crop_output_root=crop_dir / "crops",
        pooling=args.crop_pooling,
        log_every=args.log_every,
        logger=log,
    )
    save_embeddings(crop_dir / "embeddings" / "train_embeddings.npz", X_crop_train, y_crop_train, crop_train_paths)
    save_embeddings(crop_dir / "embeddings" / "test_embeddings.npz", X_crop_test, y_crop_test, crop_test_paths)
    write_csv(
        crop_dir / "summary" / "train_image_metadata.csv",
        crop_train_image_rows,
        fieldnames=list(crop_train_image_rows[0].keys()),
    )
    write_csv(
        crop_dir / "summary" / "test_image_metadata.csv",
        crop_test_image_rows,
        fieldnames=list(crop_test_image_rows[0].keys()),
    )
    write_csv(
        crop_dir / "summary" / "train_crop_metadata.csv",
        crop_train_rows,
        fieldnames=list(crop_train_rows[0].keys()),
    )
    write_csv(
        crop_dir / "summary" / "test_crop_metadata.csv",
        crop_test_rows,
        fieldnames=list(crop_test_rows[0].keys()),
    )
    crop_run_config = {
        "detector_model_name": args.detector_model_name,
        "prompt_labels_fallback": prompt_labels,
        "chat_prompt_labels": chat_prompt_labels,
        "social_prompt_labels": social_prompt_labels,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "padding_ratio": args.padding_ratio,
        "max_crops_per_image": args.max_crops_per_image,
        "crop_pooling": args.crop_pooling,
    }
    save_json(crop_dir / "summary" / "run_config.json", crop_run_config)
    log(f"saved GroundingDINO crops, metadata, and embeddings under {crop_dir}")

    fusion_dir = output_root / "fusion_concat_siglip"
    X_fusion_train, y_fusion_train, fusion_train_paths = align_and_concatenate_embeddings(
        X_global_train,
        y_global_train,
        global_train_paths,
        X_crop_train,
        y_crop_train,
        crop_train_paths,
        l2_normalize_concat=args.l2_normalize_concat,
    )
    X_fusion_test, y_fusion_test, fusion_test_paths = align_and_concatenate_embeddings(
        X_global_test,
        y_global_test,
        global_test_paths,
        X_crop_test,
        y_crop_test,
        crop_test_paths,
        l2_normalize_concat=args.l2_normalize_concat,
    )
    save_embeddings(fusion_dir / "embeddings" / "train_embeddings.npz", X_fusion_train, y_fusion_train, fusion_train_paths)
    save_embeddings(fusion_dir / "embeddings" / "test_embeddings.npz", X_fusion_test, y_fusion_test, fusion_test_paths)
    save_json(
        fusion_dir / "summary.json",
        {
            "fusion_method": "concat",
            "l2_normalize_concat": bool(args.l2_normalize_concat),
            "train_size": int(len(y_fusion_train)),
            "test_size": int(len(y_fusion_test)),
            "embedding_dim": int(X_fusion_train.shape[1]),
        },
    )
    log(f"saved concatenated fusion embeddings under {fusion_dir / 'embeddings'}")

    global_test_rows_by_path = rows_by_image_path(test_rows)
    crop_test_rows_by_path = rows_by_image_path(crop_test_image_rows)
    fusion_test_rows_by_path = rows_by_image_path(crop_test_image_rows)

    final_rows = []
    final_rows.extend(
        train_and_evaluate_stream(
            stream_name="global_siglip",
            X_train=X_global_train,
            y_train=y_global_train,
            X_test=X_global_test,
            y_test=y_global_test,
            test_paths=global_test_paths,
            output_root=output_root,
            test_rows_by_path=global_test_rows_by_path,
        )
    )
    final_rows.extend(
        train_and_evaluate_stream(
            stream_name="grounding_dino_crop_siglip",
            X_train=X_crop_train,
            y_train=y_crop_train,
            X_test=X_crop_test,
            y_test=y_crop_test,
            test_paths=crop_test_paths,
            output_root=output_root,
            test_rows_by_path=crop_test_rows_by_path,
        )
    )
    final_rows.extend(
        train_and_evaluate_stream(
            stream_name="fusion_concat_siglip",
            X_train=X_fusion_train,
            y_train=y_fusion_train,
            X_test=X_fusion_test,
            y_test=y_fusion_test,
            test_paths=fusion_test_paths,
            output_root=output_root,
            test_rows_by_path=fusion_test_rows_by_path,
        )
    )

    for row in final_rows:
        stream_root = output_root / row["stream"] / "embeddings"
        row["train_embeddings_path"] = str(stream_root / "train_embeddings.npz")
        row["test_embeddings_path"] = str(stream_root / "test_embeddings.npz")

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        summary_dir / "final_test_metrics.csv",
        final_rows,
        fieldnames=[
            "stream",
            "model",
            "train_size_total",
            "train_real_count",
            "train_fake_count",
            "test_size_total",
            "test_real_count",
            "test_fake_count",
            "embedding_dim",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "model_path",
            "report_path",
            "predictions_path",
            "train_embeddings_path",
            "test_embeddings_path",
        ],
    )
    save_json(summary_dir / "final_test_metrics.json", final_rows)
    build_final_report(final_rows, summary_dir / "final_report.md", manifest=manifest, output_root=output_root)

    best_row = max(final_rows, key=lambda row: (row["f1"], row.get("roc_auc", 0.0), row["accuracy"]))
    log(
        f"best final stream/model: {best_row['stream']} -> {best_row['model']} "
        f"(accuracy={best_row['accuracy']:.4f}, precision={best_row['precision']:.4f}, "
        f"recall={best_row['recall']:.4f}, f1={best_row['f1']:.4f}, roc_auc={best_row['roc_auc']:.4f})"
    )
    log(f"finished. final metrics: {summary_dir / 'final_test_metrics.csv'}")


if __name__ == "__main__":
    main()
