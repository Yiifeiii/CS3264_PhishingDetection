import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from poc_utils import (
    OCRService,
    build_balanced_split,
    build_comparison,
    build_crop_feature_matrix,
    build_grounding_dino_feature_matrix,
    evaluate_classifier,
    format_metrics_markdown,
    load_grounding_dino,
    load_siglip,
    parse_grounding_prompt_labels,
    save_csv,
    save_json,
    save_npz,
    save_text,
    split_rows,
    split_summary,
    summarize_crop_rows,
    train_logistic_regression,
    embed_image_paths,
)


def log_message(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_metric_summary(metrics):
    parts = []
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return ", ".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a self-contained proof of concept under proof-of-concept/: "
            "global SigLIP+LR, OCR-crop SigLIP+LR, or GroundingDINO-crop SigLIP+LR "
            "on a weighted sample of benign/phish screenshots."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("global", "ocr", "grounding_dino", "compare_ocr", "compare_grounding_dino", "all"),
        default="compare_ocr",
        help=(
            "Which experiment to run. "
            "`global`, `ocr`, and `grounding_dino` run a single method. "
            "`compare_ocr` and `compare_grounding_dino` run global plus one segmentation method."
        ),
    )
    parser.add_argument("--real-root", default="data/benign_sample_30k")
    parser.add_argument("--fake-root", default="data/phish_sample_30k")
    parser.add_argument("--output-root", default="proof-of-concept/outputs/sample_1k_300_seed42")
    parser.add_argument("--model-name", default="artifacts/siglip2-base-patch16-224")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=150)
    parser.add_argument("--priority-fraction", type=float, default=0.65)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ocr-languages", default="en")
    parser.add_argument("--ocr-min-confidence", type=float, default=0.0)
    parser.add_argument("--ocr-include-processed", action="store_true")
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    parser.add_argument("--candidate-mode", default="blocks_preferred")
    parser.add_argument("--max-crops-per-image", type=int, default=6)
    parser.add_argument("--crop-pooling", choices=("avg", "max"), default="avg")
    parser.add_argument("--grounding-dino-model-name", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument(
        "--grounding-dino-prompt-labels",
        default="logo,button,login form,sign in form,text field,password field,dialog box,pop-up,notification,qr code,barcode,card",
    )
    parser.add_argument("--grounding-dino-box-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()

    log_message(
        "starting proof-of-concept run "
        f"(mode={args.mode}, seed={args.seed}, train_per_class={args.train_per_class}, "
        f"test_per_class={args.test_per_class}, crop_pooling={args.crop_pooling})"
    )

    project_root = Path(__file__).resolve().parent.parent
    real_root = (project_root / args.real_root).resolve()
    fake_root = (project_root / args.fake_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    split_root = output_root / "split"
    global_root = output_root / "global_siglip_lr"
    ocr_root = output_root / "ocr_crop_siglip_lr"
    grounding_root = output_root / "grounding_dino_crop_siglip_lr"

    run_global = args.mode in {"global", "compare_ocr", "compare_grounding_dino", "all"}
    run_ocr = args.mode in {"ocr", "compare_ocr", "all"}
    run_grounding_dino = args.mode in {"grounding_dino", "compare_grounding_dino", "all"}

    manifest = build_balanced_split(
        real_root=real_root,
        fake_root=fake_root,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
        priority_fraction=args.priority_fraction,
    )
    summary_payload = split_summary(manifest)
    save_csv(split_root / "manifest.csv", manifest)
    save_json(split_root / "summary.json", summary_payload)
    log_message(
        "split ready: "
        f"train={summary_payload['splits']['train']['rows']} "
        f"(prioritized={summary_payload['splits']['train']['prioritized_rows']}), "
        f"test={summary_payload['splits']['test']['rows']} "
        f"(prioritized={summary_payload['splits']['test']['prioritized_rows']})"
    )
    log_message(f"split manifest: {split_root / 'manifest.csv'}")

    train_rows = split_rows(manifest, "train")
    test_rows = split_rows(manifest, "test")
    y_train = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    y_test = np.asarray([int(row["label_id"]) for row in test_rows], dtype=np.int64)

    log_message(f"loading SigLIP model from {args.model_name}")
    processor, model, resolved_device = load_siglip(args.model_name, args.device)
    log_message(f"SigLIP loaded on device={resolved_device}")

    train_paths = [row["image_path"] for row in train_rows]
    test_paths = [row["image_path"] for row in test_rows]
    global_metrics = None
    if run_global:
        log_message("running global baseline: full screenshot -> SigLIP -> LogisticRegression")
        X_train_global = embed_image_paths(
            train_paths,
            processor=processor,
            model=model,
            device=resolved_device,
            batch_size=args.batch_size,
            stage_name="global train embeddings",
            log_every=args.log_every,
            logger=log_message,
        )
        X_test_global = embed_image_paths(
            test_paths,
            processor=processor,
            model=model,
            device=resolved_device,
            batch_size=args.batch_size,
            stage_name="global test embeddings",
            log_every=args.log_every,
            logger=log_message,
        )

        log_message("fitting LogisticRegression for global baseline")
        global_clf = train_logistic_regression(X_train_global, y_train)
        global_metrics, global_predictions = evaluate_classifier(
            global_clf,
            X_test=X_test_global,
            y_test=y_test,
            rows=test_rows,
        )
        save_npz(global_root / "embeddings_train.npz", X_train_global, y_train, train_paths)
        save_npz(global_root / "embeddings_test.npz", X_test_global, y_test, test_paths)
        save_json(global_root / "metrics.json", global_metrics)
        save_csv(global_root / "test_predictions.csv", global_predictions)
        save_text(
            global_root / "report.md",
            format_metrics_markdown(
                "Global SigLIP + Logistic Regression",
                global_metrics,
                extra_lines=[
                    f"- device: `{resolved_device}`",
                    "- input: full screenshot `shot.png` only",
                    "- sampling: weighted random split with Singapore-context priority bias",
                ],
            ),
        )
        log_message(f"global metrics: {format_metric_summary(global_metrics)}")
        log_message(f"global artifacts: {global_root}")

    ocr_metrics = None
    ocr_summary = None
    if run_ocr:
        log_message("running OCR segmentation baseline: OCR crops -> SigLIP -> LogisticRegression")
        ocr_languages = [language.strip() for language in args.ocr_languages.split(",") if language.strip()]
        ocr = OCRService(languages=ocr_languages, gpu=(resolved_device == "cuda"))
        X_train_ocr, train_ocr_rows = build_crop_feature_matrix(
            train_rows,
            ocr=ocr,
            processor=processor,
            model=model,
            device=resolved_device,
            batch_size=args.batch_size,
            max_crops=args.max_crops_per_image,
            min_confidence=args.ocr_min_confidence,
            include_processed=args.ocr_include_processed,
            padding_ratio=args.padding_ratio,
            candidate_mode=args.candidate_mode,
            pooling=args.crop_pooling,
            stage_name="ocr train features",
            log_every=args.log_every,
            logger=log_message,
        )
        X_test_ocr, test_ocr_rows = build_crop_feature_matrix(
            test_rows,
            ocr=ocr,
            processor=processor,
            model=model,
            device=resolved_device,
            batch_size=args.batch_size,
            max_crops=args.max_crops_per_image,
            min_confidence=args.ocr_min_confidence,
            include_processed=args.ocr_include_processed,
            padding_ratio=args.padding_ratio,
            candidate_mode=args.candidate_mode,
            pooling=args.crop_pooling,
            stage_name="ocr test features",
            log_every=args.log_every,
            logger=log_message,
        )

        log_message("fitting LogisticRegression for OCR baseline")
        ocr_clf = train_logistic_regression(X_train_ocr, y_train)
        ocr_metrics, ocr_predictions = evaluate_classifier(
            ocr_clf,
            X_test=X_test_ocr,
            y_test=y_test,
            rows=test_ocr_rows,
        )
        ocr_summary = summarize_crop_rows(train_ocr_rows + test_ocr_rows)
        save_npz(ocr_root / "embeddings_train.npz", X_train_ocr, y_train, train_paths)
        save_npz(ocr_root / "embeddings_test.npz", X_test_ocr, y_test, test_paths)
        save_csv(ocr_root / "train_crop_stats.csv", train_ocr_rows)
        save_csv(ocr_root / "test_crop_stats.csv", test_ocr_rows)
        save_json(ocr_root / "metrics.json", ocr_metrics)
        save_json(ocr_root / "crop_summary.json", ocr_summary)
        save_csv(ocr_root / "test_predictions.csv", ocr_predictions)
        save_text(
            ocr_root / "report.md",
            format_metrics_markdown(
                "OCR Crop SigLIP + Logistic Regression",
                ocr_metrics,
                extra_lines=[
                    f"- device: `{resolved_device}`",
                    "- input: OCR-derived crops from `shot.png` only",
                    f"- candidate_mode: `{args.candidate_mode}`",
                    f"- max_crops_per_image: {args.max_crops_per_image}",
                    f"- crop_pooling: `{args.crop_pooling}`",
                    f"- ocr_include_processed: {args.ocr_include_processed}",
                    f"- mean_crop_count: {ocr_summary['mean_crop_count']:.2f}",
                ],
            ),
        )
        log_message(
            "ocr metrics: "
            f"{format_metric_summary(ocr_metrics)}, mean_crop_count={ocr_summary['mean_crop_count']:.2f}"
        )
        log_message(f"ocr artifacts: {ocr_root}")

    grounding_metrics = None
    grounding_summary = None
    if run_grounding_dino:
        log_message(
            "running GroundingDINO segmentation baseline: proposal crops -> SigLIP -> LogisticRegression"
        )
        prompt_labels = parse_grounding_prompt_labels(args.grounding_dino_prompt_labels)
        log_message(
            f"loading GroundingDINO detector {args.grounding_dino_model_name} "
            f"with prompts={prompt_labels} and crop_pooling={args.crop_pooling}"
        )
        detector_processor, detector_model, _ = load_grounding_dino(
            args.grounding_dino_model_name,
            resolved_device,
        )
        X_train_grounding, train_grounding_rows = build_grounding_dino_feature_matrix(
            train_rows,
            detector_processor=detector_processor,
            detector_model=detector_model,
            siglip_processor=processor,
            siglip_model=model,
            device=resolved_device,
            batch_size=args.batch_size,
            prompt_labels=prompt_labels,
            box_threshold=args.grounding_dino_box_threshold,
            text_threshold=args.grounding_dino_text_threshold,
            padding_ratio=args.padding_ratio,
            max_crops=args.max_crops_per_image,
            pooling=args.crop_pooling,
            signal_output_root=grounding_root / "signal_crops",
            stage_name="grounding_dino train features",
            log_every=args.log_every,
            logger=log_message,
        )
        X_test_grounding, test_grounding_rows = build_grounding_dino_feature_matrix(
            test_rows,
            detector_processor=detector_processor,
            detector_model=detector_model,
            siglip_processor=processor,
            siglip_model=model,
            device=resolved_device,
            batch_size=args.batch_size,
            prompt_labels=prompt_labels,
            box_threshold=args.grounding_dino_box_threshold,
            text_threshold=args.grounding_dino_text_threshold,
            padding_ratio=args.padding_ratio,
            max_crops=args.max_crops_per_image,
            pooling=args.crop_pooling,
            signal_output_root=grounding_root / "signal_crops",
            stage_name="grounding_dino test features",
            log_every=args.log_every,
            logger=log_message,
        )

        log_message("fitting LogisticRegression for GroundingDINO baseline")
        grounding_clf = train_logistic_regression(X_train_grounding, y_train)
        grounding_metrics, grounding_predictions = evaluate_classifier(
            grounding_clf,
            X_test=X_test_grounding,
            y_test=y_test,
            rows=test_grounding_rows,
        )
        grounding_summary = summarize_crop_rows(train_grounding_rows + test_grounding_rows)
        save_npz(grounding_root / "embeddings_train.npz", X_train_grounding, y_train, train_paths)
        save_npz(grounding_root / "embeddings_test.npz", X_test_grounding, y_test, test_paths)
        save_csv(grounding_root / "train_signal_regions.csv", train_grounding_rows)
        save_csv(grounding_root / "test_signal_regions.csv", test_grounding_rows)
        save_json(grounding_root / "metrics.json", grounding_metrics)
        save_json(grounding_root / "crop_summary.json", grounding_summary)
        save_csv(grounding_root / "test_predictions.csv", grounding_predictions)
        save_text(
            grounding_root / "report.md",
            format_metrics_markdown(
                "GroundingDINO Crop SigLIP + Logistic Regression",
                grounding_metrics,
                extra_lines=[
                    f"- device: `{resolved_device}`",
                    "- input: GroundingDINO proposal crops from `shot.png` only",
                    f"- detector: `{args.grounding_dino_model_name}`",
                    f"- prompt_labels: `{' | '.join(prompt_labels)}`",
                    f"- max_crops_per_image: {args.max_crops_per_image}",
                    f"- crop_pooling: `{args.crop_pooling}`",
                    f"- mean_crop_count: {grounding_summary['mean_crop_count']:.2f}",
                    "- selected signal region per image is saved in `train_signal_regions.csv` and `test_signal_regions.csv`.",
                ],
            ),
        )
        log_message(
            "grounding_dino metrics: "
            f"{format_metric_summary(grounding_metrics)}, "
            f"mean_crop_count={grounding_summary['mean_crop_count']:.2f}"
        )
        log_message(f"grounding_dino artifacts: {grounding_root}")

    if run_global and ocr_metrics is not None:
        comparison = build_comparison(
            global_metrics=global_metrics,
            crop_metrics=ocr_metrics,
            split_summary_payload=summary_payload,
            crop_summary=ocr_summary,
            device=resolved_device,
            segmentation_key="ocr_crop_siglip_lr",
        )
        save_json(output_root / "comparison_ocr.json", comparison)
        save_text(
            output_root / "comparison_ocr.md",
            "\n".join(
                [
                    "# Proof-of-Concept Comparison",
                    "",
                    "## Split",
                    f"- train rows: {summary_payload['splits']['train']['rows']}",
                    f"- test rows: {summary_payload['splits']['test']['rows']}",
                    f"- train prioritized rows: {summary_payload['splits']['train']['prioritized_rows']}",
                    f"- test prioritized rows: {summary_payload['splits']['test']['prioritized_rows']}",
                    "",
                    "## Results",
                    f"- global accuracy: {global_metrics['accuracy']:.4f}",
                    f"- ocr accuracy: {ocr_metrics['accuracy']:.4f}",
                    f"- accuracy delta: {comparison['delta_accuracy']:.4f}",
                    f"- global f1: {global_metrics['f1']:.4f}",
                    f"- ocr f1: {ocr_metrics['f1']:.4f}",
                    f"- f1 delta: {comparison['delta_f1']:.4f}",
                    f"- global roc_auc: {global_metrics.get('roc_auc', float('nan')):.4f}",
                    f"- ocr roc_auc: {ocr_metrics.get('roc_auc', float('nan')):.4f}",
                    f"- roc_auc delta: {comparison['delta_roc_auc']:.4f}",
                    "",
                ]
            ),
        )
        log_message(
            "ocr comparison: "
            f"delta_accuracy={comparison['delta_accuracy']:.4f}, "
            f"delta_f1={comparison['delta_f1']:.4f}, "
            f"delta_roc_auc={comparison['delta_roc_auc']:.4f}"
        )
        log_message(f"ocr comparison artifacts: {output_root / 'comparison_ocr.json'}")

    if run_global and grounding_metrics is not None:
        comparison = build_comparison(
            global_metrics=global_metrics,
            crop_metrics=grounding_metrics,
            split_summary_payload=summary_payload,
            crop_summary=grounding_summary,
            device=resolved_device,
            segmentation_key="grounding_dino_crop_siglip_lr",
        )
        save_json(output_root / "comparison_grounding_dino.json", comparison)
        save_text(
            output_root / "comparison_grounding_dino.md",
            "\n".join(
                [
                    "# Proof-of-Concept Comparison",
                    "",
                    "## Split",
                    f"- train rows: {summary_payload['splits']['train']['rows']}",
                    f"- test rows: {summary_payload['splits']['test']['rows']}",
                    f"- train prioritized rows: {summary_payload['splits']['train']['prioritized_rows']}",
                    f"- test prioritized rows: {summary_payload['splits']['test']['prioritized_rows']}",
                    "",
                    "## Results",
                    f"- global accuracy: {global_metrics['accuracy']:.4f}",
                    f"- grounding_dino accuracy: {grounding_metrics['accuracy']:.4f}",
                    f"- accuracy delta: {comparison['delta_accuracy']:.4f}",
                    f"- global f1: {global_metrics['f1']:.4f}",
                    f"- grounding_dino f1: {grounding_metrics['f1']:.4f}",
                    f"- f1 delta: {comparison['delta_f1']:.4f}",
                    f"- global roc_auc: {global_metrics.get('roc_auc', float('nan')):.4f}",
                    f"- grounding_dino roc_auc: {grounding_metrics.get('roc_auc', float('nan')):.4f}",
                    f"- roc_auc delta: {comparison['delta_roc_auc']:.4f}",
                    "",
                ]
            ),
        )
        log_message(
            "grounding_dino comparison: "
            f"delta_accuracy={comparison['delta_accuracy']:.4f}, "
            f"delta_f1={comparison['delta_f1']:.4f}, "
            f"delta_roc_auc={comparison['delta_roc_auc']:.4f}"
        )
        log_message(f"grounding_dino comparison artifacts: {output_root / 'comparison_grounding_dino.json'}")

    log_message("proof-of-concept run finished")


if __name__ == "__main__":
    main()
