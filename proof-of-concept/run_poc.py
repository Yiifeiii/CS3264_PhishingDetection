import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch

from poc_utils import (
    OCRService,
    build_balanced_split,
    build_comparison,
    build_crop_feature_matrix,
    build_grounding_dino_feature_matrix,
    encode_attention_fusion,
    canonicalize_classifier_name,
    classifier_display_name,
    evaluate_classifier,
    format_metrics_markdown,
    l2_normalize_rows,
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
    train_attention_encoder,
    train_classifier,
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


def resolve_requested_classifiers(raw_values):
    values = raw_values or ["lr"]
    if "all" in values:
        values = ["lr", "lightgbm", "xgboost", "contrastive"]

    resolved = []
    seen = set()
    for value in values:
        canonical = canonicalize_classifier_name(value)
        classifier_key = "lr" if canonical == "logreg" else canonical
        if classifier_key not in seen:
            seen.add(classifier_key)
            resolved.append(classifier_key)
    return resolved


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a self-contained proof of concept under proof-of-concept/: "
            "global SigLIP+classifier, segmentation SigLIP+classifier, and "
            "concat-fusion/attention-fusion SigLIP+classifier "
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
            "`compare_ocr` and `compare_grounding_dino` run global, one segmentation method, "
            "their concat-fusion stream, and optionally their attention-fusion stream."
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
    parser.add_argument(
        "--fusion-l2-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize concatenated global+crop embeddings before training the fusion classifier.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--classifier",
        action="append",
        choices=("lr", "lightgbm", "xgboost", "contrastive", "all"),
        help=(
            "Classifier head to train on top of the embeddings. "
            "Repeat the flag to run multiple heads, or use `--classifier all` "
            "for lr, lightgbm, xgboost, and contrastive."
        ),
    )
    parser.add_argument(
        "--include-attention-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train an additional attention-based fusion stream when both global and crop embeddings are available.",
    )
    parser.add_argument("--attention-hidden-dim", type=int, default=512)
    parser.add_argument("--attention-num-heads", type=int, default=8)
    parser.add_argument("--attention-dropout", type=float, default=0.10)
    parser.add_argument("--attention-epochs", type=int, default=60)
    parser.add_argument("--attention-train-batch-size", type=int, default=64)
    parser.add_argument("--attention-lr", type=float, default=1e-3)
    parser.add_argument("--attention-weight-decay", type=float, default=1e-4)
    parser.add_argument("--attention-patience", type=int, default=12)
    parser.add_argument("--attention-val-fraction", type=float, default=0.18)
    parser.add_argument("--attention-device", default="auto")
    parser.add_argument("--contrastive-hidden-dim", type=int, default=256)
    parser.add_argument("--contrastive-projection-dim", type=int, default=128)
    parser.add_argument("--contrastive-epochs", type=int, default=60)
    parser.add_argument("--contrastive-batch-size", type=int, default=64)
    parser.add_argument("--contrastive-lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight-decay", type=float, default=1e-4)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--contrastive-device", default="auto")
    return parser.parse_args()


def build_concat_fusion_features(X_global, X_crop, l2_normalize: bool):
    X_fusion = np.concatenate([X_global, X_crop], axis=1).astype(np.float32)
    if l2_normalize:
        X_fusion = l2_normalize_rows(X_fusion)
    return X_fusion


def build_attention_weight_rows(rows, weights):
    output_rows = []
    for row, weight_pair in zip(rows, weights.tolist()):
        enriched = dict(row)
        enriched["global_attention_weight"] = float(weight_pair[0])
        enriched["crop_attention_weight"] = float(weight_pair[1])
        output_rows.append(enriched)
    return output_rows


def maybe_save_contrastive_artifacts(stream_root, clf, canonical, X_train, y_train, train_paths, X_test, y_test, test_paths):
    if canonical != "contrastive":
        return
    save_npz(stream_root / "projected_embeddings_train.npz", clf.transform(X_train), y_train, train_paths)
    save_npz(stream_root / "projected_embeddings_test.npz", clf.transform(X_test), y_test, test_paths)
    save_json(stream_root / "contrastive_training_history.json", getattr(clf, "training_history_", []))


def classifier_train_kwargs(args):
    return {
        "seed": args.seed,
        "contrastive_hidden_dim": args.contrastive_hidden_dim,
        "contrastive_projection_dim": args.contrastive_projection_dim,
        "contrastive_epochs": args.contrastive_epochs,
        "contrastive_batch_size": args.contrastive_batch_size,
        "contrastive_learning_rate": args.contrastive_lr,
        "contrastive_weight_decay": args.contrastive_weight_decay,
        "contrastive_temperature": args.contrastive_temperature,
        "contrastive_device": args.contrastive_device,
        "log_every": args.log_every,
        "logger": log_message,
    }


def main():
    args = parse_args()
    classifier_keys = resolve_requested_classifiers(args.classifier)

    log_message(
        "starting proof-of-concept run "
        f"(mode={args.mode}, seed={args.seed}, train_per_class={args.train_per_class}, "
        f"test_per_class={args.test_per_class}, crop_pooling={args.crop_pooling}, "
        f"classifiers={classifier_keys})"
    )

    project_root = Path(__file__).resolve().parent.parent
    real_root = (project_root / args.real_root).resolve()
    fake_root = (project_root / args.fake_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    split_root = output_root / "split"

    run_global = args.mode in {"global", "compare_ocr", "compare_grounding_dino", "all"}
    run_ocr = args.mode in {"ocr", "compare_ocr", "all"}
    run_grounding_dino = args.mode in {"grounding_dino", "compare_grounding_dino", "all"}
    if (run_ocr or run_grounding_dino) and not run_global:
        run_global = True
        log_message("enabling global stream automatically because fusion requires global embeddings")

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
    X_train_global = None
    X_test_global = None
    global_metrics_by_classifier = {}
    if run_global:
        log_message("running global baseline: full screenshot -> SigLIP -> classifier")
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

        for classifier_key in classifier_keys:
            global_root = output_root / f"global_siglip_{classifier_key}"
            clf, canonical = train_classifier(
                classifier_key,
                X_train_global,
                y_train,
                **classifier_train_kwargs(args),
            )
            display_name = classifier_display_name(canonical)
            log_message(f"fitting {display_name} for global baseline")
            global_metrics, global_predictions = evaluate_classifier(
                clf,
                X_test=X_test_global,
                y_test=y_test,
                rows=test_rows,
            )
            global_metrics_by_classifier[classifier_key] = global_metrics
            save_npz(global_root / "embeddings_train.npz", X_train_global, y_train, train_paths)
            save_npz(global_root / "embeddings_test.npz", X_test_global, y_test, test_paths)
            joblib.dump(clf, global_root / "model.joblib")
            maybe_save_contrastive_artifacts(
                global_root,
                clf,
                canonical,
                X_train_global,
                y_train,
                train_paths,
                X_test_global,
                y_test,
                test_paths,
            )
            save_json(global_root / "metrics.json", global_metrics)
            save_csv(global_root / "test_predictions.csv", global_predictions)
            save_text(
                global_root / "report.md",
                format_metrics_markdown(
                    f"Global SigLIP + {display_name}",
                    global_metrics,
                    extra_lines=[
                        f"- device: `{resolved_device}`",
                        "- input: full screenshot `shot.png` only",
                        "- sampling: weighted random split with Singapore-context priority bias",
                        f"- classifier: `{classifier_key}`",
                    ],
                ),
            )
            log_message(f"global {classifier_key} metrics: {format_metric_summary(global_metrics)}")
            log_message(f"global {classifier_key} artifacts: {global_root}")

    X_train_ocr = None
    X_test_ocr = None
    ocr_metrics_by_classifier = {}
    ocr_fusion_metrics_by_classifier = {}
    ocr_attention_metrics_by_classifier = {}
    ocr_summary = None
    if run_ocr:
        log_message("running OCR segmentation baseline: OCR crops -> SigLIP -> classifier")
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

        ocr_summary = summarize_crop_rows(train_ocr_rows + test_ocr_rows)
        for classifier_key in classifier_keys:
            ocr_root = output_root / f"ocr_crop_siglip_{classifier_key}"
            clf, canonical = train_classifier(
                classifier_key,
                X_train_ocr,
                y_train,
                **classifier_train_kwargs(args),
            )
            display_name = classifier_display_name(canonical)
            log_message(f"fitting {display_name} for OCR baseline")
            ocr_metrics, ocr_predictions = evaluate_classifier(
                clf,
                X_test=X_test_ocr,
                y_test=y_test,
                rows=test_ocr_rows,
            )
            ocr_metrics_by_classifier[classifier_key] = ocr_metrics
            save_npz(ocr_root / "embeddings_train.npz", X_train_ocr, y_train, train_paths)
            save_npz(ocr_root / "embeddings_test.npz", X_test_ocr, y_test, test_paths)
            save_csv(ocr_root / "train_crop_stats.csv", train_ocr_rows)
            save_csv(ocr_root / "test_crop_stats.csv", test_ocr_rows)
            joblib.dump(clf, ocr_root / "model.joblib")
            maybe_save_contrastive_artifacts(
                ocr_root,
                clf,
                canonical,
                X_train_ocr,
                y_train,
                train_paths,
                X_test_ocr,
                y_test,
                test_paths,
            )
            save_json(ocr_root / "metrics.json", ocr_metrics)
            save_json(ocr_root / "crop_summary.json", ocr_summary)
            save_csv(ocr_root / "test_predictions.csv", ocr_predictions)
            save_text(
                ocr_root / "report.md",
                format_metrics_markdown(
                    f"OCR Crop SigLIP + {display_name}",
                    ocr_metrics,
                    extra_lines=[
                        f"- device: `{resolved_device}`",
                        "- input: OCR-derived crops from `shot.png` only",
                        f"- candidate_mode: `{args.candidate_mode}`",
                        f"- max_crops_per_image: {args.max_crops_per_image}",
                        f"- crop_pooling: `{args.crop_pooling}`",
                        f"- ocr_include_processed: {args.ocr_include_processed}",
                        f"- mean_crop_count: {ocr_summary['mean_crop_count']:.2f}",
                        f"- classifier: `{classifier_key}`",
                    ],
                ),
            )
            log_message(
                f"ocr {classifier_key} metrics: "
                f"{format_metric_summary(ocr_metrics)}, mean_crop_count={ocr_summary['mean_crop_count']:.2f}"
            )
            log_message(f"ocr {classifier_key} artifacts: {ocr_root}")

        if X_train_global is not None and X_test_global is not None:
            log_message(
                "running OCR fusion baseline: concat(global embedding, ocr crop embedding) -> classifier"
            )
            X_train_fusion_ocr = build_concat_fusion_features(
                X_train_global,
                X_train_ocr,
                l2_normalize=args.fusion_l2_normalize,
            )
            X_test_fusion_ocr = build_concat_fusion_features(
                X_test_global,
                X_test_ocr,
                l2_normalize=args.fusion_l2_normalize,
            )
            for classifier_key in classifier_keys:
                fusion_root = output_root / f"fusion_ocr_concat_siglip_{classifier_key}"
                clf, canonical = train_classifier(
                    classifier_key,
                    X_train_fusion_ocr,
                    y_train,
                    **classifier_train_kwargs(args),
                )
                display_name = classifier_display_name(canonical)
                log_message(f"fitting {display_name} for OCR fusion baseline")
                fusion_metrics, fusion_predictions = evaluate_classifier(
                    clf,
                    X_test=X_test_fusion_ocr,
                    y_test=y_test,
                    rows=test_ocr_rows,
                )
                ocr_fusion_metrics_by_classifier[classifier_key] = fusion_metrics
                save_npz(fusion_root / "embeddings_train.npz", X_train_fusion_ocr, y_train, train_paths)
                save_npz(fusion_root / "embeddings_test.npz", X_test_fusion_ocr, y_test, test_paths)
                joblib.dump(clf, fusion_root / "model.joblib")
                maybe_save_contrastive_artifacts(
                    fusion_root,
                    clf,
                    canonical,
                    X_train_fusion_ocr,
                    y_train,
                    train_paths,
                    X_test_fusion_ocr,
                    y_test,
                    test_paths,
                )
                save_json(fusion_root / "metrics.json", fusion_metrics)
                save_csv(fusion_root / "test_predictions.csv", fusion_predictions)
                save_text(
                    fusion_root / "report.md",
                    format_metrics_markdown(
                        f"Global + OCR Concat Fusion SigLIP + {display_name}",
                        fusion_metrics,
                        extra_lines=[
                            f"- device: `{resolved_device}`",
                            "- input: full screenshot plus OCR-derived crop embeddings from `shot.png` only",
                            f"- max_crops_per_image: {args.max_crops_per_image}",
                            f"- crop_pooling: `{args.crop_pooling}`",
                            f"- fusion_l2_normalize: {args.fusion_l2_normalize}",
                            f"- fused_dim: {X_train_fusion_ocr.shape[1]}",
                            f"- classifier: `{classifier_key}`",
                        ],
                    ),
                )
                log_message(
                    f"ocr fusion {classifier_key} metrics: {format_metric_summary(fusion_metrics)}"
                )
                log_message(f"ocr fusion {classifier_key} artifacts: {fusion_root}")

        if args.include_attention_fusion and X_train_global is not None and X_test_global is not None:
            log_message(
                "running OCR attention fusion baseline: attention(global embedding, ocr crop embedding) -> classifier"
            )
            attention_shared_root = output_root / "fusion_ocr_attention_siglip_shared"
            attention_model, attention_history, attention_best, attention_device = train_attention_encoder(
                X_train_global,
                X_train_ocr,
                y_train,
                seed=args.seed,
                val_fraction=args.attention_val_fraction,
                hidden_dim=args.attention_hidden_dim,
                num_heads=args.attention_num_heads,
                dropout=args.attention_dropout,
                epochs=args.attention_epochs,
                batch_size=args.attention_train_batch_size,
                learning_rate=args.attention_lr,
                weight_decay=args.attention_weight_decay,
                patience=args.attention_patience,
                device=args.attention_device,
                log_every=args.log_every,
                logger=log_message,
            )
            X_train_attention_ocr, train_attention_weights = encode_attention_fusion(
                attention_model,
                X_train_global,
                X_train_ocr,
                device=attention_device,
                batch_size=args.attention_train_batch_size,
            )
            X_test_attention_ocr, test_attention_weights = encode_attention_fusion(
                attention_model,
                X_test_global,
                X_test_ocr,
                device=attention_device,
                batch_size=args.attention_train_batch_size,
            )
            train_attention_rows = build_attention_weight_rows(train_rows, train_attention_weights)
            test_attention_rows = build_attention_weight_rows(test_ocr_rows, test_attention_weights)
            save_csv(attention_shared_root / "attention_weights_train.csv", train_attention_rows)
            save_csv(attention_shared_root / "attention_weights_test.csv", test_attention_rows)
            save_npz(attention_shared_root / "embeddings_train.npz", X_train_attention_ocr, y_train, train_paths)
            save_npz(attention_shared_root / "embeddings_test.npz", X_test_attention_ocr, y_test, test_paths)
            torch.save(
                {
                    "state_dict": attention_model.state_dict(),
                    "config": {
                        "input_dim": int(X_train_global.shape[1]),
                        "hidden_dim": args.attention_hidden_dim,
                        "num_heads": args.attention_num_heads,
                        "dropout": args.attention_dropout,
                        "seed": args.seed,
                    },
                },
                attention_shared_root / "attention_encoder.pt",
            )
            save_json(attention_shared_root / "attention_training_history.json", attention_history)
            save_json(
                attention_shared_root / "attention_best_epoch.json",
                {"epoch": attention_best["epoch"], "metrics": attention_best["metrics"]},
            )

            for classifier_key in classifier_keys:
                fusion_root = output_root / f"fusion_ocr_attention_siglip_{classifier_key}"
                clf, canonical = train_classifier(
                    classifier_key,
                    X_train_attention_ocr,
                    y_train,
                    **classifier_train_kwargs(args),
                )
                display_name = classifier_display_name(canonical)
                log_message(f"fitting {display_name} for OCR attention fusion baseline")
                attention_metrics, attention_predictions = evaluate_classifier(
                    clf,
                    X_test=X_test_attention_ocr,
                    y_test=y_test,
                    rows=test_attention_rows,
                )
                ocr_attention_metrics_by_classifier[classifier_key] = attention_metrics
                save_npz(fusion_root / "embeddings_train.npz", X_train_attention_ocr, y_train, train_paths)
                save_npz(fusion_root / "embeddings_test.npz", X_test_attention_ocr, y_test, test_paths)
                joblib.dump(clf, fusion_root / "model.joblib")
                maybe_save_contrastive_artifacts(
                    fusion_root,
                    clf,
                    canonical,
                    X_train_attention_ocr,
                    y_train,
                    train_paths,
                    X_test_attention_ocr,
                    y_test,
                    test_paths,
                )
                save_json(fusion_root / "metrics.json", attention_metrics)
                save_csv(fusion_root / "test_predictions.csv", attention_predictions)
                save_text(
                    fusion_root / "report.md",
                    format_metrics_markdown(
                        f"Global + OCR Attention Fusion SigLIP + {display_name}",
                        attention_metrics,
                        extra_lines=[
                            f"- device: `{resolved_device}`",
                            "- input: full screenshot plus OCR-derived crop embeddings from `shot.png` only",
                            f"- attention_device: `{attention_device}`",
                            f"- attention_best_epoch: {attention_best['epoch']}",
                            f"- attention_hidden_dim: {args.attention_hidden_dim}",
                            f"- attention_num_heads: {args.attention_num_heads}",
                            f"- fused_dim: {X_train_attention_ocr.shape[1]}",
                            f"- classifier: `{classifier_key}`",
                        ],
                    ),
                )
                log_message(
                    f"ocr attention fusion {classifier_key} metrics: {format_metric_summary(attention_metrics)}"
                )
                log_message(f"ocr attention fusion {classifier_key} artifacts: {fusion_root}")

    X_train_grounding = None
    X_test_grounding = None
    grounding_metrics_by_classifier = {}
    grounding_fusion_metrics_by_classifier = {}
    grounding_attention_metrics_by_classifier = {}
    grounding_summary = None
    if run_grounding_dino:
        grounding_signal_root = output_root / "grounding_dino_crop_siglip_shared" / "signal_crops"
        log_message(
            "running GroundingDINO segmentation baseline: proposal crops -> SigLIP -> classifier"
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
            signal_output_root=grounding_signal_root,
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
            signal_output_root=grounding_signal_root,
            stage_name="grounding_dino test features",
            log_every=args.log_every,
            logger=log_message,
        )

        grounding_summary = summarize_crop_rows(train_grounding_rows + test_grounding_rows)
        log_message(f"grounding_dino shared signal crops: {grounding_signal_root}")
        for classifier_key in classifier_keys:
            grounding_root = output_root / f"grounding_dino_crop_siglip_{classifier_key}"
            clf, canonical = train_classifier(
                classifier_key,
                X_train_grounding,
                y_train,
                **classifier_train_kwargs(args),
            )
            display_name = classifier_display_name(canonical)
            log_message(f"fitting {display_name} for GroundingDINO baseline")
            grounding_metrics, grounding_predictions = evaluate_classifier(
                clf,
                X_test=X_test_grounding,
                y_test=y_test,
                rows=test_grounding_rows,
            )
            grounding_metrics_by_classifier[classifier_key] = grounding_metrics
            save_npz(grounding_root / "embeddings_train.npz", X_train_grounding, y_train, train_paths)
            save_npz(grounding_root / "embeddings_test.npz", X_test_grounding, y_test, test_paths)
            save_csv(grounding_root / "train_signal_regions.csv", train_grounding_rows)
            save_csv(grounding_root / "test_signal_regions.csv", test_grounding_rows)
            joblib.dump(clf, grounding_root / "model.joblib")
            maybe_save_contrastive_artifacts(
                grounding_root,
                clf,
                canonical,
                X_train_grounding,
                y_train,
                train_paths,
                X_test_grounding,
                y_test,
                test_paths,
            )
            save_json(grounding_root / "metrics.json", grounding_metrics)
            save_json(grounding_root / "crop_summary.json", grounding_summary)
            save_csv(grounding_root / "test_predictions.csv", grounding_predictions)
            save_text(
                grounding_root / "report.md",
                format_metrics_markdown(
                    f"GroundingDINO Crop SigLIP + {display_name}",
                    grounding_metrics,
                    extra_lines=[
                        f"- device: `{resolved_device}`",
                        "- input: GroundingDINO proposal crops from `shot.png` only",
                        f"- detector: `{args.grounding_dino_model_name}`",
                        f"- prompt_labels: `{' | '.join(prompt_labels)}`",
                        f"- max_crops_per_image: {args.max_crops_per_image}",
                        f"- crop_pooling: `{args.crop_pooling}`",
                        f"- mean_crop_count: {grounding_summary['mean_crop_count']:.2f}",
                        f"- classifier: `{classifier_key}`",
                        "- selected signal region per image is saved in `train_signal_regions.csv` and `test_signal_regions.csv`.",
                    ],
                ),
            )
            log_message(
                f"grounding_dino {classifier_key} metrics: "
                f"{format_metric_summary(grounding_metrics)}, "
                f"mean_crop_count={grounding_summary['mean_crop_count']:.2f}"
            )
            log_message(f"grounding_dino {classifier_key} artifacts: {grounding_root}")

        if X_train_global is not None and X_test_global is not None:
            log_message(
                "running GroundingDINO fusion baseline: concat(global embedding, grounding_dino crop embedding) -> classifier"
            )
            X_train_fusion_grounding = build_concat_fusion_features(
                X_train_global,
                X_train_grounding,
                l2_normalize=args.fusion_l2_normalize,
            )
            X_test_fusion_grounding = build_concat_fusion_features(
                X_test_global,
                X_test_grounding,
                l2_normalize=args.fusion_l2_normalize,
            )
            for classifier_key in classifier_keys:
                fusion_root = output_root / f"fusion_grounding_dino_concat_siglip_{classifier_key}"
                clf, canonical = train_classifier(
                    classifier_key,
                    X_train_fusion_grounding,
                    y_train,
                    **classifier_train_kwargs(args),
                )
                display_name = classifier_display_name(canonical)
                log_message(f"fitting {display_name} for GroundingDINO fusion baseline")
                fusion_metrics, fusion_predictions = evaluate_classifier(
                    clf,
                    X_test=X_test_fusion_grounding,
                    y_test=y_test,
                    rows=test_grounding_rows,
                )
                grounding_fusion_metrics_by_classifier[classifier_key] = fusion_metrics
                save_npz(
                    fusion_root / "embeddings_train.npz",
                    X_train_fusion_grounding,
                    y_train,
                    train_paths,
                )
                save_npz(
                    fusion_root / "embeddings_test.npz",
                    X_test_fusion_grounding,
                    y_test,
                    test_paths,
                )
                joblib.dump(clf, fusion_root / "model.joblib")
                maybe_save_contrastive_artifacts(
                    fusion_root,
                    clf,
                    canonical,
                    X_train_fusion_grounding,
                    y_train,
                    train_paths,
                    X_test_fusion_grounding,
                    y_test,
                    test_paths,
                )
                save_json(fusion_root / "metrics.json", fusion_metrics)
                save_csv(fusion_root / "test_predictions.csv", fusion_predictions)
                save_text(
                    fusion_root / "report.md",
                    format_metrics_markdown(
                        f"Global + GroundingDINO Concat Fusion SigLIP + {display_name}",
                        fusion_metrics,
                        extra_lines=[
                            f"- device: `{resolved_device}`",
                            "- input: full screenshot plus GroundingDINO crop embeddings from `shot.png` only",
                            f"- detector: `{args.grounding_dino_model_name}`",
                            f"- max_crops_per_image: {args.max_crops_per_image}",
                            f"- crop_pooling: `{args.crop_pooling}`",
                            f"- fusion_l2_normalize: {args.fusion_l2_normalize}",
                            f"- fused_dim: {X_train_fusion_grounding.shape[1]}",
                            f"- classifier: `{classifier_key}`",
                        ],
                    ),
                )
                log_message(
                    f"grounding_dino fusion {classifier_key} metrics: "
                    f"{format_metric_summary(fusion_metrics)}"
                )
                log_message(f"grounding_dino fusion {classifier_key} artifacts: {fusion_root}")

        if args.include_attention_fusion and X_train_global is not None and X_test_global is not None:
            log_message(
                "running GroundingDINO attention fusion baseline: attention(global embedding, grounding_dino crop embedding) -> classifier"
            )
            attention_shared_root = output_root / "fusion_grounding_dino_attention_siglip_shared"
            attention_model, attention_history, attention_best, attention_device = train_attention_encoder(
                X_train_global,
                X_train_grounding,
                y_train,
                seed=args.seed,
                val_fraction=args.attention_val_fraction,
                hidden_dim=args.attention_hidden_dim,
                num_heads=args.attention_num_heads,
                dropout=args.attention_dropout,
                epochs=args.attention_epochs,
                batch_size=args.attention_train_batch_size,
                learning_rate=args.attention_lr,
                weight_decay=args.attention_weight_decay,
                patience=args.attention_patience,
                device=args.attention_device,
                log_every=args.log_every,
                logger=log_message,
            )
            X_train_attention_grounding, train_attention_weights = encode_attention_fusion(
                attention_model,
                X_train_global,
                X_train_grounding,
                device=attention_device,
                batch_size=args.attention_train_batch_size,
            )
            X_test_attention_grounding, test_attention_weights = encode_attention_fusion(
                attention_model,
                X_test_global,
                X_test_grounding,
                device=attention_device,
                batch_size=args.attention_train_batch_size,
            )
            train_attention_rows = build_attention_weight_rows(train_grounding_rows, train_attention_weights)
            test_attention_rows = build_attention_weight_rows(test_grounding_rows, test_attention_weights)
            save_csv(attention_shared_root / "attention_weights_train.csv", train_attention_rows)
            save_csv(attention_shared_root / "attention_weights_test.csv", test_attention_rows)
            save_npz(
                attention_shared_root / "embeddings_train.npz",
                X_train_attention_grounding,
                y_train,
                train_paths,
            )
            save_npz(
                attention_shared_root / "embeddings_test.npz",
                X_test_attention_grounding,
                y_test,
                test_paths,
            )
            torch.save(
                {
                    "state_dict": attention_model.state_dict(),
                    "config": {
                        "input_dim": int(X_train_global.shape[1]),
                        "hidden_dim": args.attention_hidden_dim,
                        "num_heads": args.attention_num_heads,
                        "dropout": args.attention_dropout,
                        "seed": args.seed,
                    },
                },
                attention_shared_root / "attention_encoder.pt",
            )
            save_json(attention_shared_root / "attention_training_history.json", attention_history)
            save_json(
                attention_shared_root / "attention_best_epoch.json",
                {"epoch": attention_best["epoch"], "metrics": attention_best["metrics"]},
            )

            for classifier_key in classifier_keys:
                fusion_root = output_root / f"fusion_grounding_dino_attention_siglip_{classifier_key}"
                clf, canonical = train_classifier(
                    classifier_key,
                    X_train_attention_grounding,
                    y_train,
                    **classifier_train_kwargs(args),
                )
                display_name = classifier_display_name(canonical)
                log_message(f"fitting {display_name} for GroundingDINO attention fusion baseline")
                attention_metrics, attention_predictions = evaluate_classifier(
                    clf,
                    X_test=X_test_attention_grounding,
                    y_test=y_test,
                    rows=test_attention_rows,
                )
                grounding_attention_metrics_by_classifier[classifier_key] = attention_metrics
                save_npz(
                    fusion_root / "embeddings_train.npz",
                    X_train_attention_grounding,
                    y_train,
                    train_paths,
                )
                save_npz(
                    fusion_root / "embeddings_test.npz",
                    X_test_attention_grounding,
                    y_test,
                    test_paths,
                )
                joblib.dump(clf, fusion_root / "model.joblib")
                maybe_save_contrastive_artifacts(
                    fusion_root,
                    clf,
                    canonical,
                    X_train_attention_grounding,
                    y_train,
                    train_paths,
                    X_test_attention_grounding,
                    y_test,
                    test_paths,
                )
                save_json(fusion_root / "metrics.json", attention_metrics)
                save_csv(fusion_root / "test_predictions.csv", attention_predictions)
                save_text(
                    fusion_root / "report.md",
                    format_metrics_markdown(
                        f"Global + GroundingDINO Attention Fusion SigLIP + {display_name}",
                        attention_metrics,
                        extra_lines=[
                            f"- device: `{resolved_device}`",
                            "- input: full screenshot plus GroundingDINO crop embeddings from `shot.png` only",
                            f"- detector: `{args.grounding_dino_model_name}`",
                            f"- attention_device: `{attention_device}`",
                            f"- attention_best_epoch: {attention_best['epoch']}",
                            f"- attention_hidden_dim: {args.attention_hidden_dim}",
                            f"- attention_num_heads: {args.attention_num_heads}",
                            f"- fused_dim: {X_train_attention_grounding.shape[1]}",
                            f"- classifier: `{classifier_key}`",
                        ],
                    ),
                )
                log_message(
                    f"grounding_dino attention fusion {classifier_key} metrics: "
                    f"{format_metric_summary(attention_metrics)}"
                )
                log_message(f"grounding_dino attention fusion {classifier_key} artifacts: {fusion_root}")

    if run_global and ocr_metrics_by_classifier:
        for classifier_key in classifier_keys:
            if classifier_key not in global_metrics_by_classifier or classifier_key not in ocr_metrics_by_classifier:
                continue
            global_metrics = global_metrics_by_classifier[classifier_key]
            ocr_metrics = ocr_metrics_by_classifier[classifier_key]
            fusion_metrics = ocr_fusion_metrics_by_classifier.get(classifier_key)
            comparison = build_comparison(
                global_metrics=global_metrics,
                crop_metrics=ocr_metrics,
                split_summary_payload=summary_payload,
                crop_summary=ocr_summary,
                device=resolved_device,
                global_key=f"global_siglip_{classifier_key}",
                segmentation_key=f"ocr_crop_siglip_{classifier_key}",
                fusion_metrics=fusion_metrics,
                fusion_key=f"fusion_ocr_concat_siglip_{classifier_key}",
            )
            save_json(output_root / f"comparison_ocr_{classifier_key}.json", comparison)
            report_lines = [
                "# Proof-of-Concept Comparison",
                "",
                "## Classifier",
                f"- classifier: `{classifier_key}`",
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
                f"- ocr vs global accuracy delta: {comparison['delta_accuracy']:.4f}",
                f"- global f1: {global_metrics['f1']:.4f}",
                f"- ocr f1: {ocr_metrics['f1']:.4f}",
                f"- ocr vs global f1 delta: {comparison['delta_f1']:.4f}",
                f"- global roc_auc: {global_metrics.get('roc_auc', float('nan')):.4f}",
                f"- ocr roc_auc: {ocr_metrics.get('roc_auc', float('nan')):.4f}",
                f"- ocr vs global roc_auc delta: {comparison['delta_roc_auc']:.4f}",
            ]
            if fusion_metrics is not None:
                report_lines.extend(
                    [
                        f"- fusion accuracy: {fusion_metrics['accuracy']:.4f}",
                        f"- fusion vs global accuracy delta: {comparison['delta_fusion_vs_global_accuracy']:.4f}",
                        f"- fusion vs ocr accuracy delta: {comparison['delta_fusion_vs_crop_accuracy']:.4f}",
                        f"- fusion f1: {fusion_metrics['f1']:.4f}",
                        f"- fusion vs global f1 delta: {comparison['delta_fusion_vs_global_f1']:.4f}",
                        f"- fusion vs ocr f1 delta: {comparison['delta_fusion_vs_crop_f1']:.4f}",
                        f"- fusion roc_auc: {fusion_metrics.get('roc_auc', float('nan')):.4f}",
                        f"- fusion vs global roc_auc delta: {comparison['delta_fusion_vs_global_roc_auc']:.4f}",
                        f"- fusion vs ocr roc_auc delta: {comparison['delta_fusion_vs_crop_roc_auc']:.4f}",
                    ]
                )
            report_lines.append("")
            save_text(
                output_root / f"comparison_ocr_{classifier_key}.md",
                "\n".join(report_lines),
            )
            log_parts = [
                f"ocr_vs_global_accuracy={comparison['delta_accuracy']:.4f}",
                f"ocr_vs_global_f1={comparison['delta_f1']:.4f}",
                f"ocr_vs_global_roc_auc={comparison['delta_roc_auc']:.4f}",
            ]
            if fusion_metrics is not None:
                log_parts.extend(
                    [
                        f"fusion_vs_global_accuracy={comparison['delta_fusion_vs_global_accuracy']:.4f}",
                        f"fusion_vs_ocr_accuracy={comparison['delta_fusion_vs_crop_accuracy']:.4f}",
                    ]
                )
            log_message(f"ocr comparison {classifier_key}: " + ", ".join(log_parts))
            log_message(f"ocr comparison {classifier_key} artifacts: {output_root / f'comparison_ocr_{classifier_key}.json'}")

    if run_global and ocr_attention_metrics_by_classifier:
        for classifier_key in classifier_keys:
            if classifier_key not in global_metrics_by_classifier or classifier_key not in ocr_metrics_by_classifier:
                continue
            global_metrics = global_metrics_by_classifier[classifier_key]
            ocr_metrics = ocr_metrics_by_classifier[classifier_key]
            attention_metrics = ocr_attention_metrics_by_classifier.get(classifier_key)
            if attention_metrics is None:
                continue
            comparison = build_comparison(
                global_metrics=global_metrics,
                crop_metrics=ocr_metrics,
                split_summary_payload=summary_payload,
                crop_summary=ocr_summary,
                device=resolved_device,
                global_key=f"global_siglip_{classifier_key}",
                segmentation_key=f"ocr_crop_siglip_{classifier_key}",
                fusion_metrics=attention_metrics,
                fusion_key=f"fusion_ocr_attention_siglip_{classifier_key}",
            )
            save_json(output_root / f"comparison_ocr_attention_{classifier_key}.json", comparison)
            save_text(
                output_root / f"comparison_ocr_attention_{classifier_key}.md",
                "\n".join(
                    [
                        "# Proof-of-Concept Attention Fusion Comparison",
                        "",
                        "## Classifier",
                        f"- classifier: `{classifier_key}`",
                        "",
                        "## Results",
                        f"- global accuracy: {global_metrics['accuracy']:.4f}",
                        f"- ocr accuracy: {ocr_metrics['accuracy']:.4f}",
                        f"- attention fusion accuracy: {attention_metrics['accuracy']:.4f}",
                        f"- attention vs global accuracy delta: {comparison['delta_fusion_vs_global_accuracy']:.4f}",
                        f"- attention vs ocr accuracy delta: {comparison['delta_fusion_vs_crop_accuracy']:.4f}",
                        f"- global f1: {global_metrics['f1']:.4f}",
                        f"- ocr f1: {ocr_metrics['f1']:.4f}",
                        f"- attention fusion f1: {attention_metrics['f1']:.4f}",
                        f"- attention vs global f1 delta: {comparison['delta_fusion_vs_global_f1']:.4f}",
                        f"- attention vs ocr f1 delta: {comparison['delta_fusion_vs_crop_f1']:.4f}",
                        "",
                    ]
                ),
            )
            log_message(
                f"ocr attention comparison {classifier_key}: "
                f"attention_vs_global_accuracy={comparison['delta_fusion_vs_global_accuracy']:.4f}, "
                f"attention_vs_ocr_accuracy={comparison['delta_fusion_vs_crop_accuracy']:.4f}"
            )
            log_message(
                f"ocr attention comparison {classifier_key} artifacts: "
                f"{output_root / f'comparison_ocr_attention_{classifier_key}.json'}"
            )

    if run_global and grounding_metrics_by_classifier:
        for classifier_key in classifier_keys:
            if classifier_key not in global_metrics_by_classifier or classifier_key not in grounding_metrics_by_classifier:
                continue
            global_metrics = global_metrics_by_classifier[classifier_key]
            grounding_metrics = grounding_metrics_by_classifier[classifier_key]
            fusion_metrics = grounding_fusion_metrics_by_classifier.get(classifier_key)
            comparison = build_comparison(
                global_metrics=global_metrics,
                crop_metrics=grounding_metrics,
                split_summary_payload=summary_payload,
                crop_summary=grounding_summary,
                device=resolved_device,
                global_key=f"global_siglip_{classifier_key}",
                segmentation_key=f"grounding_dino_crop_siglip_{classifier_key}",
                fusion_metrics=fusion_metrics,
                fusion_key=f"fusion_grounding_dino_concat_siglip_{classifier_key}",
            )
            save_json(output_root / f"comparison_grounding_dino_{classifier_key}.json", comparison)
            report_lines = [
                "# Proof-of-Concept Comparison",
                "",
                "## Classifier",
                f"- classifier: `{classifier_key}`",
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
                f"- grounding_dino vs global accuracy delta: {comparison['delta_accuracy']:.4f}",
                f"- global f1: {global_metrics['f1']:.4f}",
                f"- grounding_dino f1: {grounding_metrics['f1']:.4f}",
                f"- grounding_dino vs global f1 delta: {comparison['delta_f1']:.4f}",
                f"- global roc_auc: {global_metrics.get('roc_auc', float('nan')):.4f}",
                f"- grounding_dino roc_auc: {grounding_metrics.get('roc_auc', float('nan')):.4f}",
                f"- grounding_dino vs global roc_auc delta: {comparison['delta_roc_auc']:.4f}",
            ]
            if fusion_metrics is not None:
                report_lines.extend(
                    [
                        f"- fusion accuracy: {fusion_metrics['accuracy']:.4f}",
                        f"- fusion vs global accuracy delta: {comparison['delta_fusion_vs_global_accuracy']:.4f}",
                        f"- fusion vs grounding_dino accuracy delta: {comparison['delta_fusion_vs_crop_accuracy']:.4f}",
                        f"- fusion f1: {fusion_metrics['f1']:.4f}",
                        f"- fusion vs global f1 delta: {comparison['delta_fusion_vs_global_f1']:.4f}",
                        f"- fusion vs grounding_dino f1 delta: {comparison['delta_fusion_vs_crop_f1']:.4f}",
                        f"- fusion roc_auc: {fusion_metrics.get('roc_auc', float('nan')):.4f}",
                        f"- fusion vs global roc_auc delta: {comparison['delta_fusion_vs_global_roc_auc']:.4f}",
                        f"- fusion vs grounding_dino roc_auc delta: {comparison['delta_fusion_vs_crop_roc_auc']:.4f}",
                    ]
                )
            report_lines.append("")
            save_text(
                output_root / f"comparison_grounding_dino_{classifier_key}.md",
                "\n".join(report_lines),
            )
            log_parts = [
                f"grounding_vs_global_accuracy={comparison['delta_accuracy']:.4f}",
                f"grounding_vs_global_f1={comparison['delta_f1']:.4f}",
                f"grounding_vs_global_roc_auc={comparison['delta_roc_auc']:.4f}",
            ]
            if fusion_metrics is not None:
                log_parts.extend(
                    [
                        f"fusion_vs_global_accuracy={comparison['delta_fusion_vs_global_accuracy']:.4f}",
                        f"fusion_vs_grounding_accuracy={comparison['delta_fusion_vs_crop_accuracy']:.4f}",
                    ]
                )
            log_message(f"grounding_dino comparison {classifier_key}: " + ", ".join(log_parts))
            log_message(
                f"grounding_dino comparison {classifier_key} artifacts: "
                f"{output_root / f'comparison_grounding_dino_{classifier_key}.json'}"
            )

    if run_global and grounding_attention_metrics_by_classifier:
        for classifier_key in classifier_keys:
            if classifier_key not in global_metrics_by_classifier or classifier_key not in grounding_metrics_by_classifier:
                continue
            global_metrics = global_metrics_by_classifier[classifier_key]
            grounding_metrics = grounding_metrics_by_classifier[classifier_key]
            attention_metrics = grounding_attention_metrics_by_classifier.get(classifier_key)
            if attention_metrics is None:
                continue
            comparison = build_comparison(
                global_metrics=global_metrics,
                crop_metrics=grounding_metrics,
                split_summary_payload=summary_payload,
                crop_summary=grounding_summary,
                device=resolved_device,
                global_key=f"global_siglip_{classifier_key}",
                segmentation_key=f"grounding_dino_crop_siglip_{classifier_key}",
                fusion_metrics=attention_metrics,
                fusion_key=f"fusion_grounding_dino_attention_siglip_{classifier_key}",
            )
            save_json(output_root / f"comparison_grounding_dino_attention_{classifier_key}.json", comparison)
            save_text(
                output_root / f"comparison_grounding_dino_attention_{classifier_key}.md",
                "\n".join(
                    [
                        "# Proof-of-Concept Attention Fusion Comparison",
                        "",
                        "## Classifier",
                        f"- classifier: `{classifier_key}`",
                        "",
                        "## Results",
                        f"- global accuracy: {global_metrics['accuracy']:.4f}",
                        f"- grounding_dino accuracy: {grounding_metrics['accuracy']:.4f}",
                        f"- attention fusion accuracy: {attention_metrics['accuracy']:.4f}",
                        f"- attention vs global accuracy delta: {comparison['delta_fusion_vs_global_accuracy']:.4f}",
                        f"- attention vs grounding_dino accuracy delta: {comparison['delta_fusion_vs_crop_accuracy']:.4f}",
                        f"- global f1: {global_metrics['f1']:.4f}",
                        f"- grounding_dino f1: {grounding_metrics['f1']:.4f}",
                        f"- attention fusion f1: {attention_metrics['f1']:.4f}",
                        f"- attention vs global f1 delta: {comparison['delta_fusion_vs_global_f1']:.4f}",
                        f"- attention vs grounding_dino f1 delta: {comparison['delta_fusion_vs_crop_f1']:.4f}",
                        "",
                    ]
                ),
            )
            log_message(
                f"grounding_dino attention comparison {classifier_key}: "
                f"attention_vs_global_accuracy={comparison['delta_fusion_vs_global_accuracy']:.4f}, "
                f"attention_vs_grounding_accuracy={comparison['delta_fusion_vs_crop_accuracy']:.4f}"
            )
            log_message(
                f"grounding_dino attention comparison {classifier_key} artifacts: "
                f"{output_root / f'comparison_grounding_dino_attention_{classifier_key}.json'}"
            )

    log_message("proof-of-concept run finished")


if __name__ == "__main__":
    main()
