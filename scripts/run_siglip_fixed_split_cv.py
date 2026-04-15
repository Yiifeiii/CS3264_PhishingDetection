import argparse
import csv
import json
import os
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGLIP_ROOT = PROJECT_ROOT / "siglip"
if str(SIGLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(SIGLIP_ROOT))

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_siglip_learning_curve_cv import (  # noqa: E402
    CLASS_NAMES,
    CLASS_TO_ID,
    METRIC_NAMES,
    MODEL_CHOICES,
    aggregate_rows,
    bucket_name,
    choose_best_row,
    choose_best_row_for_train_size,
    collect_class_files,
    compute_metrics,
    detect_device,
    embedding_cache_matches_expected,
    extract_embeddings_for_samples,
    find_plateau_row,
    format_metric,
    format_report,
    load_embeddings,
    load_siglip_processor_and_model,
    lookup_summary_row,
    parse_train_sizes,
    predict_positive_scores,
    resolve_path,
    save_embeddings,
    split_into_folds,
    train_and_score,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run SigLIP k-fold validation on an existing train split and evaluate once "
            "on an untouched fixed test split."
        )
    )
    parser.add_argument(
        "--train-root",
        required=True,
        help="Train split root containing {real,fake} subfolders.",
    )
    parser.add_argument(
        "--test-root",
        required=True,
        help="Untouched test split root containing {real,fake} subfolders.",
    )
    parser.add_argument(
        "--train-sizes",
        required=True,
        help=(
            "Comma-separated total training sizes. Each size must be divisible by 2 "
            "because the script samples equally from real and fake."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of validation folds inside the train split.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeated subset samplings per fold and train size.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/fixed_split_cv",
        help="Directory where embeddings, reports, and final models will be written.",
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
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for embedding extraction.",
    )
    parser.add_argument(
        "--final-train-mode",
        choices=("full_train", "best_size", "plateau_size"),
        default="full_train",
        help=(
            "How to train the final model before the untouched test evaluation. "
            "`full_train` is the practical default."
        ),
    )
    parser.add_argument(
        "--plateau-ratio",
        type=float,
        default=0.95,
        help="Validation plateau threshold used for the learning-curve summary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fold assignment and subset sampling.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the output root before running the experiment.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached embeddings even if they already exist.",
    )
    parser.add_argument(
        "--save-trial-models",
        action="store_true",
        help=(
            "Save every fold/repeat classifier checkpoint used for the CV and fixed-test "
            "curves under output-root/trial_models."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = resolve_path(args.output_root)
    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = detect_device() if args.device == "auto" else args.device
    train_root = resolve_path(args.train_root)
    test_root = resolve_path(args.test_root)
    train_sizes = parse_train_sizes(args.train_sizes)

    for split_root, label in ((train_root, "train"), (test_root, "test")):
        if not split_root.exists():
            raise FileNotFoundError(f"{label} root does not exist: {split_root}")
        for class_name in CLASS_NAMES:
            class_dir = split_root / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected class directory not found: {class_dir}"
                )

    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    bucket_count = len(CLASS_NAMES)
    per_bucket_train_sizes = []
    for total_size in train_sizes:
        if total_size % bucket_count != 0:
            raise ValueError(
                f"Training size {total_size} is not divisible by {bucket_count} classes."
            )
        per_bucket_train_sizes.append(total_size // bucket_count)

    bucket_entries = []
    for class_name in CLASS_NAMES:
        train_files = collect_class_files(train_root, class_name)
        test_files = collect_class_files(test_root, class_name)
        if not train_files:
            raise ValueError(f"No {class_name} samples found under {train_root}")
        if not test_files:
            raise ValueError(f"No {class_name} samples found under {test_root}")
        if len(train_files) < args.folds:
            raise ValueError(
                f"{train_root.name}/{class_name} has only {len(train_files)} files, "
                f"which is not enough for {args.folds} folds."
            )

        bucket_entries.append(
            {
                "source_name": train_root.name,
                "class_name": class_name,
                "bucket_name": bucket_name(train_root.name, class_name),
                "train_files": train_files,
                "test_files": test_files,
            }
        )

    import random

    fold_rng = random.Random(args.seed + 1_000)
    for bucket in bucket_entries:
        shuffled_train = bucket["train_files"][:]
        fold_rng.shuffle(shuffled_train)
        bucket["train_files"] = shuffled_train
        bucket["folds"] = split_into_folds(shuffled_train, args.folds)

    min_fold_train_pool = min(
        len(bucket["train_files"]) - len(bucket["folds"][fold_idx])
        for bucket in bucket_entries
        for fold_idx in range(args.folds)
    )
    max_requested_per_bucket = per_bucket_train_sizes[-1]
    if max_requested_per_bucket > min_fold_train_pool:
        max_total = min_fold_train_pool * bucket_count
        raise ValueError(
            f"The largest requested training size needs {max_requested_per_bucket} images per class, "
            f"but the smallest fold training pool has only {min_fold_train_pool}. "
            f"Reduce --train-sizes or --folds. The largest valid total size here is {max_total}."
        )

    manifest = {
        "train_root": str(train_root),
        "test_root": str(test_root),
        "train_sizes_total": train_sizes,
        "train_sizes_per_class": per_bucket_train_sizes,
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "device": device,
        "model_name": args.model_name,
        "final_train_mode": args.final_train_mode,
        "plateau_ratio": args.plateau_ratio,
        "buckets": [],
    }

    print("Fixed-split CV plan:")
    print(f"  train root: {train_root}")
    print(f"  test root: {test_root}")
    print(f"  classes: {', '.join(CLASS_NAMES)}")
    print(f"  train sizes total: {train_sizes}")
    print(f"  train sizes per class: {per_bucket_train_sizes}")
    print(f"  folds: {args.folds}")
    print(f"  repeats per fold: {args.repeats}")
    print(f"  final train mode: {args.final_train_mode}")

    train_samples = []
    test_samples = []
    for bucket in bucket_entries:
        manifest["buckets"].append(
            {
                "bucket_name": bucket["bucket_name"],
                "class_name": bucket["class_name"],
                "train_count": len(bucket["train_files"]),
                "test_count": len(bucket["test_files"]),
                "fold_sizes": [len(items) for items in bucket["folds"]],
                "train_files": [str(path) for path in bucket["train_files"]],
                "test_files": [str(path) for path in bucket["test_files"]],
            }
        )

        class_id = CLASS_TO_ID[bucket["class_name"]]
        train_samples.extend((str(path), class_id) for path in bucket["train_files"])
        test_samples.extend((str(path), class_id) for path in bucket["test_files"])

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")

    cache_dir = output_root / "cache"
    train_embed_path = cache_dir / "train_embeddings.npz"
    test_embed_path = cache_dir / "fixed_test_embeddings.npz"

    expected_train_paths = [sample_path for sample_path, _ in train_samples]
    expected_test_paths = [sample_path for sample_path, _ in test_samples]

    train_cache_matches = embedding_cache_matches_expected(train_embed_path, expected_train_paths)
    test_cache_matches = embedding_cache_matches_expected(test_embed_path, expected_test_paths)

    need_train_embeddings = args.force or not train_cache_matches
    need_test_embeddings = args.force or not test_cache_matches

    if train_embed_path.exists() and not train_cache_matches and not args.force:
        print(
            "Train embedding cache does not match the current manifest. "
            "Rebuilding train embeddings."
        )
    if test_embed_path.exists() and not test_cache_matches and not args.force:
        print(
            "Fixed-test embedding cache does not match the current manifest. "
            "Rebuilding fixed-test embeddings."
        )

    if need_train_embeddings or need_test_embeddings:
        processor, model = load_siglip_processor_and_model(args.model_name, device)
        if need_train_embeddings:
            X_train_all, y_train_all, train_paths = extract_embeddings_for_samples(
                train_samples,
                processor=processor,
                model=model,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                desc="Extracting train embeddings",
            )
            save_embeddings(train_embed_path, X_train_all, y_train_all, train_paths)
            print(f"Saved train embeddings to {train_embed_path}")

        if need_test_embeddings:
            X_test, y_test, test_paths = extract_embeddings_for_samples(
                test_samples,
                processor=processor,
                model=model,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                desc="Extracting fixed test embeddings",
            )
            save_embeddings(test_embed_path, X_test, y_test, test_paths)
            print(f"Saved fixed test embeddings to {test_embed_path}")

    X_train_all, y_train_all, train_paths = load_embeddings(train_embed_path)
    X_test, y_test, test_paths = load_embeddings(test_embed_path)
    train_index_by_path = {path: idx for idx, path in enumerate(train_paths.tolist())}
    test_index_by_path = {path: idx for idx, path in enumerate(test_paths.tolist())}

    missing_train_paths = sorted(set(expected_train_paths) - set(train_index_by_path))
    missing_test_paths = sorted(set(expected_test_paths) - set(test_index_by_path))
    if missing_train_paths or missing_test_paths:
        messages = []
        if missing_train_paths:
            messages.append(
                f"train cache missing {len(missing_train_paths)} expected files, "
                f"for example {missing_train_paths[0]}"
            )
        if missing_test_paths:
            messages.append(
                f"test cache missing {len(missing_test_paths)} expected files, "
                f"for example {missing_test_paths[0]}"
            )
        raise RuntimeError(
            "Embedding cache validation failed after extraction/load: " + "; ".join(messages)
        )

    raw_cv_rows = []
    raw_test_curve_rows = []
    trial_models_dir = output_root / "trial_models"
    for fold_idx in range(args.folds):
        fold_number = fold_idx + 1
        fold_train_orders = {}
        fold_val_indices = []

        for bucket_idx, bucket in enumerate(bucket_entries):
            val_files = bucket["folds"][fold_idx]
            train_pool = []
            for other_fold_idx, other_fold_files in enumerate(bucket["folds"]):
                if other_fold_idx == fold_idx:
                    continue
                train_pool.extend(other_fold_files)

            val_indices = [train_index_by_path[str(path)] for path in val_files]
            fold_val_indices.extend(val_indices)

            for repeat_idx in range(args.repeats):
                repeat_rng = random.Random(
                    args.seed + 100_000 * fold_number + 10_000 * repeat_idx + bucket_idx
                )
                ordered_train_pool = train_pool[:]
                repeat_rng.shuffle(ordered_train_pool)
                fold_train_orders[(bucket_idx, repeat_idx)] = ordered_train_pool

        X_val = X_train_all[fold_val_indices]
        y_val = y_train_all[fold_val_indices]
        val_real_count = int(np.sum(y_val == CLASS_TO_ID["real"]))
        val_fake_count = int(np.sum(y_val == CLASS_TO_ID["fake"]))
        val_paths_for_fold = [train_paths[idx] for idx in fold_val_indices]

        for repeat_idx in range(args.repeats):
            repeat_number = repeat_idx + 1
            for total_size, per_bucket_size in zip(train_sizes, per_bucket_train_sizes):
                train_indices = []
                for bucket_idx, bucket in enumerate(bucket_entries):
                    selected_files = fold_train_orders[(bucket_idx, repeat_idx)][:per_bucket_size]
                    train_indices.extend(train_index_by_path[str(path)] for path in selected_files)

                X_train = X_train_all[train_indices]
                y_train = y_train_all[train_indices]
                train_real_count = int(np.sum(y_train == CLASS_TO_ID["real"]))
                train_fake_count = int(np.sum(y_train == CLASS_TO_ID["fake"]))
                train_paths_for_run = [train_paths[idx] for idx in train_indices]

                for model_name in MODEL_CHOICES:
                    clf, metrics, _, _ = train_and_score(
                        model_name=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_eval=X_val,
                        y_eval=y_val,
                    )
                    trial_model_path = ""
                    if args.save_trial_models:
                        trial_model_path = str(
                            trial_models_dir
                            / model_name
                            / f"size_{total_size:04d}"
                            / f"fold_{fold_number:02d}_repeat_{repeat_number:02d}.joblib"
                        )
                        Path(trial_model_path).parent.mkdir(parents=True, exist_ok=True)
                        joblib.dump(clf, trial_model_path)

                    row = {
                        "model": model_name,
                        "train_size_total": total_size,
                        "fold": fold_number,
                        "repeat": repeat_number,
                        "train_real_count": train_real_count,
                        "train_fake_count": train_fake_count,
                        "val_real_count": val_real_count,
                        "val_fake_count": val_fake_count,
                        "train_paths_count": len(train_paths_for_run),
                        "val_paths_count": len(val_paths_for_fold),
                        "model_path": trial_model_path,
                    }
                    row.update(metrics)
                    raw_cv_rows.append(row)

                    test_preds = clf.predict(X_test)
                    test_probs = predict_positive_scores(clf, X_test)
                    test_metrics = compute_metrics(y_test, test_preds, probs=test_probs)
                    test_row = {
                        "model": model_name,
                        "train_size_total": total_size,
                        "fold": fold_number,
                        "repeat": repeat_number,
                        "train_real_count": train_real_count,
                        "train_fake_count": train_fake_count,
                        "test_real_count": int(np.sum(y_test == CLASS_TO_ID["real"])),
                        "test_fake_count": int(np.sum(y_test == CLASS_TO_ID["fake"])),
                        "train_paths_count": len(train_paths_for_run),
                        "model_path": trial_model_path,
                    }
                    test_row.update(test_metrics)
                    raw_test_curve_rows.append(test_row)

    cv_summary_rows = aggregate_rows(raw_cv_rows, group_fields=("model", "train_size_total"))
    test_curve_summary_rows = aggregate_rows(
        raw_test_curve_rows,
        group_fields=("model", "train_size_total"),
    )

    best_overall = choose_best_row(cv_summary_rows)
    selection_rows = []
    for model_name in MODEL_CHOICES:
        best_row = choose_best_row(cv_summary_rows, model_name=model_name)
        plateau_row = find_plateau_row(
            cv_summary_rows,
            model_name=model_name,
            ratio=args.plateau_ratio,
        )
        selection_rows.append(
            {
                "model": model_name,
                "best_cv_train_size_total": best_row["train_size_total"],
                "best_cv_f1_mean": best_row["f1_mean"],
                "best_cv_f1_std": best_row["f1_std"],
                "best_cv_roc_auc_mean": best_row["roc_auc_mean"],
                "plateau_train_size_total": plateau_row["train_size_total"],
                "plateau_f1_mean": plateau_row["f1_mean"],
                "plateau_f1_std": plateau_row["f1_std"],
            }
        )

    summary_dir = output_root / "summary"
    selection_path = summary_dir / "selection_summary.csv"
    write_csv(
        selection_path,
        selection_rows,
        fieldnames=[
            "model",
            "best_cv_train_size_total",
            "best_cv_f1_mean",
            "best_cv_f1_std",
            "best_cv_roc_auc_mean",
            "plateau_train_size_total",
            "plateau_f1_mean",
            "plateau_f1_std",
        ],
    )

    best_model_per_size_rows = []
    for train_size_total in train_sizes:
        best_size_row = choose_best_row_for_train_size(
            cv_summary_rows,
            train_size_total=train_size_total,
        )
        test_curve_row = lookup_summary_row(
            test_curve_summary_rows,
            model_name=best_size_row["model"],
            train_size_total=train_size_total,
        )
        best_model_per_size_rows.append(
            {
                "train_size_total": train_size_total,
                "best_model": best_size_row["model"],
                "cv_runs": best_size_row["runs"],
                "cv_accuracy_mean": best_size_row["accuracy_mean"],
                "cv_accuracy_std": best_size_row["accuracy_std"],
                "cv_precision_mean": best_size_row["precision_mean"],
                "cv_precision_std": best_size_row["precision_std"],
                "cv_recall_mean": best_size_row["recall_mean"],
                "cv_recall_std": best_size_row["recall_std"],
                "cv_f1_mean": best_size_row["f1_mean"],
                "cv_f1_std": best_size_row["f1_std"],
                "cv_roc_auc_mean": best_size_row["roc_auc_mean"],
                "cv_roc_auc_std": best_size_row["roc_auc_std"],
                "fixed_test_runs": test_curve_row["runs"],
                "fixed_test_accuracy_mean": test_curve_row["accuracy_mean"],
                "fixed_test_accuracy_std": test_curve_row["accuracy_std"],
                "fixed_test_precision_mean": test_curve_row["precision_mean"],
                "fixed_test_precision_std": test_curve_row["precision_std"],
                "fixed_test_recall_mean": test_curve_row["recall_mean"],
                "fixed_test_recall_std": test_curve_row["recall_std"],
                "fixed_test_f1_mean": test_curve_row["f1_mean"],
                "fixed_test_f1_std": test_curve_row["f1_std"],
                "fixed_test_roc_auc_mean": test_curve_row["roc_auc_mean"],
                "fixed_test_roc_auc_std": test_curve_row["roc_auc_std"],
            }
        )

    write_csv(
        summary_dir / "cv_raw_metrics.csv",
        raw_cv_rows,
        fieldnames=[
            "model",
            "train_size_total",
            "fold",
            "repeat",
            "train_real_count",
            "train_fake_count",
            "val_real_count",
            "val_fake_count",
            "train_paths_count",
            "val_paths_count",
            "model_path",
            *METRIC_NAMES,
        ],
    )
    write_csv(
        summary_dir / "cv_summary_metrics.csv",
        cv_summary_rows,
        fieldnames=[
            "model",
            "train_size_total",
            "runs",
            *[f"{metric_name}_{suffix}" for metric_name in METRIC_NAMES for suffix in ("mean", "std")],
        ],
    )
    write_csv(
        summary_dir / "test_curve_raw_metrics.csv",
        raw_test_curve_rows,
        fieldnames=[
            "model",
            "train_size_total",
            "fold",
            "repeat",
            "train_real_count",
            "train_fake_count",
            "test_real_count",
            "test_fake_count",
            "train_paths_count",
            "model_path",
            *METRIC_NAMES,
        ],
    )
    write_csv(
        summary_dir / "test_curve_summary_metrics.csv",
        [
            {"evaluation_set": "fixed_test_curve", **row}
            for row in test_curve_summary_rows
        ],
        fieldnames=[
            "evaluation_set",
            "model",
            "train_size_total",
            "runs",
            *[f"{metric_name}_{suffix}" for metric_name in METRIC_NAMES for suffix in ("mean", "std")],
        ],
    )
    write_csv(
        summary_dir / "best_model_per_size.csv",
        best_model_per_size_rows,
        fieldnames=[
            "train_size_total",
            "best_model",
            "cv_runs",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "cv_precision_mean",
            "cv_precision_std",
            "cv_recall_mean",
            "cv_recall_std",
            "cv_f1_mean",
            "cv_f1_std",
            "cv_roc_auc_mean",
            "cv_roc_auc_std",
            "fixed_test_runs",
            "fixed_test_accuracy_mean",
            "fixed_test_accuracy_std",
            "fixed_test_precision_mean",
            "fixed_test_precision_std",
            "fixed_test_recall_mean",
            "fixed_test_recall_std",
            "fixed_test_f1_mean",
            "fixed_test_f1_std",
            "fixed_test_roc_auc_mean",
            "fixed_test_roc_auc_std",
        ],
    )

    full_train_indices = list(range(len(train_paths)))
    full_train_real_count = int(np.sum(y_train_all == CLASS_TO_ID["real"]))
    full_train_fake_count = int(np.sum(y_train_all == CLASS_TO_ID["fake"]))

    final_models_dir = output_root / "final_models"
    final_reports_dir = output_root / "final_reports"
    final_models_dir.mkdir(parents=True, exist_ok=True)
    final_reports_dir.mkdir(parents=True, exist_ok=True)

    final_test_rows = []
    for model_idx, model_name in enumerate(MODEL_CHOICES):
        best_row = choose_best_row(cv_summary_rows, model_name=model_name)
        plateau_row = find_plateau_row(
            cv_summary_rows,
            model_name=model_name,
            ratio=args.plateau_ratio,
        )

        if args.final_train_mode == "full_train":
            final_train_indices = full_train_indices
        else:
            selected_total_size = (
                best_row["train_size_total"]
                if args.final_train_mode == "best_size"
                else plateau_row["train_size_total"]
            )
            per_bucket_size = selected_total_size // bucket_count
            final_train_indices = []
            for bucket_idx, bucket in enumerate(bucket_entries):
                import random

                final_rng = random.Random(
                    args.seed + 900_000 + 1_000 * model_idx + bucket_idx
                )
                ordered_train_files = bucket["train_files"][:]
                final_rng.shuffle(ordered_train_files)
                chosen_files = ordered_train_files[:per_bucket_size]
                final_train_indices.extend(
                    train_index_by_path[str(path)] for path in chosen_files
                )

        X_train_final = X_train_all[final_train_indices]
        y_train_final = y_train_all[final_train_indices]
        train_real_count = int(np.sum(y_train_final == CLASS_TO_ID["real"]))
        train_fake_count = int(np.sum(y_train_final == CLASS_TO_ID["fake"]))

        clf, metrics, preds, probs = train_and_score(
            model_name=model_name,
            X_train=X_train_final,
            y_train=y_train_final,
            X_eval=X_test,
            y_eval=y_test,
        )

        model_path = final_models_dir / f"{model_name}.joblib"
        joblib.dump(clf, model_path)

        _, report_text = format_report(y_test, preds, probs=probs, include_confusion=True)
        report_path = final_reports_dir / f"test_metrics_{model_name}.txt"
        with open(report_path, "w") as f:
            f.write(report_text)

        final_test_rows.append(
            {
                "model": model_name,
                "final_train_mode": args.final_train_mode,
                "final_train_size_total": len(final_train_indices),
                "final_train_real_count": train_real_count,
                "final_train_fake_count": train_fake_count,
                "train_pool_size_total": len(full_train_indices),
                "train_pool_real_count": full_train_real_count,
                "train_pool_fake_count": full_train_fake_count,
                "test_size_total": len(test_index_by_path),
                "test_real_count": int(np.sum(y_test == CLASS_TO_ID["real"])),
                "test_fake_count": int(np.sum(y_test == CLASS_TO_ID["fake"])),
                "best_cv_train_size_total": best_row["train_size_total"],
                "plateau_train_size_total": plateau_row["train_size_total"],
                "model_path": str(model_path),
                "report_path": str(report_path),
                **metrics,
            }
        )

    write_csv(
        summary_dir / "final_test_metrics.csv",
        final_test_rows,
        fieldnames=[
            "model",
            "final_train_mode",
            "final_train_size_total",
            "final_train_real_count",
            "final_train_fake_count",
            "train_pool_size_total",
            "train_pool_real_count",
            "train_pool_fake_count",
            "test_size_total",
            "test_real_count",
            "test_fake_count",
            "best_cv_train_size_total",
            "plateau_train_size_total",
            "model_path",
            "report_path",
            *METRIC_NAMES,
        ],
    )

    lines = [
        "# SigLIP Fixed-Split CV Report",
        "",
        "## Setup",
        f"- Train root: {train_root}",
        f"- Test root: {test_root}",
        f"- Total train sizes evaluated: {', '.join(str(size) for size in train_sizes)}",
        f"- Train pool total: {len(full_train_indices)}",
        f"- Fixed test total: {len(test_index_by_path)}",
        f"- K folds: {args.folds}",
        f"- Repeats per fold: {args.repeats}",
        f"- Final train mode before untouched test: `{args.final_train_mode}`",
        "",
        (
            "Model and training-size selection should be based on the CV tables below. "
            "The fixed-test curve is written separately for exploratory visualization."
        ),
        "",
        "## Best CV Configuration",
        (
            f"- Overall best CV result: `{best_overall['model']}` at training size "
            f"`{best_overall['train_size_total']}` with fake f1 "
            f"`{format_metric(best_overall['f1_mean'], best_overall['f1_std'])}`"
        ),
        "",
        "## Per-Model CV Summary",
        "| Model | Best CV Size | Plateau Size | CV F1 | CV Recall | CV ROC AUC |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]

    for model_name in MODEL_CHOICES:
        best_row = choose_best_row(cv_summary_rows, model_name=model_name)
        plateau_row = find_plateau_row(
            cv_summary_rows,
            model_name=model_name,
            ratio=args.plateau_ratio,
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    str(best_row["train_size_total"]),
                    str(plateau_row["train_size_total"]),
                    format_metric(best_row["f1_mean"], best_row["f1_std"]),
                    format_metric(best_row["recall_mean"], best_row["recall_std"]),
                    format_metric(best_row["roc_auc_mean"], best_row["roc_auc_std"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Best Model At Each Training Size",
            "| Train Size | Best Model (CV) | CV F1 | CV ROC AUC | Fixed-Test F1 | Fixed-Test ROC AUC |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for row in best_model_per_size_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["train_size_total"]),
                    row["best_model"],
                    format_metric(row["cv_f1_mean"], row["cv_f1_std"]),
                    format_metric(row["cv_roc_auc_mean"], row["cv_roc_auc_std"]),
                    format_metric(row["fixed_test_f1_mean"], row["fixed_test_f1_std"]),
                    format_metric(row["fixed_test_roc_auc_mean"], row["fixed_test_roc_auc_std"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## CV Learning Curve",
            "| Model | Train Size | Fake F1 | Fake Recall | ROC AUC | Accuracy |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for model_name in MODEL_CHOICES:
        model_rows = sorted(
            [row for row in cv_summary_rows if row["model"] == model_name],
            key=lambda row: row["train_size_total"],
        )
        for row in model_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_name,
                        str(row["train_size_total"]),
                        format_metric(row["f1_mean"], row["f1_std"]),
                        format_metric(row["recall_mean"], row["recall_std"]),
                        format_metric(row["roc_auc_mean"], row["roc_auc_std"]),
                        format_metric(row["accuracy_mean"], row["accuracy_std"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Fixed-Test Curve (Exploratory Only)",
            (
                "The table below evaluates every training size against the same fixed test set. "
                "Use it for visualization only, because repeatedly inspecting the test set makes it "
                "less suitable as a strict final benchmark."
            ),
            "",
            "| Model | Train Size | Test F1 | Test Recall | Test ROC AUC | Test Accuracy |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for model_name in MODEL_CHOICES:
        model_rows = sorted(
            [row for row in test_curve_summary_rows if row["model"] == model_name],
            key=lambda row: row["train_size_total"],
        )
        for row in model_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_name,
                        str(row["train_size_total"]),
                        format_metric(row["f1_mean"], row["f1_std"]),
                        format_metric(row["recall_mean"], row["recall_std"]),
                        format_metric(row["roc_auc_mean"], row["roc_auc_std"]),
                        format_metric(row["accuracy_mean"], row["accuracy_std"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Final Untouched Test Benchmark",
            (
                f"The final test numbers below were computed after training each model with "
                f"`{args.final_train_mode}` on the train split."
            ),
            "",
            "| Model | Final Train Size | Test F1 | Test Recall | Test ROC AUC | Test Accuracy |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in sorted(final_test_rows, key=lambda item: item["model"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    str(row["final_train_size_total"]),
                    f"{row['f1']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['roc_auc']:.4f}" if "roc_auc" in row else "n/a",
                    f"{row['accuracy']:.4f}",
                ]
            )
            + " |"
        )

    final_report_path = summary_dir / "final_report.md"
    with open(final_report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\nFinished fixed-split CV experiment.")
    print(f"Manifest: {manifest_path}")
    print(f"CV raw metrics: {summary_dir / 'cv_raw_metrics.csv'}")
    print(f"CV summary metrics: {summary_dir / 'cv_summary_metrics.csv'}")
    print(f"Fixed-test curve summary: {summary_dir / 'test_curve_summary_metrics.csv'}")
    print(f"Best model per size: {summary_dir / 'best_model_per_size.csv'}")
    print(f"Final test metrics: {summary_dir / 'final_test_metrics.csv'}")
    print(f"Final report: {final_report_path}")
    if args.save_trial_models:
        print(f"Trial models: {trial_models_dir}")


if __name__ == "__main__":
    main()
