import argparse
import csv
import json
import os
import random
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".matplotlib-cache").resolve()))
SIGLIP_ROOT = PROJECT_ROOT / "siglip"
if str(SIGLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(SIGLIP_ROOT))

from classifier_utils import (  # noqa: E402
    MODEL_CHOICES,
    build_classifier,
    compute_metrics,
    format_report,
    predict_positive_scores,
)
from dataset import ImagePathDataset  # noqa: E402
from feature_utils import get_normalized_image_features  # noqa: E402
from hf_utils import load_siglip_processor_and_model  # noqa: E402


CLASS_NAMES = ("real", "fake")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a SigLIP learning-curve experiment with a fixed final test holdout "
            "and k-fold validation on the remaining development pool."
        )
    )
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        help="Labeled source dataset root. Repeat to include multiple sources.",
    )
    parser.add_argument(
        "--train-sizes",
        required=True,
        help=(
            "Comma-separated total training sizes. Each size must be divisible by "
            "(#sources * 2 classes)."
        ),
    )
    parser.add_argument(
        "--test-per-bucket",
        type=int,
        default=None,
        help="Final test images per (source, class) bucket.",
    )
    parser.add_argument(
        "--test-total",
        type=int,
        default=None,
        help="Final test images across all source/class buckets.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of validation folds inside the development pool.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeated subset samplings per fold and train size.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/learning_curve_cv",
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
        choices=("full_dev", "best_size", "plateau_size"),
        default="full_dev",
        help=(
            "How to train the final model before the untouched test evaluation. "
            "`full_dev` is the practical default."
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
        help="Random seed for test holdout, fold assignment, and subset sampling.",
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


def parse_train_sizes(raw_sizes: str):
    sizes = []
    for item in raw_sizes.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        size = int(stripped)
        if size <= 0:
            raise ValueError(f"Training sizes must be positive integers, got {size}.")
        sizes.append(size)

    if not sizes:
        raise ValueError("At least one training size must be provided.")
    if sizes != sorted(set(sizes)):
        raise ValueError("Training sizes must be unique and sorted in ascending order.")
    return sizes


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


def bucket_name(source_name: str, class_name: str):
    return f"{source_name}/{class_name}"


def collate_fn(batch):
    images, labels, paths = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(paths)


@torch.no_grad()
def extract_embeddings_for_samples(samples, processor, model, device: str, batch_size: int, num_workers: int, desc: str):
    if not samples:
        raise ValueError(f"No samples provided for {desc}.")

    dataset = ImagePathDataset(samples)
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

    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)
    paths_arr = np.array(all_paths)
    return X, y, paths_arr


def save_embeddings(path: Path, X, y, paths):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, paths=np.array(paths))


def load_embeddings(path: Path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["paths"]


def embedding_cache_matches_expected(cache_path: Path, expected_paths):
    if not cache_path.exists():
        return False

    try:
        _, _, cached_paths = load_embeddings(cache_path)
    except Exception:
        return False

    cached_list = cached_paths.tolist()
    expected_list = list(expected_paths)
    return len(cached_list) == len(expected_list) and set(cached_list) == set(expected_list)


def split_into_folds(items, num_folds: int):
    if len(items) < num_folds:
        raise ValueError(
            f"Cannot create {num_folds} folds from only {len(items)} items."
        )

    base_size, remainder = divmod(len(items), num_folds)
    folds = []
    start_idx = 0
    for fold_idx in range(num_folds):
        fold_size = base_size + (1 if fold_idx < remainder else 0)
        folds.append(items[start_idx:start_idx + fold_size])
        start_idx += fold_size
    return folds


def train_and_score(model_name: str, X_train, y_train, X_eval, y_eval):
    clf = build_classifier(model_name, y_train)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_eval)
    probs = predict_positive_scores(clf, X_eval)
    metrics = compute_metrics(y_eval, preds, probs=probs)
    return clf, metrics, preds, probs


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped[key].append(row)

    summary_rows = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["runs"] = len(group_rows)
        for metric_name in METRIC_NAMES:
            values = [row[metric_name] for row in group_rows if metric_name in row]
            summary[f"{metric_name}_mean"] = statistics.mean(values)
            summary[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    return summary_rows


def choose_best_row(summary_rows, model_name=None):
    candidates = summary_rows
    if model_name is not None:
        candidates = [row for row in summary_rows if row["model"] == model_name]
    if not candidates:
        raise ValueError(f"No summary rows available for model={model_name!r}")
    return max(
        candidates,
        key=lambda row: (
            row["f1_mean"],
            row["roc_auc_mean"],
            -row["train_size_total"],
        ),
    )


def choose_best_row_for_train_size(summary_rows, train_size_total: int):
    candidates = [
        row for row in summary_rows
        if row["train_size_total"] == train_size_total
    ]
    if not candidates:
        raise ValueError(f"No summary rows available for train_size_total={train_size_total}")
    return max(
        candidates,
        key=lambda row: (
            row["f1_mean"],
            row["roc_auc_mean"],
            row["accuracy_mean"],
            row["recall_mean"],
        ),
    )


def find_plateau_row(summary_rows, model_name: str, ratio: float):
    candidates = sorted(
        [row for row in summary_rows if row["model"] == model_name],
        key=lambda row: row["train_size_total"],
    )
    if not candidates:
        raise ValueError(f"No summary rows available for model={model_name}")
    peak = max(row["f1_mean"] for row in candidates)
    threshold = peak * ratio
    for row in candidates:
        if row["f1_mean"] >= threshold:
            return row
    return candidates[-1]


def format_metric(mean_value: float, std_value: float):
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def lookup_summary_row(summary_rows, model_name: str, train_size_total: int):
    for row in summary_rows:
        if row["model"] == model_name and row["train_size_total"] == train_size_total:
            return row
    raise KeyError(
        f"Missing summary row for model={model_name}, train_size_total={train_size_total}"
    )


def main():
    args = parse_args()
    output_root = resolve_path(args.output_root)
    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = detect_device() if args.device == "auto" else args.device
    source_roots = [resolve_path(raw_path) for raw_path in args.source_root]
    train_sizes = parse_train_sizes(args.train_sizes)

    for source_root in source_roots:
        if not source_root.exists():
            raise FileNotFoundError(f"Source root does not exist: {source_root}")

    bucket_count = len(source_roots) * len(CLASS_NAMES)
    test_per_bucket = resolve_holdout_per_bucket(
        total=args.test_total,
        per_bucket=args.test_per_bucket,
        bucket_count=bucket_count,
        split_name="test",
        default_per_bucket=20,
    )
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    per_bucket_train_sizes = []
    for total_size in train_sizes:
        if total_size % bucket_count != 0:
            raise ValueError(
                f"Training size {total_size} is not divisible by {bucket_count} buckets."
            )
        per_bucket_train_sizes.append(total_size // bucket_count)

    holdout_rng = random.Random(args.seed)
    bucket_entries = []
    for source_root in source_roots:
        source_name = source_root.name
        for class_name in CLASS_NAMES:
            files = collect_class_files(source_root, class_name)
            if not files:
                raise ValueError(f"No {class_name} samples found under {source_root}")

            shuffled = files[:]
            holdout_rng.shuffle(shuffled)
            if len(shuffled) <= test_per_bucket:
                raise ValueError(
                    f"{source_name}/{class_name} has {len(shuffled)} files, not enough "
                    f"to reserve {test_per_bucket} final test images."
                )

            test_files = shuffled[:test_per_bucket]
            dev_files = shuffled[test_per_bucket:]
            if len(dev_files) < args.folds:
                raise ValueError(
                    f"{source_name}/{class_name} leaves only {len(dev_files)} development files "
                    f"after the fixed test split, which is not enough for {args.folds} folds."
                )

            bucket_entries.append(
                {
                    "source_name": source_name,
                    "class_name": class_name,
                    "bucket_name": bucket_name(source_name, class_name),
                    "total_files": files,
                    "test_files": test_files,
                    "dev_files": dev_files,
                }
            )

    fold_rng = random.Random(args.seed + 1_000)
    for bucket in bucket_entries:
        shuffled_dev = bucket["dev_files"][:]
        fold_rng.shuffle(shuffled_dev)
        bucket["dev_files"] = shuffled_dev
        bucket["folds"] = split_into_folds(shuffled_dev, args.folds)

    min_fold_train_pool = min(
        len(bucket["dev_files"]) - len(bucket["folds"][fold_idx])
        for bucket in bucket_entries
        for fold_idx in range(args.folds)
    )
    max_requested_per_bucket = per_bucket_train_sizes[-1]
    if max_requested_per_bucket > min_fold_train_pool:
        max_total = min_fold_train_pool * bucket_count
        raise ValueError(
            f"The largest requested training size needs {max_requested_per_bucket} images per bucket, "
            f"but the smallest fold training pool has only {min_fold_train_pool}. "
            f"Reduce --train-sizes or --folds. The largest valid total size here is {max_total}."
        )

    manifest = {
        "sources": [str(source_root) for source_root in source_roots],
        "train_sizes_total": train_sizes,
        "train_sizes_per_bucket": per_bucket_train_sizes,
        "test_per_bucket": test_per_bucket,
        "test_total": test_per_bucket * bucket_count,
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "device": device,
        "model_name": args.model_name,
        "final_train_mode": args.final_train_mode,
        "plateau_ratio": args.plateau_ratio,
        "buckets": [],
    }

    print("Learning-curve CV plan:")
    print(f"  sources: {', '.join(source_root.name for source_root in source_roots)}")
    print(f"  buckets: {bucket_count}")
    print(f"  train sizes total: {train_sizes}")
    print(f"  train sizes per bucket: {per_bucket_train_sizes}")
    print(f"  fixed final test per bucket: {test_per_bucket}")
    print(f"  fixed final test total: {test_per_bucket * bucket_count}")
    print(f"  folds: {args.folds}")
    print(f"  repeats per fold: {args.repeats}")
    print(f"  final train mode: {args.final_train_mode}")

    dev_samples = []
    test_samples = []
    for bucket in bucket_entries:
        manifest["buckets"].append(
            {
                "bucket_name": bucket["bucket_name"],
                "source_name": bucket["source_name"],
                "class_name": bucket["class_name"],
                "total_count": len(bucket["total_files"]),
                "development_count": len(bucket["dev_files"]),
                "test_count": len(bucket["test_files"]),
                "fold_sizes": [len(items) for items in bucket["folds"]],
                "development_files": [str(path) for path in bucket["dev_files"]],
                "test_files": [str(path) for path in bucket["test_files"]],
            }
        )

        class_id = CLASS_TO_ID[bucket["class_name"]]
        dev_samples.extend((str(path), class_id) for path in bucket["dev_files"])
        test_samples.extend((str(path), class_id) for path in bucket["test_files"])

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")

    cache_dir = output_root / "cache"
    dev_embed_path = cache_dir / "development_embeddings.npz"
    test_embed_path = cache_dir / "final_test_embeddings.npz"

    expected_dev_paths = [sample_path for sample_path, _ in dev_samples]
    expected_test_paths = [sample_path for sample_path, _ in test_samples]

    dev_cache_matches = embedding_cache_matches_expected(dev_embed_path, expected_dev_paths)
    test_cache_matches = embedding_cache_matches_expected(test_embed_path, expected_test_paths)

    need_dev_embeddings = args.force or not dev_cache_matches
    need_test_embeddings = args.force or not test_cache_matches

    if dev_embed_path.exists() and not dev_cache_matches and not args.force:
        print(
            "Development embedding cache does not match the current manifest. "
            "Rebuilding development embeddings."
        )
    if test_embed_path.exists() and not test_cache_matches and not args.force:
        print(
            "Final test embedding cache does not match the current manifest. "
            "Rebuilding final test embeddings."
        )

    if need_dev_embeddings or need_test_embeddings:
        processor, model = load_siglip_processor_and_model(args.model_name, device)
        if need_dev_embeddings:
            X_dev, y_dev, dev_paths = extract_embeddings_for_samples(
                dev_samples,
                processor=processor,
                model=model,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                desc="Extracting development embeddings",
            )
            save_embeddings(dev_embed_path, X_dev, y_dev, dev_paths)
            print(f"Saved development embeddings to {dev_embed_path}")

        if need_test_embeddings:
            X_test, y_test, test_paths = extract_embeddings_for_samples(
                test_samples,
                processor=processor,
                model=model,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                desc="Extracting final test embeddings",
            )
            save_embeddings(test_embed_path, X_test, y_test, test_paths)
            print(f"Saved final test embeddings to {test_embed_path}")

    X_dev, y_dev, dev_paths = load_embeddings(dev_embed_path)
    X_test, y_test, test_paths = load_embeddings(test_embed_path)
    dev_index_by_path = {path: idx for idx, path in enumerate(dev_paths.tolist())}
    test_index_by_path = {path: idx for idx, path in enumerate(test_paths.tolist())}

    missing_dev_paths = sorted(set(expected_dev_paths) - set(dev_index_by_path))
    missing_test_paths = sorted(set(expected_test_paths) - set(test_index_by_path))
    if missing_dev_paths or missing_test_paths:
        messages = []
        if missing_dev_paths:
            messages.append(
                f"development cache missing {len(missing_dev_paths)} expected files, "
                f"for example {missing_dev_paths[0]}"
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

            val_indices = [dev_index_by_path[str(path)] for path in val_files]
            fold_val_indices.extend(val_indices)

            for repeat_idx in range(args.repeats):
                repeat_rng = random.Random(
                    args.seed + 100_000 * fold_number + 10_000 * repeat_idx + bucket_idx
                )
                ordered_train_pool = train_pool[:]
                repeat_rng.shuffle(ordered_train_pool)
                fold_train_orders[(bucket_idx, repeat_idx)] = ordered_train_pool

        X_val = X_dev[fold_val_indices]
        y_val = y_dev[fold_val_indices]
        val_real_count = int(np.sum(y_val == CLASS_TO_ID["real"]))
        val_fake_count = int(np.sum(y_val == CLASS_TO_ID["fake"]))
        val_paths_for_fold = [dev_paths[idx] for idx in fold_val_indices]

        for repeat_idx in range(args.repeats):
            repeat_number = repeat_idx + 1
            for total_size, per_bucket_size in zip(train_sizes, per_bucket_train_sizes):
                train_indices = []
                for bucket_idx, bucket in enumerate(bucket_entries):
                    selected_files = fold_train_orders[(bucket_idx, repeat_idx)][:per_bucket_size]
                    train_indices.extend(dev_index_by_path[str(path)] for path in selected_files)

                X_train = X_dev[train_indices]
                y_train = y_dev[train_indices]
                train_real_count = int(np.sum(y_train == CLASS_TO_ID["real"]))
                train_fake_count = int(np.sum(y_train == CLASS_TO_ID["fake"]))
                train_paths_for_run = [dev_paths[idx] for idx in train_indices]

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

    selection_path = output_root / "summary" / "selection_summary.csv"
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

    summary_dir = output_root / "summary"
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

    full_dev_indices = list(range(len(dev_paths)))
    full_dev_real_count = int(np.sum(y_dev == CLASS_TO_ID["real"]))
    full_dev_fake_count = int(np.sum(y_dev == CLASS_TO_ID["fake"]))

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

        if args.final_train_mode == "full_dev":
            final_train_indices = full_dev_indices
        else:
            if args.final_train_mode == "best_size":
                selected_total_size = best_row["train_size_total"]
            else:
                selected_total_size = plateau_row["train_size_total"]
            per_bucket_size = selected_total_size // bucket_count
            final_train_indices = []
            for bucket_idx, bucket in enumerate(bucket_entries):
                final_rng = random.Random(
                    args.seed + 900_000 + 1_000 * model_idx + bucket_idx
                )
                ordered_dev_files = bucket["dev_files"][:]
                final_rng.shuffle(ordered_dev_files)
                chosen_files = ordered_dev_files[:per_bucket_size]
                final_train_indices.extend(
                    dev_index_by_path[str(path)] for path in chosen_files
                )

        X_train_final = X_dev[final_train_indices]
        y_train_final = y_dev[final_train_indices]
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
                "development_pool_size_total": len(full_dev_indices),
                "development_pool_real_count": full_dev_real_count,
                "development_pool_fake_count": full_dev_fake_count,
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
            "development_pool_size_total",
            "development_pool_real_count",
            "development_pool_fake_count",
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
        "# SigLIP Learning Curve Report (Fixed Final Test + K-Fold CV)",
        "",
        "## Setup",
        f"- Sources: {', '.join(source_root.name for source_root in source_roots)}",
        f"- Total train sizes evaluated: {', '.join(str(size) for size in train_sizes)}",
        f"- Fixed final test per bucket: {test_per_bucket}",
        f"- Fixed final test total: {test_per_bucket * bucket_count}",
        f"- Development pool total: {len(full_dev_indices)}",
        f"- K folds: {args.folds}",
        f"- Repeats per fold: {args.repeats}",
        f"- Final train mode before untouched test: `{args.final_train_mode}`",
        "",
        (
            "Model and training-size selection should still be based on the CV tables below. "
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
                f"`{args.final_train_mode}` on the development pool."
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

    print("\nFinished fixed-test CV learning-curve experiment.")
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
