import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGLIP_ROOT = PROJECT_ROOT / "siglip"
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train SigLIP classifier heads on concatenated global-image and crop-image embeddings. "
            "Embeddings are aligned by image path before concatenation."
        )
    )
    parser.add_argument("--global-train", required=True, help="Global train embedding npz.")
    parser.add_argument("--crop-train", required=True, help="Crop train embedding npz.")
    parser.add_argument("--global-test", required=True, help="Global test embedding npz.")
    parser.add_argument("--crop-test", required=True, help="Crop test embedding npz.")
    parser.add_argument("--global-val", default=None, help="Optional global val embedding npz.")
    parser.add_argument("--crop-val", default=None, help="Optional crop val embedding npz.")
    parser.add_argument(
        "--output-root",
        default="outputs/siglip_concat_fusion",
        help="Output directory for fused embeddings, models, and reports.",
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_CHOICES, "all"],
        default="all",
        help="Classifier head to train on the concatenated embeddings.",
    )
    parser.add_argument(
        "--l2-normalize-concat",
        action="store_true",
        help="L2-normalize the concatenated feature vectors before training.",
    )
    return parser.parse_args()


def resolve_path(raw_path: str | None):
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    paths = [str(item) for item in data["paths"].tolist()]
    return X, y, paths


def align_embeddings(global_path: Path, crop_path: Path):
    X_global, y_global, paths_global = load_npz(global_path)
    X_crop, y_crop, paths_crop = load_npz(crop_path)

    global_rows = {
        path: (X_global[idx], int(y_global[idx]))
        for idx, path in enumerate(paths_global)
    }
    crop_rows = {
        path: (X_crop[idx], int(y_crop[idx]))
        for idx, path in enumerate(paths_crop)
    }

    shared_paths = sorted(set(global_rows) & set(crop_rows))
    if not shared_paths:
        raise ValueError(
            f"No shared image paths between {global_path} and {crop_path}."
        )

    X_fused = []
    y_fused = []
    for path in shared_paths:
        global_features, global_label = global_rows[path]
        crop_features, crop_label = crop_rows[path]
        if global_label != crop_label:
            raise ValueError(
                f"Label mismatch for {path}: global={global_label}, crop={crop_label}"
            )
        X_fused.append(np.concatenate([global_features, crop_features], axis=0))
        y_fused.append(global_label)

    X_fused = np.stack(X_fused, axis=0).astype(np.float32)
    y_fused = np.asarray(y_fused, dtype=np.int64)
    return X_fused, y_fused, shared_paths


def l2_normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms


def save_npz(path: Path, X, y, paths):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, paths=np.asarray(paths))


def get_model_output_paths(output_root: Path, model_name: str):
    models_dir = output_root / "models"
    reports_dir = output_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return (
        models_dir / f"{model_name}.joblib",
        reports_dir / f"val_metrics_{model_name}.txt",
        reports_dir / f"test_metrics_{model_name}.txt",
    )


def train_one_model(model_name, X_train, y_train, X_val, y_val, output_root: Path):
    clf = build_classifier(model_name, y_train)
    clf.fit(X_train, y_train)

    model_path, val_report_path, _ = get_model_output_paths(output_root, model_name)
    joblib.dump(clf, model_path)

    val_metrics = None
    if X_val is not None and y_val is not None:
        val_preds = clf.predict(X_val)
        val_probs = predict_positive_scores(clf, X_val)
        val_metrics, val_report_text = format_report(y_val, val_preds, probs=val_probs)
        with open(val_report_path, "w") as handle:
            handle.write(val_report_text)

        print(f"\nValidation metrics for {model_name}:")
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")

    print(f"Saved model to {model_path}")
    return clf, val_metrics


def evaluate_one_model(model_name, clf, X_test, y_test, output_root: Path):
    _, _, test_report_path = get_model_output_paths(output_root, model_name)
    preds = clf.predict(X_test)
    probs = predict_positive_scores(clf, X_test)
    test_metrics, test_report_text = format_report(
        y_test,
        preds,
        probs=probs,
        include_confusion=True,
    )
    with open(test_report_path, "w") as handle:
        handle.write(test_report_text)

    print(f"\nTest metrics for {model_name}:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"Saved test report to {test_report_path}")
    return test_metrics


def main():
    args = parse_args()

    global_train = resolve_path(args.global_train)
    crop_train = resolve_path(args.crop_train)
    global_val = resolve_path(args.global_val)
    crop_val = resolve_path(args.crop_val)
    global_test = resolve_path(args.global_test)
    crop_test = resolve_path(args.crop_test)
    output_root = resolve_path(args.output_root)
    embeddings_dir = output_root / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing concatenated global+crop embeddings...")
    X_train, y_train, train_paths = align_embeddings(global_train, crop_train)
    X_test, y_test, test_paths = align_embeddings(global_test, crop_test)
    X_val = y_val = val_paths = None
    if global_val and crop_val:
        X_val, y_val, val_paths = align_embeddings(global_val, crop_val)
    elif global_val or crop_val:
        raise ValueError("Provide both --global-val and --crop-val, or neither.")

    if args.l2_normalize_concat:
        X_train = l2_normalize_rows(X_train)
        X_test = l2_normalize_rows(X_test)
        if X_val is not None:
            X_val = l2_normalize_rows(X_val)

    save_npz(embeddings_dir / "train_embeddings.npz", X_train, y_train, train_paths)
    save_npz(embeddings_dir / "test_embeddings.npz", X_test, y_test, test_paths)
    if X_val is not None:
        save_npz(embeddings_dir / "val_embeddings.npz", X_val, y_val, val_paths)

    print(f"Train embeddings shape: {X_train.shape}")
    if X_val is not None:
        print(f"Val embeddings shape: {X_val.shape}")
    print(f"Test embeddings shape: {X_test.shape}")

    selected_models = MODEL_CHOICES if args.model == "all" else (args.model,)
    summary = {
        "fusion_method": "concat",
        "l2_normalize_concat": bool(args.l2_normalize_concat),
        "embedding_dim": int(X_train.shape[1]),
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]) if X_val is not None else 0,
        "test_size": int(X_test.shape[0]),
        "global_train": str(global_train),
        "crop_train": str(crop_train),
        "global_val": str(global_val) if global_val else None,
        "crop_val": str(crop_val) if crop_val else None,
        "global_test": str(global_test),
        "crop_test": str(crop_test),
        "models": {},
    }

    for model_name in selected_models:
        print("\n" + "=" * 80)
        print(f"Training concatenated fusion model: {model_name}")
        clf, val_metrics = train_one_model(
            model_name,
            X_train,
            y_train,
            X_val,
            y_val,
            output_root=output_root,
        )
        test_metrics = evaluate_one_model(
            model_name,
            clf,
            X_test,
            y_test,
            output_root=output_root,
        )
        summary["models"][model_name] = {
            "val": val_metrics,
            "test": test_metrics,
        }

    with open(output_root / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSaved summary to {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
