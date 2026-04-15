import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from poc_utils import compute_metrics, ensure_dir, save_csv, save_json, save_npz, save_text, train_logistic_regression


def log_message(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fuse saved global-image and crop-image embeddings without rerunning detectors. "
            "The default fusion is concatenation followed by LogisticRegression."
        )
    )
    parser.add_argument(
        "--global-root",
        required=True,
        help="Directory containing embeddings_train.npz and embeddings_test.npz for the global baseline.",
    )
    parser.add_argument(
        "--crop-root",
        required=True,
        help="Directory containing embeddings_train.npz and embeddings_test.npz for the crop baseline.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where fusion outputs will be written.",
    )
    parser.add_argument(
        "--fusion-method",
        choices=("concat", "avg_prob"),
        default="concat",
        help="Fusion strategy. `concat` trains on concatenated embeddings. `avg_prob` averages test probabilities.",
    )
    parser.add_argument(
        "--l2-normalize-concat",
        action="store_true",
        help="L2-normalize concatenated features before LogisticRegression.",
    )
    return parser.parse_args()


def load_embedding_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    paths = [str(item) for item in data["paths"].tolist()]
    return X, y, paths


def align_by_path(X, y, paths):
    rows = sorted(zip(paths, X, y), key=lambda item: item[0])
    aligned_paths = [path for path, _, _ in rows]
    aligned_X = np.stack([features for _, features, _ in rows], axis=0)
    aligned_y = np.asarray([label for _, _, label in rows], dtype=np.int64)
    return aligned_X, aligned_y, aligned_paths


def l2_normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms


def build_prediction_rows(paths, y_true, y_pred, y_prob):
    rows = []
    for path, truth, pred, prob in zip(paths, y_true, y_pred, y_prob):
        rows.append(
            {
                "image_path": path,
                "label_id": int(truth),
                "pred_label_id": int(pred),
                "fake_probability": float(prob),
                "correct": int(int(truth) == int(pred)),
            }
        )
    return rows


def format_report(title: str, metrics, notes):
    lines = [f"# {title}", "", "## Metrics"]
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]:.4f}")
    lines.append(f"- confusion_matrix: {metrics['confusion_matrix']}")
    lines.extend(["", "## Notes", *notes, ""])
    return "\n".join(lines)


def main():
    args = parse_args()

    global_root = Path(args.global_root).resolve()
    crop_root = Path(args.crop_root).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    log_message(
        f"starting embedding fusion (method={args.fusion_method}, "
        f"global_root={global_root.name}, crop_root={crop_root.name})"
    )

    Xg_train, yg_train, pg_train = load_embedding_npz(global_root / "embeddings_train.npz")
    Xg_test, yg_test, pg_test = load_embedding_npz(global_root / "embeddings_test.npz")
    Xc_train, yc_train, pc_train = load_embedding_npz(crop_root / "embeddings_train.npz")
    Xc_test, yc_test, pc_test = load_embedding_npz(crop_root / "embeddings_test.npz")

    Xg_train, yg_train, pg_train = align_by_path(Xg_train, yg_train, pg_train)
    Xg_test, yg_test, pg_test = align_by_path(Xg_test, yg_test, pg_test)
    Xc_train, yc_train, pc_train = align_by_path(Xc_train, yc_train, pc_train)
    Xc_test, yc_test, pc_test = align_by_path(Xc_test, yc_test, pc_test)

    if pg_train != pc_train or pg_test != pc_test:
        raise ValueError("Global and crop embeddings do not align on the same image paths.")
    if not np.array_equal(yg_train, yc_train) or not np.array_equal(yg_test, yc_test):
        raise ValueError("Global and crop embeddings do not align on the same labels.")

    if args.fusion_method == "avg_prob":
        log_message("training separate LogisticRegression heads for probability averaging")
        global_clf = train_logistic_regression(Xg_train, yg_train)
        crop_clf = train_logistic_regression(Xc_train, yc_train)
        global_prob = global_clf.predict_proba(Xg_test)[:, 1]
        crop_prob = crop_clf.predict_proba(Xc_test)[:, 1]
        y_prob = (global_prob + crop_prob) / 2.0
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_metrics(yg_test, y_pred, y_prob)
        notes = [
            "- fusion_method: `avg_prob`",
            f"- global_root: `{global_root}`",
            f"- crop_root: `{crop_root}`",
            "- test probability = average(global_prob, crop_prob)",
        ]
    else:
        log_message("building concatenated train/test embeddings")
        X_train = np.concatenate([Xg_train, Xc_train], axis=1)
        X_test = np.concatenate([Xg_test, Xc_test], axis=1)
        if args.l2_normalize_concat:
            X_train = l2_normalize_rows(X_train)
            X_test = l2_normalize_rows(X_test)

        log_message("training LogisticRegression on concatenated embeddings")
        clf = train_logistic_regression(X_train, yg_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_metrics(yg_test, y_pred, y_prob)
        save_npz(output_root / "embeddings_train.npz", X_train, yg_train, pg_train)
        save_npz(output_root / "embeddings_test.npz", X_test, yg_test, pg_test)
        notes = [
            "- fusion_method: `concat`",
            f"- global_root: `{global_root}`",
            f"- crop_root: `{crop_root}`",
            f"- concatenated_dim: {X_train.shape[1]}",
            f"- l2_normalize_concat: {args.l2_normalize_concat}",
        ]

    prediction_rows = build_prediction_rows(pg_test, yg_test, y_pred, y_prob)
    save_json(output_root / "metrics.json", metrics)
    save_csv(output_root / "test_predictions.csv", prediction_rows)
    save_text(output_root / "report.md", format_report("Embedding Fusion Report", metrics, notes))

    summary = ", ".join(
        f"{key}={metrics[key]:.4f}"
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
        if key in metrics
    )
    log_message(f"fusion metrics: {summary}")
    log_message(f"fusion artifacts: {output_root}")


if __name__ == "__main__":
    main()
