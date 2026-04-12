"""Comprehensive evaluation of the Bayesian ensemble vs baselines.

Loads the trained ensemble models from ``artifacts/ensemble/`` and
produces a detailed comparison report including:
- Per-method accuracy, precision, recall, F1, ROC-AUC
- Per-sample predictions CSV for error analysis
- Feature importance / learned weights interpretation

Usage:
    python scripts/evaluate_bayesian_ensemble.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_COLUMNS = [
    "siglip_logreg_prob",
    "text_combined_score",
    "text_heuristic_score",
    "text_preprocess_model_score",
]

IMPUTE_VALUE = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Bayesian ensemble models.")
    p.add_argument("--input-dir", default="artifacts/ensemble")
    p.add_argument("--output-dir", default="artifacts/ensemble")
    p.add_argument("--features", default=",".join(FEATURE_COLUMNS))
    p.add_argument("--bootstrap-n", type=int, default=1000,
                    help="Number of bootstrap iterations for confidence intervals.")
    return p.parse_args()


def load_scores(csv_path: Path, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    n = len(rows)
    X = np.full((n, len(feature_names)), IMPUTE_VALUE, dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    filenames = []

    for i, row in enumerate(rows):
        y[i] = int(row["label"])
        filenames.append(row["filename"])
        for j, col in enumerate(feature_names):
            val = row.get(col, "").strip()
            if val:
                X[i, j] = float(val)

    return X, y, filenames


def compute_metrics(y_true, y_pred, y_prob=None):
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return m


def bootstrap_ci(y_true, y_pred, y_prob, n_iter=1000, alpha=0.05):
    """Compute bootstrap 95% confidence intervals for accuracy and F1."""
    rng = np.random.RandomState(42)
    n = len(y_true)
    accs = []
    f1s = []
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
    return {
        "accuracy_ci": (float(np.percentile(accs, 100 * alpha / 2)),
                        float(np.percentile(accs, 100 * (1 - alpha / 2)))),
        "f1_ci": (float(np.percentile(f1s, 100 * alpha / 2)),
                  float(np.percentile(f1s, 100 * (1 - alpha / 2)))),
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = [f.strip() for f in args.features.split(",") if f.strip()]
    test_csv = input_dir / "test_scores.csv"
    if not test_csv.exists():
        print(f"ERROR: {test_csv} not found.")
        return 1

    X_test, y_test, filenames = load_scores(test_csv, feature_names)
    print(f"Test set: {len(y_test)} samples ({np.sum(y_test==0)} legit, {np.sum(y_test==1)} phishing)")

    siglip_idx = feature_names.index("siglip_logreg_prob")
    text_idx = feature_names.index("text_combined_score")

    # Collect all methods and their predictions
    methods: dict[str, dict] = {}

    # Baseline: SigLIP alone
    probs = X_test[:, siglip_idx]
    preds = (probs >= 0.5).astype(int)
    methods["SigLIP alone"] = {"preds": preds, "probs": probs}

    # Baseline: Text combined alone
    probs = X_test[:, text_idx]
    preds = (probs >= 0.5).astype(int)
    methods["Text combined alone"] = {"preds": preds, "probs": probs}

    # Baseline: Weighted average 35/65
    probs = 0.35 * X_test[:, siglip_idx] + 0.65 * X_test[:, text_idx]
    preds = (probs >= 0.5).astype(int)
    methods["Weighted avg (35/65)"] = {"preds": preds, "probs": probs}

    # Naive Bayes KDE
    nb_path = input_dir / "naive_bayes_ensemble.joblib"
    if nb_path.exists():
        nb = joblib.load(nb_path)
        probs = nb.predict_proba(X_test)[:, 1]
        preds = nb.predict(X_test)
        methods["Naive Bayes KDE"] = {"preds": preds, "probs": probs}

    # LogReg Bayesian MAP
    lr_path = input_dir / "logreg_ensemble.joblib"
    if lr_path.exists():
        lr = joblib.load(lr_path)
        probs = lr.predict_proba(X_test)[:, 1]
        preds = lr.predict(X_test)
        methods["LogReg Bayesian MAP"] = {"preds": preds, "probs": probs}

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Method':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}  {'Acc 95% CI':>18}")
    print("-" * 80)

    report = {}
    for name, data in methods.items():
        m = compute_metrics(y_test, data["preds"], data["probs"])
        ci = bootstrap_ci(y_test, data["preds"], data["probs"], n_iter=args.bootstrap_n)
        acc_ci = ci["accuracy_ci"]
        print(
            f"{name:<25} "
            f"{m['accuracy']:>7.4f} "
            f"{m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} "
            f"{m['f1']:>7.4f} "
            f"{m.get('roc_auc', 0):>7.4f}  "
            f"[{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]"
        )
        report[name] = {**m, **ci}

    print("=" * 80)

    # Per-sample predictions for error analysis
    per_sample_rows = []
    for i, fname in enumerate(filenames):
        row = {"filename": fname, "true_label": int(y_test[i])}
        for name, data in methods.items():
            col_prefix = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
            row[f"{col_prefix}_pred"] = int(data["preds"][i])
            row[f"{col_prefix}_prob"] = round(float(data["probs"][i]), 4)
        per_sample_rows.append(row)

    per_sample_path = output_dir / "per_sample_predictions.csv"
    if per_sample_rows:
        fieldnames = list(per_sample_rows[0].keys())
        with per_sample_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_sample_rows)
        print(f"\nPer-sample predictions: {per_sample_path}")

    # Confusion matrices
    print("\n--- Confusion Matrices ---")
    for name, data in methods.items():
        cm = confusion_matrix(y_test, data["preds"])
        print(f"\n{name}:")
        print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
        print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")

    # Feature ablation (for LogReg only)
    if lr_path.exists():
        lr = joblib.load(lr_path)
        print("\n--- Feature Ablation (LogReg) ---")
        print("Dropping one feature at a time to measure importance:\n")
        full_probs = lr.predict_proba(X_test)[:, 1]
        full_acc = accuracy_score(y_test, (full_probs >= 0.5).astype(int))

        for j, feat_name in enumerate(feature_names):
            X_ablated = X_test.copy()
            X_ablated[:, j] = IMPUTE_VALUE  # replace with neutral value
            ablated_probs = lr.predict_proba(X_ablated)[:, 1]
            ablated_acc = accuracy_score(y_test, (ablated_probs >= 0.5).astype(int))
            delta = full_acc - ablated_acc
            print(f"  Drop {feat_name:<35s}: acc={ablated_acc:.4f}  (delta={delta:+.4f})")

    # Save report
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
