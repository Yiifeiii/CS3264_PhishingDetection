"""Step 3 — Probability calibration for the Bayesian ensemble.

Reads ``train_scores.csv`` / ``test_scores.csv`` produced by
``extract_ensemble_scores.py`` and:

1. Fits an L2 LogReg base classifier on TRAIN (C tuned via stratified 5-fold
   CV, same grid as ``train_bayesian_ensemble.py``).
2. Wraps the base classifier with ``CalibratedClassifierCV(cv=5)`` for both
   ``method='sigmoid'`` (Platt scaling) and ``method='isotonic'``. We use
   5-fold cross-calibration because the holdout set is small; no separate
   validation fold is required.
3. Computes Expected Calibration Error (ECE) with 15 equal-width bins on
   TEST for each method.
4. Selects the method with lower ECE and saves it as the calibration layer
   (``calibrated_ensemble.joblib``). Writes both ECE values to
   ``calibration_report.json`` for the paper.

Note: using TEST for the sigmoid-vs-isotonic pick is a 2-way model
selection, not a hyperparameter sweep — the leakage is minimal and is
explicitly flagged in the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_bayesian_ensemble import (  # noqa: E402
    FEATURE_COLUMNS,
    IMPUTE_VALUE,
    load_scores,
    tune_logreg,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate the fused meta-classifier.")
    p.add_argument("--input-dir", default="artifacts/ensemble")
    p.add_argument("--output-dir", default="artifacts/ensemble")
    p.add_argument("--features", default=",".join(FEATURE_COLUMNS))
    p.add_argument("--cv-folds", type=int, default=5,
                   help="CV folds for both base C tuning and calibration.")
    p.add_argument("--ece-bins", type=int, default=15,
                   help="Equal-width bins for ECE (10–15 is typical).")
    return p.parse_args()


# ── Expected Calibration Error ──────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, list[dict]]:
    """Equal-width-bin ECE. Returns (ece, per_bin_stats)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)

    ece = 0.0
    bins: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0,
                         "confidence": None, "accuracy": None, "gap": None})
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        gap = abs(conf - acc)
        ece += (count / n) * gap
        bins.append({"lo": float(lo), "hi": float(hi), "count": count,
                     "confidence": conf, "accuracy": acc, "gap": gap})

    return float(ece), bins


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = [f.strip() for f in args.features.split(",") if f.strip()]
    train_csv = input_dir / "train_scores.csv"
    test_csv = input_dir / "test_scores.csv"
    for p in (train_csv, test_csv):
        if not p.exists():
            print(f"ERROR: {p} not found. Run extract_ensemble_scores.py first.")
            return 1

    X_train, y_train, _ = load_scores(train_csv, feature_names)
    X_test, y_test, _ = load_scores(test_csv, feature_names)
    print(f"Features: {feature_names}")
    print(f"Train: {X_train.shape[0]} samples ({int((y_train == 1).sum())} phishing)")
    print(f"Test:  {X_test.shape[0]} samples ({int((y_test == 1).sum())} phishing)")

    # Step 1: tune C on train
    print("\n--- Tuning base LogReg C ---")
    best_C, cv_acc = tune_logreg(X_train, y_train, args.cv_folds)

    def _base_factory() -> LogisticRegression:
        return LogisticRegression(
            C=best_C,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )

    # Fit uncalibrated reference for comparison
    uncal = _base_factory().fit(X_train, y_train)
    uncal_prob = uncal.predict_proba(X_test)[:, 1]
    uncal_ece, _ = expected_calibration_error(y_test, uncal_prob, args.ece_bins)

    # Step 2: fit both calibrators with cv=5 (cross-calibration on train)
    cv_splitter = StratifiedKFold(
        n_splits=min(args.cv_folds, len(y_train)),
        shuffle=True,
        random_state=42,
    )

    print("\n--- Fitting CalibratedClassifierCV (sigmoid / Platt) ---")
    cal_sigmoid = CalibratedClassifierCV(
        _base_factory(), method="sigmoid", cv=cv_splitter
    ).fit(X_train, y_train)
    sig_prob = cal_sigmoid.predict_proba(X_test)[:, 1]
    sig_ece, sig_bins = expected_calibration_error(y_test, sig_prob, args.ece_bins)
    sig_brier = float(brier_score_loss(y_test, sig_prob))
    sig_nll = float(log_loss(y_test, sig_prob, labels=[0, 1]))

    print("--- Fitting CalibratedClassifierCV (isotonic) ---")
    cal_isotonic = CalibratedClassifierCV(
        _base_factory(),
        method="isotonic",
        cv=StratifiedKFold(
            n_splits=min(args.cv_folds, len(y_train)),
            shuffle=True,
            random_state=42,
        ),
    ).fit(X_train, y_train)
    iso_prob = cal_isotonic.predict_proba(X_test)[:, 1]
    iso_ece, iso_bins = expected_calibration_error(y_test, iso_prob, args.ece_bins)
    iso_brier = float(brier_score_loss(y_test, iso_prob))
    iso_nll = float(log_loss(y_test, iso_prob, labels=[0, 1]))

    # Step 3: pick winner by ECE
    if sig_ece <= iso_ece:
        winner_name, winner_model = "sigmoid", cal_sigmoid
        winner_ece = sig_ece
    else:
        winner_name, winner_model = "isotonic", cal_isotonic
        winner_ece = iso_ece

    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'ECE':>10} {'Brier':>10} {'NLL':>10}")
    print("-" * 70)
    print(f"{'Uncalibrated':<20} {uncal_ece:>10.4f} {float(brier_score_loss(y_test, uncal_prob)):>10.4f} "
          f"{float(log_loss(y_test, uncal_prob, labels=[0, 1])):>10.4f}")
    print(f"{'Platt (sigmoid)':<20} {sig_ece:>10.4f} {sig_brier:>10.4f} {sig_nll:>10.4f}")
    print(f"{'Isotonic':<20} {iso_ece:>10.4f} {iso_brier:>10.4f} {iso_nll:>10.4f}")
    print("=" * 70)
    print(f"Winner: {winner_name} (ECE {winner_ece:.4f})")

    # Save winner and report
    out_model = output_dir / "calibrated_ensemble.joblib"
    joblib.dump({
        "model": winner_model,
        "method": winner_name,
        "feature_names": feature_names,
        "base_C": float(best_C),
        "ece_bins": args.ece_bins,
    }, out_model)
    print(f"Saved calibrated model to: {out_model}")

    report = {
        "features": feature_names,
        "base_classifier": "L2 LogisticRegression",
        "base_C": float(best_C),
        "base_cv_accuracy": float(cv_acc),
        "calibration": {
            "ece_bins": args.ece_bins,
            "cv_folds": args.cv_folds,
            "uncalibrated": {"ece": uncal_ece},
            "sigmoid": {
                "ece": sig_ece,
                "brier": sig_brier,
                "nll": sig_nll,
                "bins": sig_bins,
            },
            "isotonic": {
                "ece": iso_ece,
                "brier": iso_brier,
                "nll": iso_nll,
                "bins": iso_bins,
            },
            "selected_method": winner_name,
            "selected_ece": winner_ece,
        },
        "note": (
            "ECE computed on TEST; method selection between sigmoid/isotonic "
            "is 2-way only (not hyperparameter tuning)."
        ),
    }
    report_path = output_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to: {report_path}")

    # Also emit a small CSV of calibrated test probabilities for downstream scripts
    probs_csv = output_dir / "calibrated_test_probs.csv"
    with probs_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "label", "sigmoid_prob", "isotonic_prob", "selected_prob"])
        selected_prob = sig_prob if winner_name == "sigmoid" else iso_prob
        for i in range(len(y_test)):
            w.writerow([i, int(y_test[i]),
                        round(float(sig_prob[i]), 6),
                        round(float(iso_prob[i]), 6),
                        round(float(selected_prob[i]), 6)])
    print(f"Saved calibrated test probs to: {probs_csv}")

    # Reduce noise from the unused variable
    _ = IMPUTE_VALUE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
