"""Step 5 — Threshold optimisation with Optuna.

Runs a TPE study over the (t_low, t_high) decision boundaries of the
calibrated fused classifier with the constraint ``t_low < t_high``.

Objective
---------
F-β on validation with β=2. This upweights recall — i.e. penalises false
negatives (missed scams predicted as low-risk) more heavily than false
positives — which matches the application requirement to not let scams
through.

A sample labelled 1 (phishing) is a hit whenever the predicted verdict is
``medium`` or ``high``. A sample labelled 0 (legit) is a hit whenever the
predicted verdict is ``low``. The F-β score is computed on the
``phishing-or-medium-flag`` prediction vs the ground-truth label.

Validation data
---------------
To keep TEST untouched, we get out-of-fold calibrated probabilities on
TRAIN via ``cross_val_predict`` with ``CalibratedClassifierCV``. Optuna
tunes thresholds on those OOF probs, and we finally report the verdict
metrics on TEST using the frozen thresholds.

Output
------
artifacts/ensemble/thresholds.json
artifacts/ensemble/threshold_tuning_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_bayesian_ensemble import FEATURE_COLUMNS, load_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna threshold tuning for the calibrated ensemble.")
    p.add_argument("--input-dir", default="artifacts/ensemble")
    p.add_argument("--output-dir", default="artifacts/ensemble")
    p.add_argument("--features", default=",".join(FEATURE_COLUMNS))
    p.add_argument("--n-trials", type=int, default=200,
                   help="Optuna trials (100–300 is the usable range).")
    p.add_argument("--beta", type=float, default=2.0,
                   help="F-β recall weight (β=2 upweights false negatives).")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="CV folds for OOF calibrated probabilities on TRAIN.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def verdicts_from_probs(probs: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """Return 0 = low, 1 = medium, 2 = high."""
    verdicts = np.zeros_like(probs, dtype=np.int64)
    verdicts[probs >= t_low] = 1
    verdicts[probs >= t_high] = 2
    return verdicts


def binary_flag_from_verdicts(verdicts: np.ndarray) -> np.ndarray:
    """Treat medium + high as positive flags (scam)."""
    return (verdicts >= 1).astype(np.int64)


def metrics_at_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    t_low: float,
    t_high: float,
    beta: float,
) -> dict:
    verdicts = verdicts_from_probs(probs, t_low, t_high)
    y_pred = binary_flag_from_verdicts(verdicts)
    n = len(y_true)
    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "fbeta": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "pct_low": float((verdicts == 0).sum() / n),
        "pct_medium": float((verdicts == 1).sum() / n),
        "pct_high": float((verdicts == 2).sum() / n),
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cal_path = input_dir / "calibrated_ensemble.joblib"
    if not cal_path.exists():
        print(f"ERROR: {cal_path} not found. Run calibrate_bayesian_ensemble.py first.")
        return 1

    bundle = joblib.load(cal_path)
    feature_names = bundle["feature_names"]
    best_C = float(bundle["base_C"])
    method = bundle["method"]
    calibrated_model = bundle["model"]
    print(f"Calibrated model: method={method}, base C={best_C:.6f}, features={feature_names}")

    train_csv = input_dir / "train_scores.csv"
    test_csv = input_dir / "test_scores.csv"
    X_train, y_train, _ = load_scores(train_csv, feature_names)
    X_test, y_test, _ = load_scores(test_csv, feature_names)
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # OOF calibrated probabilities on TRAIN.
    # We rebuild a CalibratedClassifierCV with the same hyperparameters so
    # cross_val_predict yields OOF calibrated probs without leakage into TEST.
    cv_splitter = StratifiedKFold(
        n_splits=min(args.cv_folds, len(y_train)),
        shuffle=True,
        random_state=args.seed,
    )
    base_factory = lambda: CalibratedClassifierCV(  # noqa: E731
        LogisticRegression(
            C=best_C, class_weight="balanced", max_iter=5000, random_state=args.seed,
        ),
        method=method,
        cv=StratifiedKFold(
            n_splits=min(args.cv_folds, len(y_train)),
            shuffle=True,
            random_state=args.seed,
        ),
    )
    print(f"Computing OOF calibrated probs on TRAIN ({args.cv_folds} folds)...")
    oof_probs = cross_val_predict(
        base_factory(),
        X_train,
        y_train,
        cv=cv_splitter,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    # ── Optuna study ────────────────────────────────────────────────
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        t_low = trial.suggest_float("t_low", 0.0, 1.0)
        t_high = trial.suggest_float("t_high", 0.0, 1.0)
        if t_low >= t_high:
            # Enforce the strict ordering constraint via a hard prune
            raise optuna.exceptions.TrialPruned()
        preds = binary_flag_from_verdicts(verdicts_from_probs(oof_probs, t_low, t_high))
        return fbeta_score(y_train, preds, beta=args.beta, zero_division=0)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    t_low = float(study.best_params["t_low"])
    t_high = float(study.best_params["t_high"])
    best_fbeta = float(study.best_value)
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f"\nOptuna best: t_low={t_low:.4f}, t_high={t_high:.4f}, "
          f"F-β({args.beta})={best_fbeta:.4f}")
    print(f"  ({len(study.trials)} trials, {n_pruned} pruned by constraint)")

    # ── Evaluate on TRAIN (OOF) and TEST at chosen thresholds ───────
    train_metrics = metrics_at_thresholds(y_train, oof_probs, t_low, t_high, args.beta)
    test_probs = calibrated_model.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at_thresholds(y_test, test_probs, t_low, t_high, args.beta)

    # Add macro F1 / AUC on test for the paper results table
    test_pred = binary_flag_from_verdicts(verdicts_from_probs(test_probs, t_low, t_high))
    if len(np.unique(y_test)) > 1:
        test_metrics["roc_auc"] = float(roc_auc_score(y_test, test_probs))
    test_metrics["macro_f1"] = float(fbeta_score(
        y_test, test_pred, beta=1.0, average="macro", zero_division=0,
    ))

    print("\n" + "=" * 72)
    print("Frozen thresholds @ test:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<12}: {v:.4f}")
    print("=" * 72)

    # ── Save ────────────────────────────────────────────────────────
    thresholds_path = output_dir / "thresholds.json"
    thresholds_path.write_text(json.dumps({
        "t_low": t_low,
        "t_high": t_high,
        "beta": args.beta,
        "calibration_method": method,
    }, indent=2), encoding="utf-8")
    print(f"Saved thresholds to: {thresholds_path}")

    report = {
        "n_trials": args.n_trials,
        "pruned_trials": n_pruned,
        "beta": args.beta,
        "calibration_method": method,
        "features": feature_names,
        "oof_cv_folds": args.cv_folds,
        "best_params": {"t_low": t_low, "t_high": t_high},
        "objective_value_oof": best_fbeta,
        "train_oof_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    report_path = output_dir / "threshold_tuning_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
