"""Train a Bayesian meta-classifier that fuses the two sub-model scores.

Reads the CSV outputs of ``extract_ensemble_scores.py`` (one row per image
with scores from Model A and Model B) and trains two ensemble variants:

1. **Naive Bayes with KDE** — estimates class-conditional score densities
   P(s | phishing) and P(s | legit) via Kernel Density Estimation, then
   combines via log-likelihood ratios.  Pure Bayesian, highly interpretable.

2. **L2-Regularised Logistic Regression** (= Bayesian MAP estimation) —
   the L2 penalty is equivalent to a Gaussian prior N(0, 1/C) on the
   weights.  Regularisation strength C is tuned via stratified 5-fold CV.

Sub-models being combined:
    - Model A: fuse_siglip_DINO          → ``fuse_siglip_dino_prob``
    - Model B: ocr_ollama_distilbert     → ``ocr_distilbert_combined``
                                           (``_heuristic`` and ``_model``
                                           are also available as auxiliary
                                           features.)

Both models are trained on **train** scores and evaluated on **test** scores.

Output
------
artifacts/ensemble/naive_bayes_ensemble.joblib
artifacts/ensemble/logreg_ensemble.joblib
artifacts/ensemble/ensemble_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KernelDensity


# ── Feature definitions ─────────────────────────────────────────────

IMAGE_FEATURE = "fuse_siglip_dino_prob"             # Model A
TEXT_FEATURE = "ocr_distilbert_combined"            # Model B (primary)

FEATURE_COLUMNS = [
    IMAGE_FEATURE,
    TEXT_FEATURE,
    "ocr_distilbert_heuristic",
    "ocr_distilbert_model",
]

IMPUTE_VALUE = 0.5  # neutral score for missing text values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Bayesian ensemble meta-classifier.")
    p.add_argument("--input-dir", default="artifacts/ensemble",
                    help="Dir with train_scores.csv, test_scores.csv")
    p.add_argument("--output-dir", default="artifacts/ensemble",
                    help="Dir for trained models and reports")
    p.add_argument("--features", default=",".join(FEATURE_COLUMNS),
                    help="Comma-separated feature columns to use.")
    p.add_argument("--cv-folds", type=int, default=5,
                    help="Number of CV folds for hyperparameter tuning.")
    return p.parse_args()


# ── Data loading ────────────────────────────────────────────────────

def load_scores(csv_path: Path, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load CSV and return (X, y, filenames). Missing values are imputed."""
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    n = len(rows)
    X = np.full((n, len(feature_names)), IMPUTE_VALUE, dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    filenames: list[str] = []

    for i, row in enumerate(rows):
        y[i] = int(row["label"])
        filenames.append(row.get("filename", row.get("image_path", "")))
        for j, col in enumerate(feature_names):
            val = row.get(col, "").strip()
            if val:
                X[i, j] = float(val)

    return X, y, filenames


def load_score_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def feature_presence_mask(rows: list[dict], feature_name: str) -> np.ndarray:
    return np.array([bool(str(row.get(feature_name, "")).strip()) for row in rows], dtype=bool)


def best_threshold_by_accuracy(train_probs: np.ndarray, y_train: np.ndarray) -> float:
    if train_probs.size == 0:
        return 0.5

    candidates = np.unique(train_probs)
    candidates = np.concatenate(([0.0], candidates, [0.5, 1.0]))
    best_threshold = 0.5
    best_accuracy = -1.0

    for threshold in np.unique(candidates):
        preds = (train_probs >= threshold).astype(int)
        acc = accuracy_score(y_train, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)

    return best_threshold


def weighted_average_with_fallback(
    X: np.ndarray,
    img_idx: int,
    txt_idx: int,
    text_present_mask: np.ndarray,
    image_weight: float = 0.35,
) -> np.ndarray:
    probs = weighted_average_baseline(X, img_idx, txt_idx, image_weight=image_weight)
    probs = probs.copy()
    probs[~text_present_mask] = X[~text_present_mask, img_idx]
    return probs


# ── Metrics ─────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return m


# ── Method 1: Naive Bayes with KDE ─────────────────────────────────

@dataclass
class NaiveBayesKDE:
    """Naive Bayes classifier using Kernel Density Estimation.

    For each feature, fits P(feature | class=0) and P(feature | class=1)
    using KDE on the training data.  At inference, computes:

        log P(phishing | features) / P(legit | features)
          = log(prior_ratio)
            + sum_i [ log P(feature_i | phishing) - log P(feature_i | legit) ]

    and converts to a probability via the sigmoid function.
    """
    bandwidth: float = 0.05
    kdes_phishing: list = field(default_factory=list)
    kdes_legit: list = field(default_factory=list)
    log_prior_ratio: float = 0.0
    n_features: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesKDE":
        self.n_features = X.shape[1]
        X_phish = X[y == 1]
        X_legit = X[y == 0]

        n_phish = max(len(X_phish), 1)
        n_legit = max(len(X_legit), 1)
        self.log_prior_ratio = float(np.log(n_phish / n_legit))

        self.kdes_phishing = []
        self.kdes_legit = []
        for j in range(self.n_features):
            kde_p = KernelDensity(bandwidth=self.bandwidth, kernel="gaussian")
            kde_l = KernelDensity(bandwidth=self.bandwidth, kernel="gaussian")
            kde_p.fit(X_phish[:, j:j+1])
            kde_l.fit(X_legit[:, j:j+1])
            self.kdes_phishing.append(kde_p)
            self.kdes_legit.append(kde_l)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 2) array of [P(legit), P(phishing)]."""
        log_odds = np.full(X.shape[0], self.log_prior_ratio)
        for j in range(self.n_features):
            log_p = self.kdes_phishing[j].score_samples(X[:, j:j+1])
            log_l = self.kdes_legit[j].score_samples(X[:, j:j+1])
            log_odds += log_p - log_l

        prob_phishing = 1.0 / (1.0 + np.exp(-log_odds))
        return np.column_stack([1.0 - prob_phishing, prob_phishing])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


def tune_kde_bandwidth(X: np.ndarray, y: np.ndarray, cv_folds: int) -> float:
    """Grid search over bandwidths using CV accuracy."""
    bandwidths = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
    best_bw = 0.05
    best_acc = -1.0

    skf = StratifiedKFold(n_splits=min(cv_folds, len(y)), shuffle=True, random_state=42)

    for bw in bandwidths:
        accs = []
        for train_idx, val_idx in skf.split(X, y):
            nb = NaiveBayesKDE(bandwidth=bw)
            nb.fit(X[train_idx], y[train_idx])
            preds = nb.predict(X[val_idx])
            accs.append(accuracy_score(y[val_idx], preds))
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_bw = bw

    print(f"  Best KDE bandwidth: {best_bw} (CV accuracy: {best_acc:.4f})")
    return best_bw


# ── Method 2: L2 Logistic Regression (Bayesian MAP) ────────────────

def tune_logreg(X: np.ndarray, y: np.ndarray, cv_folds: int) -> tuple[float, float]:
    """Grid search over C (= 1/lambda) using CV accuracy.

    L2 regularisation is equivalent to a Gaussian prior N(0, 1/C) on weights.
    Smaller C = stronger prior = more regularisation.
    """
    C_values = np.logspace(-3, 3, 25)
    best_C = 1.0
    best_acc = -1.0

    skf = StratifiedKFold(n_splits=min(cv_folds, len(y)), shuffle=True, random_state=42)

    for C in C_values:
        accs = []
        for train_idx, val_idx in skf.split(X, y):
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=42)
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[val_idx])
            accs.append(accuracy_score(y[val_idx], preds))
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_C = C

    print(f"  Best C: {best_C:.6f} (CV accuracy: {best_acc:.4f})")
    return best_C, best_acc


# ── Baselines ───────────────────────────────────────────────────────

def weighted_average_baseline(X: np.ndarray, img_idx: int, txt_idx: int,
                               image_weight: float = 0.35) -> np.ndarray:
    """Replicate the original risk_fusion_service.py weighted average."""
    return image_weight * X[:, img_idx] + (1.0 - image_weight) * X[:, txt_idx]


# ── Main ────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = [f.strip() for f in args.features.split(",") if f.strip()]
    print(f"Features: {feature_names}")

    train_csv = input_dir / "train_scores.csv"
    test_csv = input_dir / "test_scores.csv"
    for p in [train_csv, test_csv]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run extract_ensemble_scores.py first.")
            return 1

    train_rows = load_score_rows(train_csv)
    test_rows = load_score_rows(test_csv)
    X_train, y_train, _ = load_scores(train_csv, feature_names)
    X_test, y_test, _ = load_scores(test_csv, feature_names)
    train_text_present = feature_presence_mask(train_rows, TEXT_FEATURE)
    test_text_present = feature_presence_mask(test_rows, TEXT_FEATURE)
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"Train class balance: {np.sum(y_train==0)} legit, {np.sum(y_train==1)} phishing")
    print(f"Test class balance: {np.sum(y_test==0)} legit, {np.sum(y_test==1)} phishing")
    print(
        f"Text-present rows: train={int(train_text_present.sum())}/{len(train_text_present)}, "
        f"test={int(test_text_present.sum())}/{len(test_text_present)}"
    )

    report: dict = {"features": feature_names, "results": {}}

    img_idx = feature_names.index(IMAGE_FEATURE)
    txt_idx = feature_names.index(TEXT_FEATURE)

    # ── Baseline 1: fuse_siglip_DINO alone ──────────────────────────
    probs = X_test[:, img_idx]
    preds = (probs >= 0.5).astype(int)
    m = compute_metrics(y_test, preds, probs)
    report["results"]["fuse_siglip_dino_alone"] = m
    print(f"\n[Baseline] fuse_siglip_DINO alone: acc={m['accuracy']:.4f} f1={m['f1']:.4f}")

    # ── Baseline 2: ocr_ollama_distilbert alone ─────────────────────
    if train_text_present.any() and test_text_present.any():
        train_text_probs = X_train[train_text_present, txt_idx]
        train_text_labels = y_train[train_text_present]
        test_text_probs = X_test[test_text_present, txt_idx]
        test_text_labels = y_test[test_text_present]
        text_threshold = best_threshold_by_accuracy(train_text_probs, train_text_labels)
        preds = (test_text_probs >= text_threshold).astype(int)
        m = compute_metrics(test_text_labels, preds, test_text_probs)
        m["threshold"] = float(text_threshold)
        m["rows_used_train"] = int(train_text_present.sum())
        m["rows_used_test"] = int(test_text_present.sum())
        report["results"]["ocr_ollama_distilbert_alone"] = m
        print(
            f"[Baseline] ocr_ollama_distilbert alone: acc={m['accuracy']:.4f} "
            f"f1={m['f1']:.4f} threshold={text_threshold:.4f} "
            f"(rows used: train={m['rows_used_train']}, test={m['rows_used_test']})"
        )
    else:
        m = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "threshold": None,
            "rows_used_train": int(train_text_present.sum()),
            "rows_used_test": int(test_text_present.sum()),
        }
        report["results"]["ocr_ollama_distilbert_alone"] = m
        print("[Baseline] ocr_ollama_distilbert alone: no text-present rows available")

    # ── Baseline 3: Weighted average (35/65) ────────────────────────
    wa_probs = weighted_average_with_fallback(
        X_test,
        img_idx,
        txt_idx,
        test_text_present,
        image_weight=0.35,
    )
    preds = (wa_probs >= 0.5).astype(int)
    m = compute_metrics(y_test, preds, wa_probs)
    report["results"]["weighted_average_35_65"] = m
    print(f"[Baseline] Weighted avg (35/65): acc={m['accuracy']:.4f} f1={m['f1']:.4f}")

    # ── Method 1: Naive Bayes KDE ───────────────────────────────────
    print("\n--- Naive Bayes with KDE ---")
    best_bw = tune_kde_bandwidth(X_train, y_train, args.cv_folds)
    nb = NaiveBayesKDE(bandwidth=best_bw)
    nb.fit(X_train, y_train)

    nb_probs_test = nb.predict_proba(X_test)[:, 1]
    nb_preds_test = nb.predict(X_test)
    m = compute_metrics(y_test, nb_preds_test, nb_probs_test)
    report["results"]["naive_bayes_kde"] = {**m, "bandwidth": best_bw}
    print(f"[Naive Bayes KDE] acc={m['accuracy']:.4f} f1={m['f1']:.4f}")

    joblib.dump(nb, output_dir / "naive_bayes_ensemble.joblib")

    # ── Method 2: L2 Logistic Regression (Bayesian MAP) ─────────────
    print("\n--- L2 Logistic Regression (Bayesian MAP) ---")
    best_C, cv_acc = tune_logreg(X_train, y_train, args.cv_folds)
    lr = LogisticRegression(C=best_C, class_weight="balanced", max_iter=5000, random_state=42)
    lr.fit(X_train, y_train)

    lr_probs_test = lr.predict_proba(X_test)[:, 1]
    lr_preds_test = lr.predict(X_test)
    m = compute_metrics(y_test, lr_preds_test, lr_probs_test)
    report["results"]["logreg_bayesian_map"] = {
        **m,
        "C": float(best_C),
        "cv_accuracy": float(cv_acc),
        "coefficients": {name: float(w) for name, w in zip(feature_names, lr.coef_[0])},
        "intercept": float(lr.intercept_[0]),
    }
    print(f"[LogReg Bayesian MAP] acc={m['accuracy']:.4f} f1={m['f1']:.4f}")
    print(f"  Learned weights:")
    for name, w in zip(feature_names, lr.coef_[0]):
        print(f"    {name}: {w:.4f}")
    print(f"    intercept: {lr.intercept_[0]:.4f}")

    joblib.dump(lr, output_dir / "logreg_ensemble.joblib")

    # ── Summary table ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Method':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 70)
    for method, metrics in report["results"].items():
        label = method.replace("_", " ").title()[:29]
        auc = metrics.get("roc_auc", 0)
        print(
            f"{label:<30} "
            f"{metrics['accuracy']:>7.4f} "
            f"{metrics['precision']:>7.4f} "
            f"{metrics['recall']:>7.4f} "
            f"{metrics['f1']:>7.4f} "
            f"{auc:>7.4f}"
        )
    print("=" * 70)

    report_path = output_dir / "ensemble_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
