import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import MODEL_FILE_BY_NAME, REPORT_DIR

MODEL_CHOICES = ("logreg", "lightgbm", "xgboost")


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["paths"]


def get_model_path(model_name):
    return MODEL_FILE_BY_NAME[model_name]


def get_report_path(split_name, model_name):
    return REPORT_DIR / f"{split_name}_metrics_{model_name}.txt"


def _compute_xgb_scale_pos_weight(labels):
    labels = np.asarray(labels)
    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)
    if positives == 0 or negatives == 0:
        return 1.0
    return negatives / positives


def build_classifier(model_name, y_train):
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
        )

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "lightgbm is not installed. Install it with `pip install lightgbm` "
                "or reinstall from `requirements.txt`."
            ) from exc

        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it with `pip install xgboost` "
                "or reinstall from `requirements.txt`."
            ) from exc

        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=_compute_xgb_scale_pos_weight(y_train),
        )

    raise ValueError(f"Unsupported model: {model_name}")


def predict_positive_scores(clf, features):
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(features)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return probs[:, 1]
        return probs.ravel()

    if hasattr(clf, "decision_function"):
        raw_scores = clf.decision_function(features)
        return 1.0 / (1.0 + np.exp(-raw_scores))

    return None


def compute_metrics(y_true, preds, probs=None):
    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
    }

    unique_labels = np.unique(y_true)
    if probs is not None and unique_labels.size > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, probs)

    return metrics


def format_report(y_true, preds, probs=None, include_confusion=False):
    metrics = compute_metrics(y_true, preds, probs=probs)
    lines = [f"{key}: {value:.6f}" for key, value in metrics.items()]
    lines.append("")
    lines.append(classification_report(y_true, preds, digits=6, zero_division=0))

    if include_confusion:
        lines.append("Confusion matrix:")
        lines.append(str(confusion_matrix(y_true, preds)))

    return metrics, "\n".join(lines)
