import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib

from classifier_utils import (
    MODEL_CHOICES,
    build_classifier,
    format_report,
    get_model_path,
    get_report_path,
    load_npz,
)
from config import TRAIN_EMBED_FILE, VAL_EMBED_FILE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[*MODEL_CHOICES, "all"],
        default="logreg",
        help="Classifier to train on top of SigLIP embeddings.",
    )
    return parser.parse_args()


def train_one_model(model_name, X_train, y_train, X_val, y_val):
    clf = build_classifier(model_name, y_train)
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    val_probs = None
    if hasattr(clf, "predict_proba") or hasattr(clf, "decision_function"):
        from classifier_utils import predict_positive_scores

        val_probs = predict_positive_scores(clf, X_val)

    metrics, report_text = format_report(y_val, val_preds, probs=val_probs)

    print(f"\nValidation metrics for {model_name}:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification report:")
    print(report_text.split("\n\n", maxsplit=1)[1])

    model_path = get_model_path(model_name)
    joblib.dump(clf, model_path)
    print(f"Saved model to {model_path}")

    report_path = get_report_path("val", model_name)
    with open(report_path, "w") as f:
        f.write(report_text)

    return metrics


def main():
    args = parse_args()
    X_train, y_train, _ = load_npz(TRAIN_EMBED_FILE)
    X_val, y_val, _ = load_npz(VAL_EMBED_FILE)

    selected_models = MODEL_CHOICES if args.model == "all" else (args.model,)

    trained_any_model = False
    for model_name in selected_models:
        try:
            train_one_model(model_name, X_train, y_train, X_val, y_val)
            trained_any_model = True
        except ImportError as exc:
            print(f"Skipping {model_name}: {exc}")

    if not trained_any_model:
        raise RuntimeError(
            "No classifiers were trained. Install the requested optional dependencies "
            "or choose a supported available model."
        )


if __name__ == "__main__":
    main()
