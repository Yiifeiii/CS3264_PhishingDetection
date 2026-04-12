import argparse

import joblib

from classifier_utils import (
    MODEL_CHOICES,
    format_report,
    get_model_path,
    get_report_path,
    load_npz,
    predict_positive_scores,
)
from config import TEST_EMBED_FILE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[*MODEL_CHOICES, "all"],
        default="logreg",
        help="Classifier checkpoint to evaluate.",
    )
    return parser.parse_args()


def evaluate_one_model(model_name, X_test, y_test):
    clf = joblib.load(get_model_path(model_name))

    preds = clf.predict(X_test)
    probs = predict_positive_scores(clf, X_test)
    metrics, report_text = format_report(
        y_test,
        preds,
        probs=probs,
        include_confusion=True,
    )

    print(f"Test metrics for {model_name}:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nDetailed report:")
    print(report_text)

    with open(get_report_path("test", model_name), "w") as f:
        f.write(report_text)

    return metrics


def print_summary(results):
    ordered_metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc")
    header = ["model", *ordered_metric_names]
    col_widths = {name: len(name) for name in header}

    for model_name, metrics in results.items():
        col_widths["model"] = max(col_widths["model"], len(model_name))
        for metric_name in ordered_metric_names:
            value = metrics.get(metric_name, float("nan"))
            col_widths[metric_name] = max(
                col_widths[metric_name],
                len(f"{value:.4f}"),
            )

    print("\nSummary:")
    print(
        "  ".join(
            name.ljust(col_widths[name]) for name in header
        )
    )
    for model_name, metrics in sorted(
        results.items(),
        key=lambda item: (item[1].get("f1", float("-inf")), item[1].get("roc_auc", float("-inf"))),
        reverse=True,
    ):
        row = [model_name.ljust(col_widths["model"])]
        for metric_name in ordered_metric_names:
            row.append(f"{metrics.get(metric_name, float('nan')):.4f}".ljust(col_widths[metric_name]))
        print("  ".join(row))


def main():
    args = parse_args()
    X_test, y_test, _ = load_npz(TEST_EMBED_FILE)

    selected_models = MODEL_CHOICES if args.model == "all" else (args.model,)
    results = {}

    for idx, model_name in enumerate(selected_models):
        if idx > 0:
            print("\n" + "=" * 80)
        results[model_name] = evaluate_one_model(model_name, X_test, y_test)

    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
