import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


MODEL_NAMES = ("logreg", "lightgbm", "xgboost")
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")
REPORT_SPLITS = ("val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full SigLIP learning-curve experiment: create fixed-holdout splits, "
            "extract embeddings, train all classifier heads, evaluate them, and write a final report."
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
        help="Comma-separated total training sizes. Must be divisible by (#sources * 2 classes).",
    )
    parser.add_argument(
        "--val-per-bucket",
        type=int,
        default=None,
        help="Validation images per (source, class) bucket.",
    )
    parser.add_argument(
        "--val-total",
        type=int,
        default=None,
        help="Total validation images across all source/class buckets.",
    )
    parser.add_argument(
        "--test-per-bucket",
        type=int,
        default=None,
        help="Test images per (source, class) bucket.",
    )
    parser.add_argument(
        "--test-total",
        type=int,
        default=None,
        help="Total test images across all source/class buckets.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated subsets per training size.",
    )
    parser.add_argument(
        "--split-root",
        default="data/learning_curve",
        help="Directory where learning-curve splits will be created.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/learning_curve",
        help="Directory where per-run outputs and aggregate reports will be written.",
    )
    parser.add_argument(
        "--model-name",
        default="artifacts/siglip2-base-patch16-224",
        help="SigLIP model name or local artifact path.",
    )
    parser.add_argument(
        "--python-bin",
        default="./venv/bin/python",
        help="Python executable used to invoke the existing training scripts.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="SIGLIP_NUM_WORKERS override for embedding extraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split generation.",
    )
    parser.add_argument(
        "--clear-splits",
        action="store_true",
        help="Delete the split root before creating new learning-curve splits.",
    )
    parser.add_argument(
        "--clear-outputs",
        action="store_true",
        help="Delete the output root before running experiments.",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Reuse an existing split manifest instead of regenerating splits.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun extraction/training/evaluation even if outputs already exist.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def run_command(cmd, cwd: Path, env=None):
    rendered = " ".join(str(part) for part in cmd)
    print(f"\n>>> {rendered}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def create_splits(args, project_root: Path, split_root: Path):
    if args.clear_splits and split_root.exists():
        shutil.rmtree(split_root)

    cmd = [
        args.python_bin,
        "scripts/create_siglip_learning_curve_splits.py",
        "--output-root",
        str(split_root),
        "--train-sizes",
        args.train_sizes,
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
    ]

    if args.val_total is not None:
        cmd.extend(["--val-total", str(args.val_total)])
    elif args.val_per_bucket is not None:
        cmd.extend(["--val-per-bucket", str(args.val_per_bucket)])

    if args.test_total is not None:
        cmd.extend(["--test-total", str(args.test_total)])
    elif args.test_per_bucket is not None:
        cmd.extend(["--test-per-bucket", str(args.test_per_bucket)])

    if args.clear_splits:
        cmd.append("--clear-output")

    for source_root in args.source_root:
        cmd.extend(["--source-root", source_root])

    run_command(cmd, cwd=project_root)


def load_manifest(split_root: Path):
    manifest_path = split_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Learning-curve manifest not found at {manifest_path}. "
            "Create the splits first or remove --skip-split."
        )

    with open(manifest_path) as f:
        return json.load(f)


def report_path_for(output_dir: Path, split_name: str, model_name: str) -> Path:
    return output_dir / "reports" / f"{split_name}_metrics_{model_name}.txt"


def parse_report(report_path: Path):
    metrics = {}
    with open(report_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                break
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", maxsplit=1)
            key = key.strip()
            value = value.strip()
            if key in METRIC_NAMES:
                metrics[key] = float(value)
    return metrics


def maybe_run_experiment(args, project_root: Path, experiment_name: str, split_root: Path, output_root: Path):
    split_dir = split_root / experiment_name
    run_output_dir = output_root / experiment_name

    expected_reports = [
        report_path_for(run_output_dir, split_name, model_name)
        for split_name in REPORT_SPLITS
        for model_name in MODEL_NAMES
    ]
    all_reports_exist = all(path.exists() for path in expected_reports)

    if all_reports_exist and not args.force:
        print(f"\n>>> Skipping completed experiment {experiment_name}")
        return

    env = os.environ.copy()
    env["SIGLIP_DATA_DIR"] = str(split_dir)
    env["SIGLIP_OUTPUT_DIR"] = str(run_output_dir)
    env["SIGLIP_MODEL_NAME"] = args.model_name
    env["SIGLIP_NUM_WORKERS"] = str(args.num_workers)

    run_command(
        [args.python_bin, "siglip/extract_embeddings.py"],
        cwd=project_root,
        env=env,
    )
    run_command(
        [args.python_bin, "siglip/train_classifier.py", "--model", "all"],
        cwd=project_root,
        env=env,
    )
    run_command(
        [args.python_bin, "siglip/evaluate.py", "--model", "all"],
        cwd=project_root,
        env=env,
    )


def collect_rows(manifest, output_root: Path):
    count_lookup = {
        experiment["name"]: experiment["counts"]
        for experiment in manifest["experiments"]
    }

    rows = []
    for experiment in manifest["experiments"]:
        name = experiment["name"]
        total_train_size = experiment["total_train_size"]
        repeat = experiment["repeat"]
        run_output_dir = output_root / name
        counts = count_lookup[name]

        for split_name in REPORT_SPLITS:
            for model_name in MODEL_NAMES:
                report_path = report_path_for(run_output_dir, split_name, model_name)
                if not report_path.exists():
                    raise FileNotFoundError(f"Missing expected report: {report_path}")

                metrics = parse_report(report_path)
                row = {
                    "experiment": name,
                    "split": split_name,
                    "model": model_name,
                    "train_size_total": total_train_size,
                    "repeat": repeat,
                    "train_real_count": counts["train"]["real"],
                    "train_fake_count": counts["train"]["fake"],
                    "val_real_count": counts["val"]["real"],
                    "val_fake_count": counts["val"]["fake"],
                    "test_real_count": counts["test"]["real"],
                    "test_fake_count": counts["test"]["fake"],
                }
                row.update(metrics)
                rows.append(row)
    return rows


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["split"], row["model"], row["train_size_total"])
        grouped[key].append(row)

    summary_rows = []
    for (split_name, model_name, train_size_total), group_rows in sorted(grouped.items()):
        summary = {
            "split": split_name,
            "model": model_name,
            "train_size_total": train_size_total,
            "repeats": len(group_rows),
        }
        for metric_name in METRIC_NAMES:
            values = [row[metric_name] for row in group_rows if metric_name in row]
            summary[f"{metric_name}_mean"] = statistics.mean(values)
            summary[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    return summary_rows


def find_plateau_row(summary_rows, split_name: str, model_name: str, metric_name: str, ratio: float = 0.95):
    candidates = sorted(
        [
            row for row in summary_rows
            if row["split"] == split_name and row["model"] == model_name
        ],
        key=lambda row: row["train_size_total"],
    )
    peak = max(row[f"{metric_name}_mean"] for row in candidates)
    threshold = peak * ratio
    for row in candidates:
        if row[f"{metric_name}_mean"] >= threshold:
            return row
    return candidates[-1]


def format_metric(mean_value: float, std_value: float):
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def paired_test_row(summary_rows, model_name: str, train_size_total: int):
    for row in summary_rows:
        if (
            row["split"] == "test"
            and row["model"] == model_name
            and row["train_size_total"] == train_size_total
        ):
            return row
    raise KeyError(f"Missing test summary for {model_name} at train size {train_size_total}")


def write_markdown_report(path: Path, manifest, summary_rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    sources = ", ".join(Path(source).name for source in manifest["sources"])
    train_sizes = ", ".join(str(size) for size in manifest["train_sizes_total"])

    best_overall = max(
        [row for row in summary_rows if row["split"] == "val"],
        key=lambda row: row["f1_mean"],
    )

    lines = [
        "# SigLIP Learning Curve Report",
        "",
        "## Setup",
        f"- Sources: {sources}",
        f"- Total training sizes: {train_sizes}",
        f"- Validation holdout per bucket: {manifest['val_per_bucket']}",
        f"- Test holdout per bucket: {manifest['test_per_bucket']}",
        f"- Repeats per size: {manifest['repeats']}",
        "",
        "## Best Overall Validation Configuration",
        (
            f"- Model: `{best_overall['model']}` at train size `{best_overall['train_size_total']}` "
            f"with validation fake f1 `{format_metric(best_overall['f1_mean'], best_overall['f1_std'])}`"
        ),
        "",
        "## Per-Model Recommendation",
        "| Model | Plateau Train Size | Val F1 | Test F1 | Test Recall | Test ROC AUC |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]

    for model_name in MODEL_NAMES:
        plateau_row = find_plateau_row(summary_rows, "val", model_name, "f1")
        paired_test = paired_test_row(summary_rows, model_name, plateau_row["train_size_total"])
        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    str(plateau_row["train_size_total"]),
                    format_metric(plateau_row["f1_mean"], plateau_row["f1_std"]),
                    format_metric(paired_test["f1_mean"], paired_test["f1_std"]),
                    format_metric(paired_test["recall_mean"], paired_test["recall_std"]),
                    format_metric(paired_test["roc_auc_mean"], paired_test["roc_auc_std"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Validation Learning Curve",
            "| Model | Train Size | Fake F1 | Fake Recall | ROC AUC |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )

    for model_name in MODEL_NAMES:
        model_rows = sorted(
            [
                row for row in summary_rows
                if row["split"] == "val" and row["model"] == model_name
            ],
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
                    ]
                )
                + " |"
            )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    split_root = resolve_path(project_root, args.split_root)
    output_root = resolve_path(project_root, args.output_root)

    if args.clear_outputs and output_root.exists():
        shutil.rmtree(output_root)

    if not args.skip_split:
        create_splits(args, project_root, split_root)

    manifest = load_manifest(split_root)

    for experiment in manifest["experiments"]:
        maybe_run_experiment(
            args=args,
            project_root=project_root,
            experiment_name=experiment["name"],
            split_root=split_root,
            output_root=output_root,
        )

    raw_rows = collect_rows(manifest, output_root)
    raw_fieldnames = [
        "experiment",
        "split",
        "model",
        "train_size_total",
        "repeat",
        "train_real_count",
        "train_fake_count",
        "val_real_count",
        "val_fake_count",
        "test_real_count",
        "test_fake_count",
        *METRIC_NAMES,
    ]
    summary_rows = aggregate_rows(raw_rows)
    summary_fieldnames = [
        "split",
        "model",
        "train_size_total",
        "repeats",
        *[f"{metric_name}_{suffix}" for metric_name in METRIC_NAMES for suffix in ("mean", "std")],
    ]

    summary_dir = output_root / "summary"
    write_csv(summary_dir / "raw_metrics.csv", raw_rows, raw_fieldnames)
    write_csv(summary_dir / "summary_metrics.csv", summary_rows, summary_fieldnames)
    write_markdown_report(summary_dir / "final_report.md", manifest, summary_rows)

    print("\nFinished learning-curve experiment.")
    print(f"Raw metrics: {summary_dir / 'raw_metrics.csv'}")
    print(f"Summary metrics: {summary_dir / 'summary_metrics.csv'}")
    print(f"Markdown report: {summary_dir / 'final_report.md'}")


if __name__ == "__main__":
    main()
