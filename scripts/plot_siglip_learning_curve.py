import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = ("logreg", "lightgbm", "xgboost")
MODEL_COLORS = {
    "logreg": "#1f77b4",
    "lightgbm": "#2ca02c",
    "xgboost": "#d62728",
}
METRICS = (
    ("accuracy", "Accuracy"),
    ("f1", "Fake F1"),
    ("recall", "Fake Recall"),
    ("roc_auc", "ROC AUC"),
)
SPLITS = ("val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SigLIP learning curves from the aggregated summary_metrics.csv file."
    )
    parser.add_argument(
        "--summary-csv",
        default="outputs/learning_curve_chat_social_curve/summary/summary_metrics.csv",
        help="Path to the aggregated summary metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/learning_curve_chat_social_curve/summary",
        help="Directory to save the generated plots.",
    )
    return parser.parse_args()


def style_axis(ax, split_name: str, metric_label: str):
    ax.set_title(f"{split_name.upper()} {metric_label}")
    ax.set_xlabel("Training Size")
    ax.set_ylabel(metric_label)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)


def plot_model_lines(ax, panel_df: pd.DataFrame, metric_name: str):
    for model_name in MODEL_ORDER:
        model_df = panel_df[panel_df["model"] == model_name].sort_values("train_size_total")
        if model_df.empty:
            continue

        x = model_df["train_size_total"].to_numpy()
        mean = model_df[f"{metric_name}_mean"].to_numpy()
        std = model_df[f"{metric_name}_std"].to_numpy()
        color = MODEL_COLORS[model_name]

        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=color,
            label=model_name,
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.14,
        )


def plot_panel(ax, df: pd.DataFrame, split_name: str, metric_name: str, metric_label: str):
    style_axis(ax, split_name, metric_label)
    panel_df = df[df["split"] == split_name].copy()
    plot_model_lines(ax, panel_df, metric_name)


def plot_grid(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=False, sharey=True)

    for row_idx, split_name in enumerate(SPLITS):
        for col_idx, (metric_name, metric_label) in enumerate(METRICS):
            plot_panel(axes[row_idx, col_idx], df, split_name, metric_name, metric_label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("SigLIP Learning Curves on Chat + Social", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    png_path = output_dir / "learning_curve_grid.png"
    svg_path = output_dir / "learning_curve_grid.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def plot_cv_grid(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharex=False, sharey=True)

    for col_idx, (metric_name, metric_label) in enumerate(METRICS):
        ax = axes[col_idx]
        style_axis(ax, "CV Validation", metric_label)
        plot_model_lines(ax, df, metric_name)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("SigLIP Learning Curves (K-Fold CV Validation)", fontsize=16, y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    png_path = output_dir / "learning_curve_cv_grid.png"
    svg_path = output_dir / "learning_curve_cv_grid.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def plot_named_grid(df: pd.DataFrame, output_dir: Path, title_prefix: str, stem_prefix: str):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharex=False, sharey=True)

    for col_idx, (metric_name, metric_label) in enumerate(METRICS):
        ax = axes[col_idx]
        style_axis(ax, title_prefix, metric_label)
        plot_model_lines(ax, df, metric_name)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"SigLIP Learning Curves ({title_prefix})", fontsize=16, y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    png_path = output_dir / f"{stem_prefix}_grid.png"
    svg_path = output_dir / f"{stem_prefix}_grid.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def plot_test_f1(df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_panel(ax, df, "test", "f1", "Fake F1")
    ax.legend(frameon=False)
    ax.set_title("Test Fake F1 vs Training Size")
    fig.tight_layout()

    png_path = output_dir / "learning_curve_test_f1.png"
    svg_path = output_dir / "learning_curve_test_f1.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def plot_test_accuracy(df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_panel(ax, df, "test", "accuracy", "Accuracy")
    ax.legend(frameon=False)
    ax.set_title("Test Accuracy vs Training Size")
    fig.tight_layout()

    png_path = output_dir / "learning_curve_test_accuracy.png"
    svg_path = output_dir / "learning_curve_test_accuracy.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def plot_cv_metric(
    df: pd.DataFrame,
    output_dir: Path,
    metric_name: str,
    metric_label: str,
    stem: str,
    axis_title: str = "CV Validation",
    chart_title: str | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    style_axis(ax, axis_title, metric_label)
    plot_model_lines(ax, df, metric_name)
    ax.legend(frameon=False)
    if chart_title is None:
        chart_title = f"{axis_title} {metric_label} vs Training Size"
    ax.set_title(chart_title)
    fig.tight_layout()

    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def plot_final_test_bars(df: pd.DataFrame, output_dir: Path, metric_name: str, metric_label: str, stem: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_df = df.set_index("model").reindex(MODEL_ORDER).dropna(subset=[metric_name], how="all")
    colors = [MODEL_COLORS[model_name] for model_name in plot_df.index]

    bars = ax.bar(plot_df.index.tolist(), plot_df[metric_name].to_numpy(), color=colors, width=0.62)
    ax.set_title(f"Final Untouched Test {metric_label}")
    ax.set_ylabel(metric_label)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

    for bar, value in zip(bars, plot_df[metric_name].to_numpy()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.02, 1.0),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()

    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def main():
    args = parse_args()
    summary_csv = Path(args.summary_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    if "split" in df.columns:
        grid_png, grid_svg = plot_grid(df, output_dir)
        test_f1_png, test_f1_svg = plot_test_f1(df, output_dir)
        test_acc_png, test_acc_svg = plot_test_accuracy(df, output_dir)

        print(f"Saved {grid_png}")
        print(f"Saved {grid_svg}")
        print(f"Saved {test_f1_png}")
        print(f"Saved {test_f1_svg}")
        print(f"Saved {test_acc_png}")
        print(f"Saved {test_acc_svg}")
        return

    if "final_train_mode" in df.columns:
        test_f1_png, test_f1_svg = plot_final_test_bars(
            df,
            output_dir,
            metric_name="f1",
            metric_label="Fake F1",
            stem="final_test_f1",
        )
        test_acc_png, test_acc_svg = plot_final_test_bars(
            df,
            output_dir,
            metric_name="accuracy",
            metric_label="Accuracy",
            stem="final_test_accuracy",
        )
        test_roc_png, test_roc_svg = plot_final_test_bars(
            df,
            output_dir,
            metric_name="roc_auc",
            metric_label="ROC AUC",
            stem="final_test_roc_auc",
        )

        print(f"Saved {test_f1_png}")
        print(f"Saved {test_f1_svg}")
        print(f"Saved {test_acc_png}")
        print(f"Saved {test_acc_svg}")
        print(f"Saved {test_roc_png}")
        print(f"Saved {test_roc_svg}")
        return

    if "evaluation_set" in df.columns:
        eval_label = df["evaluation_set"].iloc[0]
        title_prefix = "Fixed Test Curve" if eval_label == "fixed_test_curve" else str(eval_label)
        stem_prefix = "learning_curve_test" if eval_label == "fixed_test_curve" else "learning_curve_named"
        grid_png, grid_svg = plot_named_grid(df, output_dir, title_prefix=title_prefix, stem_prefix=stem_prefix)
        f1_png, f1_svg = plot_cv_metric(
            df,
            output_dir,
            metric_name="f1",
            metric_label="Fake F1",
            stem=f"{stem_prefix}_f1",
            axis_title=title_prefix,
            chart_title=f"{title_prefix} Fake F1 vs Training Size",
        )
        acc_png, acc_svg = plot_cv_metric(
            df,
            output_dir,
            metric_name="accuracy",
            metric_label="Accuracy",
            stem=f"{stem_prefix}_accuracy",
            axis_title=title_prefix,
            chart_title=f"{title_prefix} Accuracy vs Training Size",
        )

        print(f"Saved {grid_png}")
        print(f"Saved {grid_svg}")
        print(f"Saved {f1_png}")
        print(f"Saved {f1_svg}")
        print(f"Saved {acc_png}")
        print(f"Saved {acc_svg}")
        return

    cv_grid_png, cv_grid_svg = plot_cv_grid(df, output_dir)
    cv_f1_png, cv_f1_svg = plot_cv_metric(
        df,
        output_dir,
        metric_name="f1",
        metric_label="Fake F1",
        stem="learning_curve_cv_f1",
        axis_title="CV Validation",
        chart_title="CV Validation Fake F1 vs Training Size",
    )
    cv_acc_png, cv_acc_svg = plot_cv_metric(
        df,
        output_dir,
        metric_name="accuracy",
        metric_label="Accuracy",
        stem="learning_curve_cv_accuracy",
        axis_title="CV Validation",
        chart_title="CV Validation Accuracy vs Training Size",
    )

    print(f"Saved {cv_grid_png}")
    print(f"Saved {cv_grid_svg}")
    print(f"Saved {cv_f1_png}")
    print(f"Saved {cv_f1_svg}")
    print(f"Saved {cv_acc_png}")
    print(f"Saved {cv_acc_svg}")


if __name__ == "__main__":
    main()
