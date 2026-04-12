from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config


def parse_args() -> argparse.Namespace:
    cfg = Config()
    default_base_model = "artifacts/distilbert_df_sample10/model"
    if not (PROJECT_ROOT / default_base_model).exists():
        default_base_model = cfg.TEXT_PHISHING_MODEL_NAME

    parser = argparse.ArgumentParser(
        description=(
            "Continue fine-tuning a DistilBERT text model on OCR-derived text from "
            "data/combined_split. This keeps class weighting enabled through the "
            "underlying trainer and writes a new model folder without overwriting "
            "the source weights."
        )
    )
    parser.add_argument(
        "--split-dir",
        default="data/combined_split",
        help="Directory produced by split_combined_dataset.py.",
    )
    parser.add_argument(
        "--base-model",
        default=default_base_model,
        help="Previously fine-tuned text model to continue training from.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/distilbert_df_then_image",
        help="Directory for the newly fine-tuned model and training artifacts.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=2.0,
        help="Number of epochs for the image-OCR continuation stage.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for continuation training. Lower default helps avoid over-updating on the smaller image dataset.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for continuation training.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer max sequence length.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=5,
        help="Minimum OCR-prepared text length required to keep an image row.",
    )
    parser.add_argument(
        "--objective",
        default="accuracy",
        choices=["accuracy", "f1", "precision", "recall"],
        help="Metric used to pick the best final combined decision boundary on validation.",
    )
    parser.add_argument(
        "--chinese-policy",
        default="route",
        choices=["route"],
        help="Route policy stays fixed to the best-performing pipeline setup.",
    )
    parser.add_argument(
        "--rule-weight",
        type=float,
        default=cfg.TEXT_RULE_WEIGHT,
        help="TEXT_RULE_WEIGHT used by the downstream combined pipeline tuning step.",
    )
    parser.add_argument(
        "--model-weight",
        type=float,
        default=cfg.TEXT_MODEL_WEIGHT,
        help="TEXT_MODEL_WEIGHT used by the downstream combined pipeline tuning step.",
    )
    parser.add_argument(
        "--grid-search-weights",
        action="store_true",
        help="Search rule/model weight combinations instead of using the fixed weights above.",
    )
    parser.add_argument(
        "--rule-weight-step",
        type=float,
        default=0.05,
        help="Grid step for TEXT_RULE_WEIGHT when --grid-search-weights is enabled.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Grid step for MEDIUM_RISK_THRESHOLD during validation tuning.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_script = PROJECT_ROOT / "scripts" / "train_distilbert_pipeline.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Missing image training script: {train_script}")

    command = [
        sys.executable,
        str(train_script),
        "--split-dir",
        args.split_dir,
        "--base-model",
        args.base_model,
        "--output-dir",
        args.output_dir,
        "--chinese-policy",
        args.chinese_policy,
        "--objective",
        args.objective,
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--max-length",
        str(args.max_length),
        "--min-text-length",
        str(args.min_text_length),
        "--rule-weight",
        str(args.rule_weight),
        "--model-weight",
        str(args.model_weight),
        "--rule-weight-step",
        str(args.rule_weight_step),
        "--threshold-step",
        str(args.threshold_step),
    ]

    if args.grid_search_weights:
        command.append("--grid-search-weights")
    if args.overwrite_output_dir:
        command.append("--overwrite-output-dir")

    print("Running continuation fine-tune with class-weighted image OCR data...")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
