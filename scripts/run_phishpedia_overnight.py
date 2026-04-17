from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from urllib import error as urlerror
from urllib import request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config


def parse_args() -> argparse.Namespace:
    cfg = Config()
    default_base_model = "models/text_phishing_base"
    if not (PROJECT_ROOT / default_base_model).exists():
        default_base_model = cfg.TEXT_PHISHING_MODEL_NAME

    parser = argparse.ArgumentParser(
        description=(
            "Sample a Phishpedia split matched to english_data_split, then train and ablate "
            "EasyOCR, EasyOCR + Grounding DINO, and LLaMA/Ollama overnight."
        )
    )
    parser.add_argument(
        "--reference-split-dir",
        default="data/english_data_split",
        help="Reference split whose train/val/test class counts should be matched.",
    )
    parser.add_argument(
        "--phishing-source-dir",
        default="data/phishpedia_phishing",
        help="Phishpedia phishing source directory.",
    )
    parser.add_argument(
        "--benign-source-dir",
        default="data/phishpedia_benign",
        help="Phishpedia benign source directory.",
    )
    parser.add_argument(
        "--sampled-split-dir",
        default="data/phishpedia_matched_split",
        help="Where the matched Phishpedia train/val/test split will be created.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phishpedia_overnight",
        help="Root directory for trained models, logs, and ablation outputs.",
    )
    parser.add_argument(
        "--train-backends",
        default="easyocr,easyocr_grounded,llama",
        help="Comma-separated backends to train and evaluate: easyocr, easyocr_grounded, llama.",
    )
    parser.add_argument(
        "--base-model",
        default=default_base_model,
        help="Starting DistilBERT checkpoint for all three training runs.",
    )
    parser.add_argument(
        "--objective",
        choices=["accuracy", "f1", "precision", "recall"],
        default="accuracy",
        help="Validation objective used in training and ablation threshold selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Deterministic sampling seed for the matched Phishpedia split.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Number of fine-tuning epochs for each backend-specific text model.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for DistilBERT fine-tuning.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for DistilBERT fine-tuning.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device evaluation batch size.",
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
        help="Minimum prepared text length required to keep a sample during training.",
    )
    parser.add_argument(
        "--chinese-policy",
        default="route",
        choices=["route"],
        help="Chinese routing policy for the shared OCR/text pipeline.",
    )
    parser.add_argument(
        "--easyocr-grounding-dino-model",
        default=getattr(cfg, "OCR_EASYOCR_GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny"),
        help="Grounding DINO model used for the grounded EasyOCR run.",
    )
    parser.add_argument(
        "--easyocr-grounding-dino-prompt",
        default=getattr(
            cfg,
            "OCR_EASYOCR_GROUNDING_DINO_PROMPT",
            "text. paragraph. text block. message. chat bubble. dialog.",
        ),
        help="Grounding DINO prompt used for the grounded EasyOCR run.",
    )
    parser.add_argument(
        "--easyocr-grounding-box-threshold",
        type=float,
        default=float(getattr(cfg, "OCR_EASYOCR_GROUNDING_BOX_THRESHOLD", 0.25)),
        help="Grounding DINO box threshold for the grounded EasyOCR run.",
    )
    parser.add_argument(
        "--easyocr-grounding-text-threshold",
        type=float,
        default=float(getattr(cfg, "OCR_EASYOCR_GROUNDING_TEXT_THRESHOLD", 0.25)),
        help="Grounding DINO text threshold for the grounded EasyOCR run.",
    )
    parser.add_argument(
        "--easyocr-grounding-max-regions",
        type=int,
        default=int(getattr(cfg, "OCR_EASYOCR_GROUNDING_MAX_REGIONS", 6)),
        help="Maximum number of grounded OCR regions to consider.",
    )
    parser.add_argument(
        "--easyocr-grounding-padding-ratio",
        type=float,
        default=float(getattr(cfg, "OCR_EASYOCR_GROUNDING_PADDING_RATIO", 0.03)),
        help="Extra crop padding ratio for grounded EasyOCR.",
    )
    parser.add_argument(
        "--easyocr-grounding-text-aggregation",
        choices=["concat", "max_model", "hybrid_max_model"],
        default=str(getattr(cfg, "OCR_EASYOCR_GROUNDING_TEXT_AGGREGATION", "concat")),
        help="How grounded OCR crop texts are aggregated before text-model scoring.",
    )
    parser.add_argument(
        "--ollama-model",
        default=getattr(cfg, "OCR_OLLAMA_MODEL", "llama3.2-vision"),
        help="Ollama vision model for the llama backend.",
    )
    parser.add_argument(
        "--ollama-host",
        default=getattr(cfg, "OCR_OLLAMA_HOST", "http://localhost:11434"),
        help="Base URL for the Ollama server.",
    )
    parser.add_argument(
        "--ollama-timeout-seconds",
        type=int,
        default=int(getattr(cfg, "OCR_OLLAMA_TIMEOUT_SECONDS", 150)),
        help="Per-image Ollama timeout in seconds.",
    )
    parser.add_argument(
        "--show-misclassifications",
        type=int,
        default=3,
        help="How many sample test misclassifications to print during ablation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional image limit per class for smoke tests.",
    )
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Reuse an existing sampled split instead of recreating it.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip all fine-tuning runs and reuse existing model directories under output-root.",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the final ablation run.",
    )
    parser.add_argument(
        "--overwrite-sampled-split",
        action="store_true",
        help="Allow rebuilding sampled-split-dir if it already exists.",
    )
    parser.add_argument(
        "--overwrite-training-dirs",
        action="store_true",
        help="Allow overwriting existing training output directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    return parser.parse_args()


def parse_backend_list(value: str) -> list[str]:
    aliases = {
        "easyocr": "easyocr",
        "easyocr_grounded": "easyocr_grounded",
        "easyocr+dino": "easyocr_grounded",
        "easyocr_dino": "easyocr_grounded",
        "ollama": "llama",
        "llama": "llama",
    }
    parsed: list[str] = []
    for token in str(value).split(","):
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        backend = aliases.get(cleaned)
        if backend is None:
            raise ValueError(
                f"Unsupported backend '{token}'. Allowed backends: easyocr, easyocr_grounded, llama."
            )
        if backend not in parsed:
            parsed.append(backend)
    if not parsed:
        raise ValueError("At least one backend must be provided.")
    return parsed


def render_backend_label(backend: str) -> str:
    labels = {
        "easyocr": "EasyOCR",
        "easyocr_grounded": "EasyOCR + Grounding DINO",
        "llama": "LLaMA via Ollama",
    }
    return labels.get(backend, backend)


def build_sample_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/sample_phishpedia_to_match_english.py",
        "--reference-split-dir",
        args.reference_split_dir,
        "--phishing-source-dir",
        args.phishing_source_dir,
        "--benign-source-dir",
        args.benign_source_dir,
        "--output-dir",
        args.sampled_split_dir,
        "--seed",
        str(args.seed),
    ]
    if args.overwrite_sampled_split:
        command.append("--overwrite-output-dir")
    return command


def build_train_command(
    args: argparse.Namespace,
    backend: str,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_distilbert_pipeline.py",
        "--split-dir",
        args.sampled_split_dir,
        "--base-model",
        args.base_model,
        "--output-dir",
        str(output_dir),
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
        "--chinese-policy",
        args.chinese_policy,
    ]
    if args.overwrite_training_dirs:
        command.append("--overwrite-output-dir")

    if backend == "easyocr":
        command.extend(["--ocr-backend", "easyocr"])
    elif backend == "easyocr_grounded":
        command.extend(
            [
                "--ocr-backend",
                "easyocr",
                "--easyocr-use-grounding-dino",
                "--easyocr-grounding-dino-model",
                args.easyocr_grounding_dino_model,
                "--easyocr-grounding-dino-prompt",
                args.easyocr_grounding_dino_prompt,
                "--easyocr-grounding-box-threshold",
                str(args.easyocr_grounding_box_threshold),
                "--easyocr-grounding-text-threshold",
                str(args.easyocr_grounding_text_threshold),
                "--easyocr-grounding-max-regions",
                str(args.easyocr_grounding_max_regions),
                "--easyocr-grounding-padding-ratio",
                str(args.easyocr_grounding_padding_ratio),
                "--easyocr-grounding-text-aggregation",
                args.easyocr_grounding_text_aggregation,
            ]
        )
    elif backend == "llama":
        command.extend(
            [
                "--ocr-backend",
                "ollama",
                "--ollama-model",
                args.ollama_model,
                "--ollama-host",
                args.ollama_host,
                "--ollama-timeout-seconds",
                str(args.ollama_timeout_seconds),
            ]
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return command


def build_ablation_command(
    args: argparse.Namespace,
    backends: list[str],
    model_dirs: dict[str, Path],
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_english_ocr_ablation.py",
        "--val-phishing-dir",
        str(Path(args.sampled_split_dir) / "val" / "fake"),
        "--val-non-phishing-dir",
        str(Path(args.sampled_split_dir) / "val" / "real"),
        "--test-phishing-dir",
        str(Path(args.sampled_split_dir) / "test" / "fake"),
        "--test-non-phishing-dir",
        str(Path(args.sampled_split_dir) / "test" / "real"),
        "--backends",
        ",".join(backends),
        "--objective",
        args.objective,
        "--output-dir",
        str(output_dir),
        "--show-misclassifications",
        str(args.show_misclassifications),
        "--easyocr-text-model",
        str(model_dirs["easyocr"] / "model"),
        "--easyocr-grounded-text-model",
        str(model_dirs["easyocr_grounded"] / "model"),
        "--llama-text-model",
        str(model_dirs["llama"] / "model"),
        "--ollama-model",
        args.ollama_model,
        "--ollama-host",
        args.ollama_host,
        "--ollama-timeout-seconds",
        str(args.ollama_timeout_seconds),
        "--easyocr-grounding-dino-model",
        args.easyocr_grounding_dino_model,
        "--easyocr-grounding-dino-prompt",
        args.easyocr_grounding_dino_prompt,
        "--easyocr-grounding-box-threshold",
        str(args.easyocr_grounding_box_threshold),
        "--easyocr-grounding-text-threshold",
        str(args.easyocr_grounding_text_threshold),
        "--easyocr-grounding-max-regions",
        str(args.easyocr_grounding_max_regions),
        "--easyocr-grounding-padding-ratio",
        str(args.easyocr_grounding_padding_ratio),
        "--easyocr-grounding-text-aggregation",
        args.easyocr_grounding_text_aggregation,
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def render_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_command(command: list[str], log_path: Path, dry_run: bool) -> dict[str, object]:
    rendered = render_command(command)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[dry-run] {rendered}")
        return {
            "command": rendered,
            "log_path": str(log_path),
            "exit_code": 0,
            "duration_seconds": 0.0,
            "status": "skipped_dry_run",
        }

    start_time = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(f"$ {rendered}\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        exit_code = process.wait()

    duration = round(time.time() - start_time, 2)
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, command)

    return {
        "command": rendered,
        "log_path": str(log_path),
        "exit_code": exit_code,
        "duration_seconds": duration,
        "status": "completed",
    }


def validate_paths_for_ablation(backends: list[str], model_dirs: dict[str, Path]) -> None:
    for backend in backends:
        model_dir = model_dirs[backend] / "model"
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Missing trained model for {render_backend_label(backend)}: {model_dir}. "
                "Either run training first or point the overnight run at an output-root "
                "that already contains the trained models."
            )


def write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_ollama_ready(args: argparse.Namespace) -> str:
    endpoint = args.ollama_host.rstrip("/") + "/api/tags"
    try:
        with request.urlopen(endpoint, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urlerror.URLError as exc:
        raise RuntimeError(
            f"Ollama backend requested, but the server is unreachable at {args.ollama_host}. "
            "Start Ollama first, then rerun the overnight script."
        ) from exc

    models = payload.get("models") or []
    available_model_names = {
        str(item.get("name") or "").strip()
        for item in models
        if isinstance(item, dict)
    }
    requested = str(args.ollama_model).strip()
    if not requested:
        return requested
    if requested in available_model_names:
        return requested

    latest_alias = f"{requested}:latest"
    if latest_alias in available_model_names:
        return latest_alias

    base_matches = [
        name for name in sorted(available_model_names)
        if name.split(":", 1)[0] == requested
    ]
    if len(base_matches) == 1:
        return base_matches[0]

    raise RuntimeError(
        f"Ollama backend requested model '{requested}', but it was not found on {args.ollama_host}. "
        f"Available models: {sorted(name for name in available_model_names if name)}"
    )


def main() -> int:
    args = parse_args()
    backends = parse_backend_list(args.train_backends)

    sampled_split_dir = PROJECT_ROOT / args.sampled_split_dir
    output_root = PROJECT_ROOT / args.output_root
    model_root = output_root / "models"
    ablation_output_dir = output_root / "ablation"
    logs_dir = output_root / "logs"
    summary_path = output_root / "run_summary.json"

    model_dirs = {
        "easyocr": model_root / "easyocr",
        "easyocr_grounded": model_root / "easyocr_grounded",
        "llama": model_root / "llama",
    }

    run_summary: dict[str, object] = {
        "reference_split_dir": str(PROJECT_ROOT / args.reference_split_dir),
        "phishing_source_dir": str(PROJECT_ROOT / args.phishing_source_dir),
        "benign_source_dir": str(PROJECT_ROOT / args.benign_source_dir),
        "sampled_split_dir": str(sampled_split_dir),
        "output_root": str(output_root),
        "backends": backends,
        "objective": args.objective,
        "base_model": args.base_model,
        "seed": args.seed,
        "steps": [],
    }

    output_root.mkdir(parents=True, exist_ok=True)

    try:
        if "llama" in backends and not args.dry_run:
            resolved_ollama_model = ensure_ollama_ready(args)
            if resolved_ollama_model != args.ollama_model:
                print(
                    f"\n=== Ollama model resolved: {args.ollama_model} -> {resolved_ollama_model} ==="
                )
                args.ollama_model = resolved_ollama_model

        if not args.skip_sampling:
            print("\n=== Sampling matched Phishpedia split ===")
            step_result = run_command(
                build_sample_command(args),
                logs_dir / "01_sample_phishpedia.log",
                args.dry_run,
            )
            run_summary["steps"].append({"step": "sample_split", **step_result})
        else:
            print("\n=== Skipping sampling; reusing existing sampled split ===")
            run_summary["steps"].append({"step": "sample_split", "status": "skipped"})

        if not args.skip_training:
            for index, backend in enumerate(backends, start=1):
                print(f"\n=== Training {render_backend_label(backend)} text model ===")
                step_result = run_command(
                    build_train_command(args, backend, model_dirs[backend]),
                    logs_dir / f"{index + 1:02d}_train_{backend}.log",
                    args.dry_run,
                )
                run_summary["steps"].append(
                    {
                        "step": f"train_{backend}",
                        "backend": backend,
                        "output_dir": str(model_dirs[backend]),
                        **step_result,
                    }
                )
        else:
            print("\n=== Skipping training; reusing existing trained model directories ===")
            run_summary["steps"].append({"step": "train_all", "status": "skipped"})

        if not args.skip_ablation:
            if not args.dry_run:
                validate_paths_for_ablation(backends, model_dirs)
            print("\n=== Running Phishpedia OCR ablation ===")
            step_result = run_command(
                build_ablation_command(args, backends, model_dirs, ablation_output_dir),
                logs_dir / "99_run_ablation.log",
                args.dry_run,
            )
            run_summary["steps"].append(
                {
                    "step": "run_ablation",
                    "output_dir": str(ablation_output_dir),
                    **step_result,
                }
            )
        else:
            print("\n=== Skipping ablation ===")
            run_summary["steps"].append({"step": "run_ablation", "status": "skipped"})

        run_summary["status"] = "completed"
        write_summary(summary_path, run_summary)
        print(f"\nRun summary saved to: {summary_path.resolve()}")
        return 0
    except Exception as exc:
        run_summary["status"] = "failed"
        run_summary["error"] = str(exc)
        write_summary(summary_path, run_summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
