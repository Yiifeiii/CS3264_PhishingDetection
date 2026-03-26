import argparse
from pathlib import Path

from models.model_factory import SUPPORTED_MODELS, create_model
from ocr.ocr_service import OCRService
from preprocess.image_preprocessor import ImagePreprocessor
from utils.benchmarking import (
    DatasetSample,
    discover_labeled_images,
    format_summary_table,
    summarize_predictions,
    write_benchmark_csv,
)
from utils.config import Config
from utils.distildire_finetune import run_distildire_finetune
from utils.safe_finetune import run_safe_finetune


class App:
    def __init__(self):
        self.cfg = Config()
        self.preprocessor = ImagePreprocessor()
        self._ocr = None

    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = OCRService()
        return self._ocr

    def predict(
        self,
        model_name: str,
        image_path: str,
        include_ocr: bool = False,
        distildire_model_path: str | None = None,
        distildire_adm_model_path: str | None = None,
    ) -> dict:
        model = self._create_model(
            model_name,
            distildire_model_path=distildire_model_path,
            distildire_adm_model_path=distildire_adm_model_path,
        )
        image = self.preprocessor.load_image(image_path)
        result = model.predict_image(image)

        if include_ocr:
            result["ocr_text"] = self.ocr.extract_text(image_path)

        return result

    def benchmark(
        self,
        model_names: list[str],
        dataset_root: str,
        limit: int | None = None,
        distildire_model_path: str | None = None,
        distildire_adm_model_path: str | None = None,
    ) -> tuple[list[dict], dict[str, dict]]:
        samples = discover_labeled_images(dataset_root)
        if not samples:
            raise ValueError(
                f"No labeled images found under {dataset_root}. "
                "Expected paths containing real/fake or 0_real/1_fake."
            )

        if limit is not None:
            samples = samples[:limit]

        rows = []
        summary_by_model = {}

        for model_name in model_names:
            model = self._create_model(
                model_name,
                distildire_model_path=distildire_model_path,
                distildire_adm_model_path=distildire_adm_model_path,
            )
            model_rows = self._run_model_on_samples(model_name, model, samples)
            rows.extend(model_rows)
            summary_by_model[model_name] = summarize_predictions(model_rows)

        return rows, summary_by_model

    def _create_model(
        self,
        model_name: str,
        distildire_model_path: str | None = None,
        distildire_adm_model_path: str | None = None,
    ):
        if model_name == "distildire":
            return create_model(
                model_name,
                self.cfg,
                model_path=distildire_model_path,
                adm_model_path=distildire_adm_model_path,
            )

        return create_model(model_name, self.cfg)

    def _run_model_on_samples(
        self,
        model_name: str,
        model,
        samples: list[DatasetSample],
    ) -> list[dict]:
        rows = []

        for sample in samples:
            image = self.preprocessor.load_image(str(sample.path))
            result = model.predict_image(image)
            rows.append(
                {
                    "model": model_name,
                    "path": str(sample.path),
                    "label": sample.label,
                    "prediction": result["prediction"],
                    "correct": result["prediction"] == sample.label,
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                }
            )

        return rows


def parse_args(cfg: Config):
    parser = argparse.ArgumentParser(
        description="Run image deepfake detectors and benchmark them on a labeled dataset.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Run one model on one image.")
    predict_parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=cfg.MODEL_BACKEND,
        help="Model backend to use.",
    )
    predict_parser.add_argument(
        "--image",
        required=True,
        help="Path to the image to evaluate.",
    )
    predict_parser.add_argument(
        "--ocr",
        action="store_true",
        help="Run OCR on the image and print the extracted text.",
    )
    predict_parser.add_argument(
        "--distildire-model-path",
        default=None,
        help="Optional DistilDIRE checkpoint override for predict.",
    )
    predict_parser.add_argument(
        "--distildire-adm-model-path",
        default=None,
        help="Optional ADM checkpoint override for DistilDIRE predict.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark one or more models on a labeled dataset.",
    )
    benchmark_parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
        help="Model backends to benchmark.",
    )
    benchmark_parser.add_argument(
        "--dataset-root",
        default=cfg.BENCHMARK_DATASET_ROOT,
        help="Dataset root containing real/ and fake/ folders.",
    )
    benchmark_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to evaluate per benchmark run.",
    )
    benchmark_parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save per-image benchmark results as CSV.",
    )
    benchmark_parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print per-image predictions after the summary table.",
    )
    benchmark_parser.add_argument(
        "--distildire-model-path",
        default=None,
        help="Optional DistilDIRE checkpoint override for benchmark.",
    )
    benchmark_parser.add_argument(
        "--distildire-adm-model-path",
        default=None,
        help="Optional ADM checkpoint override for DistilDIRE benchmark.",
    )

    finetune_parser = subparsers.add_parser(
        "finetune-safe",
        help="Fine-tune SAFE on a local labeled dataset.",
    )
    finetune_parser.add_argument(
        "--train-data-path",
        required=True,
        help="Path to the training dataset root containing real/fake or 0_real/1_fake folders.",
    )
    finetune_parser.add_argument(
        "--val-data-path",
        required=True,
        help="Path to the validation dataset root containing real/fake or 0_real/1_fake folders.",
    )
    finetune_parser.add_argument(
        "--output-dir",
        default=cfg.SAFE_FINETUNE_OUTPUT_DIR,
        help="Directory to save fine-tuned checkpoints.",
    )
    finetune_parser.add_argument(
        "--pretrained-path",
        default=cfg.SAFE_MODEL_PATH,
        help="SAFE checkpoint used to initialize fine-tuning.",
    )
    finetune_parser.add_argument("--epochs", type=int, default=5)
    finetune_parser.add_argument("--batch-size", type=int, default=8)
    finetune_parser.add_argument("--lr", type=float, default=1e-4)
    finetune_parser.add_argument("--weight-decay", type=float, default=1e-4)
    finetune_parser.add_argument("--num-workers", type=int, default=2)
    finetune_parser.add_argument(
        "--val-ratio",
        type=float,
        default=cfg.SAFE_VAL_RATIO,
        help="Validation ratio used when train and val paths point to the same root.",
    )
    finetune_parser.add_argument("--input-size", type=int, default=cfg.SAFE_INPUT_SIZE)
    finetune_parser.add_argument(
        "--transform-mode",
        default=cfg.SAFE_TRANSFORM_MODE,
        choices=["crop", "resize_BILINEAR", "resize_NEAREST", "source"],
    )
    finetune_parser.add_argument("--num-train", type=int, default=None)
    finetune_parser.add_argument("--num-val", type=int, default=None)
    finetune_parser.add_argument("--device", default=cfg.DEVICE)
    finetune_parser.add_argument("--seed", type=int, default=42)
    finetune_parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the SAFE backbone and train only the classifier head.",
    )

    distildire_finetune_parser = subparsers.add_parser(
        "finetune-distildire",
        help="Fine-tune DistilDIRE on a local labeled dataset.",
    )
    distildire_finetune_parser.add_argument(
        "--train-data-path",
        default=cfg.DISTILDIRE_FINETUNE_TRAIN_PATH,
        help="Path to the training dataset root containing real/fake folders.",
    )
    distildire_finetune_parser.add_argument(
        "--val-data-path",
        default=cfg.DISTILDIRE_FINETUNE_VAL_PATH,
        help="Path to the validation dataset root containing real/fake folders.",
    )
    distildire_finetune_parser.add_argument(
        "--output-dir",
        default=cfg.DISTILDIRE_FINETUNE_OUTPUT_DIR,
        help="Directory to save fine-tuned checkpoints.",
    )
    distildire_finetune_parser.add_argument(
        "--pretrained-path",
        default=cfg.DISTILDIRE_MODEL_PATH,
        help="DistilDIRE checkpoint used to initialize fine-tuning.",
    )
    distildire_finetune_parser.add_argument(
        "--adm-model-path",
        default=cfg.DISTILDIRE_ADM_MODEL_PATH,
        help="ADM diffusion checkpoint used to compute first-step noise.",
    )
    distildire_finetune_parser.add_argument("--epochs", type=int, default=3)
    distildire_finetune_parser.add_argument("--batch-size", type=int, default=4)
    distildire_finetune_parser.add_argument("--lr", type=float, default=1e-4)
    distildire_finetune_parser.add_argument("--weight-decay", type=float, default=1e-4)
    distildire_finetune_parser.add_argument("--num-workers", type=int, default=0)
    distildire_finetune_parser.add_argument(
        "--val-ratio",
        type=float,
        default=cfg.DISTILDIRE_VAL_RATIO,
        help="Validation ratio used when train and val paths point to the same root.",
    )
    distildire_finetune_parser.add_argument(
        "--input-size",
        type=int,
        default=cfg.DISTILDIRE_IMAGE_SIZE,
    )
    distildire_finetune_parser.add_argument(
        "--fake-threshold",
        type=float,
        default=cfg.DISTILDIRE_FAKE_THRESHOLD,
        help="Probability threshold used for validation metrics and predictions.",
    )
    distildire_finetune_parser.add_argument(
        "--num-train",
        type=int,
        default=None,
        help="Optional cap on training images with balanced label sampling.",
    )
    distildire_finetune_parser.add_argument(
        "--num-val",
        type=int,
        default=None,
        help="Optional cap on validation images with balanced label sampling.",
    )
    distildire_finetune_parser.add_argument("--device", default=cfg.DEVICE)
    distildire_finetune_parser.add_argument("--seed", type=int, default=42)
    distildire_finetune_parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the DistilDIRE backbone and train only the student head.",
    )

    return parser.parse_args()


def print_prediction(image_path: str, model_name: str, result: dict):
    print(f"\nImage: {image_path}")
    print("Model Backend:", model_name)
    print("Prediction:", result["prediction"])
    print("Confidence:", round(result["confidence"], 4))
    print("Probabilities:", result["probabilities"])
    if "ocr_text" in result:
        print("OCR Text:", result["ocr_text"])


def print_benchmark_details(rows: list[dict]):
    for row in rows:
        file_name = Path(row["path"]).name
        print(
            f"{row['model']:<12} "
            f"{row['label']:<5} "
            f"{row['prediction']:<5} "
            f"{row['confidence']:.4f} "
            f"{file_name}"
        )


def main():
    cfg = Config()
    args = parse_args(cfg)
    app = App()

    if args.command == "predict":
        result = app.predict(
            args.model,
            args.image,
            include_ocr=args.ocr,
            distildire_model_path=args.distildire_model_path,
            distildire_adm_model_path=args.distildire_adm_model_path,
        )
        print_prediction(args.image, args.model, result)
        return

    if args.command == "finetune-safe":
        run_safe_finetune(args)
        return

    if args.command == "finetune-distildire":
        run_distildire_finetune(args)
        return

    rows, summary_by_model = app.benchmark(
        model_names=args.models,
        dataset_root=args.dataset_root,
        limit=args.limit,
        distildire_model_path=args.distildire_model_path,
        distildire_adm_model_path=args.distildire_adm_model_path,
    )

    print(format_summary_table(summary_by_model))

    best_model = max(
        summary_by_model.items(),
        key=lambda item: (item[1]["accuracy"], item[1]["f1"]),
    )
    print(
        f"\nBest model by accuracy then F1: {best_model[0]} "
        f"(accuracy={best_model[1]['accuracy']:.3f}, f1={best_model[1]['f1']:.3f})"
    )

    if args.output_csv:
        write_benchmark_csv(rows, args.output_csv)
        print(f"Saved benchmark rows to: {args.output_csv}")

    if args.show_details:
        print("\nDetailed predictions")
        print("--------------------")
        print_benchmark_details(rows)


if __name__ == "__main__":
    main()
