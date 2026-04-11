import argparse
import shutil
from pathlib import Path

from transformers import AutoModel, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download SigLIP model files into a local artifacts directory."
    )
    parser.add_argument(
        "--model-id",
        default="google/siglip2-base-patch16-224",
        help="Hugging Face model id to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/siglip2-base-patch16-224"),
        help="Directory to save the processor and model files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir, force):
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Use --force to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    ensure_output_dir(args.output_dir, args.force)

    print(f"Downloading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    print(f"Downloading model: {args.model_id}")
    model = AutoModel.from_pretrained(args.model_id)

    processor.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    abs_output_dir = args.output_dir.resolve()
    print(f"Saved artifacts to: {abs_output_dir}")
    print("Use this path for training or inference:")
    print(f"SIGLIP_MODEL_NAME={abs_output_dir}")


if __name__ == "__main__":
    main()
