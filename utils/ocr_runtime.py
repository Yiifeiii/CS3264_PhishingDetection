from __future__ import annotations

import argparse

from ocr.ocr_service import (
    DEFAULT_TRANSFORMERS_MODEL,
    DEFAULT_TRANSFORMERS_TASK_PROMPT,
    OCRService,
    SUPPORTED_OCR_BACKENDS,
)


def add_ocr_runtime_args(
    parser: argparse.ArgumentParser,
    cfg,
    *,
    default_backend: str | None = None,
    default_timeout_seconds: int | None = None,
) -> argparse.ArgumentParser:
    backend_default = default_backend or getattr(cfg, "OCR_BACKEND", "easyocr")
    ollama_model_default = getattr(cfg, "OCR_OLLAMA_MODEL", "llama3.2-vision")
    ollama_host_default = getattr(cfg, "OCR_OLLAMA_HOST", "http://localhost:11434")
    ollama_timeout_default = (
        default_timeout_seconds
        if default_timeout_seconds is not None
        else int(getattr(cfg, "OCR_OLLAMA_TIMEOUT_SECONDS", 150))
    )
    easyocr_use_grounding_dino_default = bool(getattr(cfg, "OCR_EASYOCR_USE_GROUNDING_DINO", False))
    easyocr_grounding_dino_model_default = getattr(
        cfg,
        "OCR_EASYOCR_GROUNDING_DINO_MODEL",
        "IDEA-Research/grounding-dino-tiny",
    )
    easyocr_grounding_dino_prompt_default = getattr(
        cfg,
        "OCR_EASYOCR_GROUNDING_DINO_PROMPT",
        "text. paragraph. text block. message. chat bubble. dialog.",
    )
    easyocr_grounding_box_threshold_default = float(getattr(cfg, "OCR_EASYOCR_GROUNDING_BOX_THRESHOLD", 0.25))
    easyocr_grounding_text_threshold_default = float(getattr(cfg, "OCR_EASYOCR_GROUNDING_TEXT_THRESHOLD", 0.25))
    easyocr_grounding_max_regions_default = int(getattr(cfg, "OCR_EASYOCR_GROUNDING_MAX_REGIONS", 6))
    easyocr_grounding_padding_ratio_default = float(getattr(cfg, "OCR_EASYOCR_GROUNDING_PADDING_RATIO", 0.03))
    easyocr_grounding_text_aggregation_default = str(
        getattr(cfg, "OCR_EASYOCR_GROUNDING_TEXT_AGGREGATION", "concat")
    )
    transformers_model_default = getattr(cfg, "OCR_TRANSFORMERS_MODEL", DEFAULT_TRANSFORMERS_MODEL)
    transformers_task_prompt_default = getattr(
        cfg,
        "OCR_TRANSFORMERS_TASK_PROMPT",
        DEFAULT_TRANSFORMERS_TASK_PROMPT,
    )
    transformers_max_new_tokens_default = int(getattr(cfg, "OCR_TRANSFORMERS_MAX_NEW_TOKENS", 1024))
    transformers_num_beams_default = int(getattr(cfg, "OCR_TRANSFORMERS_NUM_BEAMS", 3))

    parser.add_argument(
        "--ocr-backend",
        choices=list(SUPPORTED_OCR_BACKENDS),
        default=backend_default,
        help="OCR backend to run through the pipeline.",
    )
    parser.add_argument(
        "--ollama-model",
        default=ollama_model_default,
        help="Ollama vision model used when --ocr-backend ollama.",
    )
    parser.add_argument(
        "--ollama-host",
        default=ollama_host_default,
        help="Base URL for the Ollama server when --ocr-backend ollama.",
    )
    parser.add_argument(
        "--ollama-timeout-seconds",
        type=int,
        default=ollama_timeout_default,
        help="Per-image Ollama request timeout when --ocr-backend ollama.",
    )
    parser.add_argument(
        "--ollama-disable-cleaning",
        action="store_true",
        help="Disable Ollama OCR post-cleaning and use the raw model response.",
    )
    parser.add_argument(
        "--easyocr-use-grounding-dino",
        action="store_true",
        default=easyocr_use_grounding_dino_default,
        help="For --ocr-backend easyocr, run Grounding DINO first to crop likely text regions before OCR.",
    )
    parser.add_argument(
        "--easyocr-grounding-dino-model",
        default=easyocr_grounding_dino_model_default,
        help="Grounding DINO model used before EasyOCR when --easyocr-use-grounding-dino is enabled.",
    )
    parser.add_argument(
        "--easyocr-grounding-dino-prompt",
        default=easyocr_grounding_dino_prompt_default,
        help="Lowercased dot-separated text prompt sent to Grounding DINO.",
    )
    parser.add_argument(
        "--easyocr-grounding-box-threshold",
        type=float,
        default=easyocr_grounding_box_threshold_default,
        help="Grounding DINO box threshold used before EasyOCR.",
    )
    parser.add_argument(
        "--easyocr-grounding-text-threshold",
        type=float,
        default=easyocr_grounding_text_threshold_default,
        help="Grounding DINO text threshold used before EasyOCR.",
    )
    parser.add_argument(
        "--easyocr-grounding-max-regions",
        type=int,
        default=easyocr_grounding_max_regions_default,
        help="Maximum number of Grounding DINO regions to OCR.",
    )
    parser.add_argument(
        "--easyocr-grounding-padding-ratio",
        type=float,
        default=easyocr_grounding_padding_ratio_default,
        help="Extra padding ratio added around each Grounding DINO crop before EasyOCR.",
    )
    parser.add_argument(
        "--easyocr-grounding-text-aggregation",
        choices=["concat", "max_model", "hybrid_max_model"],
        default=easyocr_grounding_text_aggregation_default,
        help="How grounded OCR crop texts are aggregated before text-model scoring.",
    )
    parser.add_argument(
        "--transformers-model",
        default=transformers_model_default,
        help="Transformers vision OCR model used when --ocr-backend transformers.",
    )
    parser.add_argument(
        "--transformers-task-prompt",
        default=transformers_task_prompt_default,
        help="Task prompt for the transformers OCR backend. Florence-2 OCR uses <OCR>.",
    )
    parser.add_argument(
        "--transformers-max-new-tokens",
        type=int,
        default=transformers_max_new_tokens_default,
        help="Max generated tokens for the transformers OCR backend.",
    )
    parser.add_argument(
        "--transformers-num-beams",
        type=int,
        default=transformers_num_beams_default,
        help="Beam count for deterministic transformers OCR decoding.",
    )
    parser.add_argument(
        "--transformers-disable-cleaning",
        action="store_true",
        help="Disable transformers OCR post-cleaning and use the raw model response.",
    )
    return parser


def build_ocr_service(cfg, args: argparse.Namespace) -> OCRService:
    ollama_disable_cleaning = getattr(args, "ollama_disable_cleaning", None)
    if ollama_disable_cleaning is None:
        ollama_clean_output = bool(getattr(cfg, "OCR_OLLAMA_CLEAN_OUTPUT", True))
    else:
        ollama_clean_output = not bool(ollama_disable_cleaning)

    transformers_disable_cleaning = getattr(args, "transformers_disable_cleaning", None)
    if transformers_disable_cleaning is None:
        transformers_clean_output = bool(getattr(cfg, "OCR_TRANSFORMERS_CLEAN_OUTPUT", True))
    else:
        transformers_clean_output = not bool(transformers_disable_cleaning)

    return OCRService(
        list(getattr(cfg, "OCR_LANGUAGES", ("en",))),
        gpu=(getattr(cfg, "DEVICE", "cpu") == "cuda"),
        backend=str(getattr(args, "ocr_backend", getattr(cfg, "OCR_BACKEND", "easyocr"))),
        ollama_model=str(getattr(args, "ollama_model", getattr(cfg, "OCR_OLLAMA_MODEL", "llama3.2-vision"))),
        ollama_host=str(getattr(args, "ollama_host", getattr(cfg, "OCR_OLLAMA_HOST", "http://localhost:11434"))),
        ollama_timeout_seconds=int(
            getattr(args, "ollama_timeout_seconds", getattr(cfg, "OCR_OLLAMA_TIMEOUT_SECONDS", 150))
        ),
        ollama_clean_output=ollama_clean_output,
        easyocr_use_grounding_dino=bool(
            getattr(args, "easyocr_use_grounding_dino", getattr(cfg, "OCR_EASYOCR_USE_GROUNDING_DINO", False))
        ),
        easyocr_grounding_dino_model=str(
            getattr(
                args,
                "easyocr_grounding_dino_model",
                getattr(cfg, "OCR_EASYOCR_GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny"),
            )
        ),
        easyocr_grounding_dino_prompt=str(
            getattr(
                args,
                "easyocr_grounding_dino_prompt",
                getattr(
                    cfg,
                    "OCR_EASYOCR_GROUNDING_DINO_PROMPT",
                    "text. paragraph. text block. message. chat bubble. dialog.",
                ),
            )
        ),
        easyocr_grounding_box_threshold=float(
            getattr(
                args,
                "easyocr_grounding_box_threshold",
                getattr(cfg, "OCR_EASYOCR_GROUNDING_BOX_THRESHOLD", 0.25),
            )
        ),
        easyocr_grounding_text_threshold=float(
            getattr(
                args,
                "easyocr_grounding_text_threshold",
                getattr(cfg, "OCR_EASYOCR_GROUNDING_TEXT_THRESHOLD", 0.25),
            )
        ),
        easyocr_grounding_max_regions=int(
            getattr(
                args,
                "easyocr_grounding_max_regions",
                getattr(cfg, "OCR_EASYOCR_GROUNDING_MAX_REGIONS", 6),
            )
        ),
        easyocr_grounding_padding_ratio=float(
            getattr(
                args,
                "easyocr_grounding_padding_ratio",
                getattr(cfg, "OCR_EASYOCR_GROUNDING_PADDING_RATIO", 0.03),
            )
        ),
        easyocr_grounding_text_aggregation=str(
            getattr(
                args,
                "easyocr_grounding_text_aggregation",
                getattr(cfg, "OCR_EASYOCR_GROUNDING_TEXT_AGGREGATION", "concat"),
            )
        ),
        transformers_model=str(
            getattr(args, "transformers_model", getattr(cfg, "OCR_TRANSFORMERS_MODEL", DEFAULT_TRANSFORMERS_MODEL))
        ),
        transformers_task_prompt=str(
            getattr(
                args,
                "transformers_task_prompt",
                getattr(cfg, "OCR_TRANSFORMERS_TASK_PROMPT", DEFAULT_TRANSFORMERS_TASK_PROMPT),
            )
        ),
        transformers_max_new_tokens=int(
            getattr(args, "transformers_max_new_tokens", getattr(cfg, "OCR_TRANSFORMERS_MAX_NEW_TOKENS", 1024))
        ),
        transformers_num_beams=int(
            getattr(args, "transformers_num_beams", getattr(cfg, "OCR_TRANSFORMERS_NUM_BEAMS", 3))
        ),
        transformers_clean_output=transformers_clean_output,
    )
