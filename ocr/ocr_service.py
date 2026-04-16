from __future__ import annotations

import base64
import json
import re
import warnings
from urllib import request

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import easyocr
except ImportError:  # pragma: no cover - exercised only in optional-install setups
    easyocr = None

DEFAULT_TRANSFORMERS_MODEL = "florence-community/Florence-2-base-ft"
DEFAULT_TRANSFORMERS_TASK_PROMPT = "<OCR>"
DEFAULT_EASYOCR_GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_EASYOCR_GROUNDING_DINO_PROMPT = "text. paragraph. text block. message. chat bubble. dialog."

DEFAULT_OLLAMA_PROMPT = (
    "Transcribe the visible text from this image as faithfully as possible.\n"
    "Do not paraphrase, summarize, translate, correct spelling, or explain.\n"
    "Preserve original wording, suspicious spelling, URLs, phone numbers, codes, amounts, usernames, and punctuation exactly as shown whenever possible.\n"
    "If the image is a chat, messaging app, SMS, WhatsApp, Telegram, iMessage, or similar conversation screenshot, "
    "return only the actual sender message body text from the conversation content.\n"
    "Exclude app UI and chrome such as header phone numbers, profile names outside the message body, timestamps, online/typing status, "
    "signal bars, battery, keyboard letters, input box text, 'message', 'report junk', 'pinned message', contact banners, "
    "'the sender is not in your contact list', encryption banners, and footer buttons.\n"
    "If it is not a chat screenshot, return all visible text in reading order.\n"
    "Return a JSON object with exactly one key: text.\n"
    'Example schema: {"text": "..."}\n'
    "The text value must contain only the transcription itself. No markdown. No quotes around the whole answer. "
    "No prefaces like 'The extracted text is'."
)

OLLAMA_SYSTEM_PROMPT = (
    "You are a strict OCR transcription engine. "
    "You must only return JSON matching the provided schema. "
    "Do not describe the image. Do not explain. Do not summarize."
)

OLLAMA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
    },
    "required": ["text"],
}

LEADING_WRAPPER_PATTERNS = (
    re.compile(r"^\s*the extracted text is\s*:?\s*", re.IGNORECASE),
    re.compile(r"^\s*the text in the image reads\s*:?\s*", re.IGNORECASE),
    re.compile(r"^\s*the image contains(?: the following)? text\s*:?\s*", re.IGNORECASE),
    re.compile(r"^\s*verbatim transcription\s*:?\s*", re.IGNORECASE),
    re.compile(r"^\s*transcription\s*:?\s*", re.IGNORECASE),
    re.compile(r"^\s*extracted text\s*:?\s*", re.IGNORECASE),
)

INLINE_UI_PHRASES = (
    "the sender is not in your contact list",
    "report junk",
)

UI_NOISE_PATTERNS = (
    re.compile(r"^\s*(report junk|message|messages|view profile|pinned message|learn more|block|delete|accept)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(online|typing|active now|last seen just now|business account|text message|sms|imessage)\s*$", re.IGNORECASE),
    re.compile(r"^\s*messages? and calls are secured with end-to-end encryption\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*only people in this chat can read,? listen,? or share them\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(today|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,2}:\d{2}\s*(?:AM|PM)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:\d{1,2}:\d{2}\s*(?:AM|PM)?\s+){2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:[A-Z]\s+){6,}[A-Z]?\s*$"),
    re.compile(r"^\s*(?:Q W E R T Y|A S D F G H J K L|Z X C V B N M).*$", re.IGNORECASE),
)

SUPPORTED_OCR_BACKENDS = ("easyocr", "ollama", "transformers")


class OCRService:
    def __init__(
        self,
        languages=None,
        gpu=False,
        backend: str = "easyocr",
        ollama_model: str = "llama3.2-vision",
        ollama_host: str = "http://localhost:11434",
        ollama_prompt: str = DEFAULT_OLLAMA_PROMPT,
        ollama_timeout_seconds: int = 300,
        ollama_clean_output: bool = True,
        easyocr_use_grounding_dino: bool = False,
        easyocr_grounding_dino_model: str = DEFAULT_EASYOCR_GROUNDING_DINO_MODEL,
        easyocr_grounding_dino_prompt: str = DEFAULT_EASYOCR_GROUNDING_DINO_PROMPT,
        easyocr_grounding_box_threshold: float = 0.25,
        easyocr_grounding_text_threshold: float = 0.25,
        easyocr_grounding_max_regions: int = 6,
        easyocr_grounding_padding_ratio: float = 0.03,
        transformers_model: str = DEFAULT_TRANSFORMERS_MODEL,
        transformers_task_prompt: str = DEFAULT_TRANSFORMERS_TASK_PROMPT,
        transformers_max_new_tokens: int = 1024,
        transformers_num_beams: int = 3,
        transformers_clean_output: bool = True,
    ):
        if languages is None:
            languages = ["en"]
        self.requested_languages = list(languages)
        self.active_languages = []
        self.load_error = None
        self.reader = None
        self.gpu = bool(gpu)
        self.backend = str(backend or "easyocr").strip().lower()
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.ollama_prompt = ollama_prompt
        self.ollama_timeout_seconds = ollama_timeout_seconds
        self.ollama_clean_output = ollama_clean_output
        self.easyocr_use_grounding_dino = bool(easyocr_use_grounding_dino)
        self.easyocr_grounding_dino_model = str(
            easyocr_grounding_dino_model or DEFAULT_EASYOCR_GROUNDING_DINO_MODEL
        ).strip()
        self.easyocr_grounding_dino_prompt = str(
            easyocr_grounding_dino_prompt or DEFAULT_EASYOCR_GROUNDING_DINO_PROMPT
        ).strip()
        self.easyocr_grounding_box_threshold = float(easyocr_grounding_box_threshold)
        self.easyocr_grounding_text_threshold = float(easyocr_grounding_text_threshold)
        self.easyocr_grounding_max_regions = int(easyocr_grounding_max_regions)
        self.easyocr_grounding_padding_ratio = float(easyocr_grounding_padding_ratio)
        self.easyocr_grounding_processor = None
        self.easyocr_grounding_model = None
        self.easyocr_grounding_device = "cuda" if self.gpu else "cpu"
        self.last_grounding_dino_regions = []
        self.transformers_model = str(transformers_model or DEFAULT_TRANSFORMERS_MODEL).strip()
        self.transformers_task_prompt = str(transformers_task_prompt or DEFAULT_TRANSFORMERS_TASK_PROMPT).strip()
        self.transformers_max_new_tokens = int(transformers_max_new_tokens)
        self.transformers_num_beams = int(transformers_num_beams)
        self.transformers_clean_output = bool(transformers_clean_output)
        self.transformers_processor = None
        self.transformers_ocr_model = None
        self.transformers_device = "cuda" if self.gpu else "cpu"
        self.transformers_torch_dtype = None

        if self.backend not in SUPPORTED_OCR_BACKENDS:
            raise ValueError(
                f"Unsupported OCR backend: {self.backend}. "
                f"Choose from: {', '.join(SUPPORTED_OCR_BACKENDS)}."
            )

        if self.backend == "ollama":
            self.active_languages = [f"ollama:{self.ollama_model}"]
            return

        if self.backend == "transformers":
            self.active_languages = [f"transformers:{self.transformers_model}"]
            return

        if easyocr is None:
            self.load_error = (
                "easyocr is not installed. Install it or switch to backend='ollama' or backend='transformers'."
            )
            return

        try:
            self.reader = easyocr.Reader(self.requested_languages, gpu=gpu)
            self.active_languages = list(self.requested_languages)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.load_error = str(exc)
            fallback_languages = ["en"] if "en" in self.requested_languages else None
            if fallback_languages and fallback_languages != self.requested_languages:
                self.reader = easyocr.Reader(fallback_languages, gpu=gpu)
                self.active_languages = fallback_languages
            else:
                self.reader = None

    def extract_text(self, image_path: str) -> str:
        if self.backend == "ollama":
            return self._extract_text_ollama(image_path)

        if self.backend == "transformers":
            return self._extract_text_transformers(image_path)

        return self._extract_text_easyocr(image_path)

    def _extract_text_easyocr(self, image_path: str) -> str:
        if self.reader is None:
            return ""

        try:
            original_image = self._load_image_rgb(image_path)
        except Exception:
            return ""

        original_text = self._safe_readtext(np.array(original_image))
        processed_image = self._preprocess_for_ocr(original_image)
        processed_text = self._safe_readtext(processed_image)

        candidates = [original_text, processed_text]
        self.last_grounding_dino_regions = []
        if self.easyocr_use_grounding_dino:
            grounded_text = self._extract_text_easyocr_grounded(original_image)
            if grounded_text:
                candidates.append(grounded_text)

        return self._pick_best_texts(*candidates)

    def _load_image_rgb(self, image_path: str):
        return Image.open(image_path).convert("RGB")

    def _preprocess_for_ocr(self, image):
        if not isinstance(image, Image.Image):
            image = self._load_image_rgb(image)

        # Upscale small screenshot text so the OCR recognizer gets clearer glyphs.
        width, height = image.size
        image = image.resize((max(width * 2, 1), max(height * 2, 1)), Image.Resampling.LANCZOS)

        grayscale = ImageOps.grayscale(image)
        grayscale = ImageOps.autocontrast(grayscale)
        grayscale = ImageEnhance.Contrast(grayscale).enhance(1.8)
        grayscale = ImageEnhance.Sharpness(grayscale).enhance(2.2)
        grayscale = grayscale.filter(ImageFilter.MedianFilter(size=3))

        return np.array(grayscale)

    def _safe_readtext(self, image) -> str:
        try:
            return self._join_results(self.reader.readtext(image, detail=0))
        except Exception:
            return ""

    def _join_results(self, results) -> str:
        return " ".join(str(item).strip() for item in results if str(item).strip())

    def _pick_best_texts(self, *texts: str) -> str:
        candidates = [str(text).strip() for text in texts]
        candidates = [text for text in candidates if text]
        if not candidates:
            return ""
        return max(candidates, key=self._text_quality_score)

    def _text_quality_score(self, text: str) -> tuple:
        normalized = text.strip()
        alnum_count = sum(char.isalnum() for char in normalized)
        word_count = len([part for part in normalized.split() if part])
        unique_chars = len(set(normalized.lower()))
        suspicious_symbols = len(re.findall(r"[^A-Za-z0-9\s\u4e00-\u9fff.,:;!?$%&@()+/\-_'\"]", normalized))
        return (
            alnum_count - suspicious_symbols * 2,
            word_count,
            unique_chars,
            -len(normalized),
        )

    def _extract_text_easyocr_grounded(self, image: Image.Image) -> str:
        try:
            regions = self._detect_grounding_dino_regions(image)
        except Exception as exc:
            self.load_error = str(exc)
            return ""

        self.last_grounding_dino_regions = regions
        if not regions:
            return ""

        texts = []
        for region in regions:
            crop = self._crop_image_with_box(
                image,
                region["box"],
                padding_ratio=self.easyocr_grounding_padding_ratio,
            )
            crop_text = self._extract_text_from_image_variants(crop)
            if crop_text:
                texts.append(crop_text)

        return self._join_unique_text_segments(texts)

    def _extract_text_from_image_variants(self, image: Image.Image) -> str:
        original_text = self._safe_readtext(np.array(image))
        processed_image = self._preprocess_for_ocr(image)
        processed_text = self._safe_readtext(processed_image)
        return self._pick_best_texts(original_text, processed_text)

    def _join_unique_text_segments(self, texts) -> str:
        lines = []
        seen = set()
        for text in texts:
            normalized = str(text or "").strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(normalized)
        return "\n".join(lines).strip()

    def _ensure_easyocr_grounding_dino_ready(self) -> None:
        if self.easyocr_grounding_processor is not None and self.easyocr_grounding_model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:  # pragma: no cover - dependency failure is environment-specific
            raise RuntimeError(
                "EasyOCR Grounding DINO mode requires transformers zero-shot object detection support."
            ) from exc

        self.easyocr_grounding_device = "cuda" if self.gpu and torch.cuda.is_available() else "cpu"
        self.easyocr_grounding_processor = AutoProcessor.from_pretrained(
            self.easyocr_grounding_dino_model,
            use_fast=False,
        )
        self.easyocr_grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.easyocr_grounding_dino_model,
        ).to(self.easyocr_grounding_device)
        self.easyocr_grounding_model.eval()

    def _detect_grounding_dino_regions(self, image: Image.Image) -> list[dict[str, object]]:
        import torch

        self._ensure_easyocr_grounding_dino_ready()
        assert self.easyocr_grounding_processor is not None
        assert self.easyocr_grounding_model is not None

        prompt_labels = self._grounding_dino_prompt_labels(self.easyocr_grounding_dino_prompt)
        prompt = self._normalize_grounding_dino_prompt(self.easyocr_grounding_dino_prompt)
        inputs = self.easyocr_grounding_processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )
        prepared_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                prepared_inputs[key] = value.to(self.easyocr_grounding_device)
            else:
                prepared_inputs[key] = value

        with torch.inference_mode():
            outputs = self.easyocr_grounding_model(**prepared_inputs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            results = self.easyocr_grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.easyocr_grounding_box_threshold,
                text_threshold=self.easyocr_grounding_text_threshold,
                target_sizes=[image.size[::-1]],
                text_labels=[prompt_labels],
            )
        if not results:
            return []

        image_width, image_height = image.size
        raw_regions = []
        result = results[0]
        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("text_labels") or result.get("labels")
        if boxes is None or scores is None or labels is None:
            return []

        for box_tensor, score_tensor, label_value in zip(boxes, scores, labels):
            box = [int(round(float(value))) for value in box_tensor.tolist()]
            clipped_box = self._clip_box_to_image(box, image_width, image_height)
            if clipped_box is None:
                continue
            x0, y0, x1, y1 = clipped_box
            box_width = x1 - x0
            box_height = y1 - y0
            if box_width < 16 or box_height < 16:
                continue

            raw_regions.append(
                {
                    "box": clipped_box,
                    "score": float(score_tensor),
                    "label": str(label_value or "").strip(),
                }
            )

        merged_regions = self._merge_grounding_dino_regions(raw_regions)
        selected_regions = merged_regions[:max(self.easyocr_grounding_max_regions, 1)]
        selected_regions.sort(key=lambda item: (item["box"][1], item["box"][0]))
        return selected_regions

    def _normalize_grounding_dino_prompt(self, prompt: str) -> str:
        parts = self._grounding_dino_prompt_labels(prompt)
        return " ".join(f"{part}." for part in parts)

    def _grounding_dino_prompt_labels(self, prompt: str) -> list[str]:
        parts = []
        for raw_part in str(prompt or "").split("."):
            token = raw_part.strip().lower()
            if token:
                parts.append(token)
        if not parts:
            parts = ["text", "message", "dialog"]
        return parts

    def _merge_grounding_dino_regions(self, regions) -> list[dict[str, object]]:
        if not regions:
            return []

        ranked = sorted(
            regions,
            key=lambda item: (float(item["score"]), self._box_area(item["box"])),
            reverse=True,
        )
        merged: list[dict[str, object]] = []
        for region in ranked:
            matched = False
            for existing in merged:
                if self._should_merge_boxes(existing["box"], region["box"]):
                    existing["box"] = self._union_boxes(existing["box"], region["box"])
                    existing["score"] = max(float(existing["score"]), float(region["score"]))
                    labels = set(existing.get("labels") or [])
                    if existing.get("label"):
                        labels.add(str(existing["label"]))
                    if region.get("label"):
                        labels.add(str(region["label"]))
                    existing["labels"] = sorted(labels)
                    matched = True
                    break
            if not matched:
                merged.append(dict(region))

        merged.sort(
            key=lambda item: (float(item["score"]), self._box_area(item["box"])),
            reverse=True,
        )
        return merged

    def _should_merge_boxes(self, first_box, second_box) -> bool:
        iou = self._box_iou(first_box, second_box)
        if iou >= 0.25:
            return True

        intersection = self._intersection_area(first_box, second_box)
        if intersection <= 0:
            return False

        smaller_area = max(min(self._box_area(first_box), self._box_area(second_box)), 1)
        return (intersection / smaller_area) >= 0.6

    def _box_iou(self, first_box, second_box) -> float:
        intersection = self._intersection_area(first_box, second_box)
        if intersection <= 0:
            return 0.0
        union = self._box_area(first_box) + self._box_area(second_box) - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _intersection_area(self, first_box, second_box) -> int:
        x0 = max(first_box[0], second_box[0])
        y0 = max(first_box[1], second_box[1])
        x1 = min(first_box[2], second_box[2])
        y1 = min(first_box[3], second_box[3])
        if x1 <= x0 or y1 <= y0:
            return 0
        return int((x1 - x0) * (y1 - y0))

    def _box_area(self, box) -> int:
        return max(int(box[2] - box[0]), 0) * max(int(box[3] - box[1]), 0)

    def _union_boxes(self, first_box, second_box) -> list[int]:
        return [
            int(min(first_box[0], second_box[0])),
            int(min(first_box[1], second_box[1])),
            int(max(first_box[2], second_box[2])),
            int(max(first_box[3], second_box[3])),
        ]

    def _clip_box_to_image(self, box, image_width: int, image_height: int) -> list[int] | None:
        x0, y0, x1, y1 = [int(value) for value in box]
        x0 = max(0, min(x0, image_width - 1))
        y0 = max(0, min(y0, image_height - 1))
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    def _crop_image_with_box(self, image: Image.Image, box, padding_ratio: float = 0.0) -> Image.Image:
        image_width, image_height = image.size
        x0, y0, x1, y1 = [int(value) for value in box]
        padding_x = int(round((x1 - x0) * max(padding_ratio, 0.0)))
        padding_y = int(round((y1 - y0) * max(padding_ratio, 0.0)))
        clipped_box = self._clip_box_to_image(
            [x0 - padding_x, y0 - padding_y, x1 + padding_x, y1 + padding_y],
            image_width,
            image_height,
        )
        if clipped_box is None:
            return image.copy()
        return image.crop(tuple(clipped_box))

    def _extract_text_ollama(self, image_path: str) -> str:
        try:
            raw = self._call_ollama(image_path)
            return self._clean_vlm_output(raw) if self.ollama_clean_output else raw
        except Exception as exc:
            self.load_error = str(exc)
            return ""

    def _extract_text_transformers(self, image_path: str) -> str:
        try:
            self._ensure_transformers_ready()
            image = self._load_image_rgb(image_path)
            raw = self._call_transformers(image)
            return self._clean_vlm_output(raw) if self.transformers_clean_output else raw
        except Exception as exc:
            self.load_error = str(exc)
            return ""

    def _call_ollama(self, image_path: str) -> str:
        with open(image_path, "rb") as handle:
            image_b64 = base64.b64encode(handle.read()).decode("utf-8")
        endpoint = self.ollama_host.rstrip("/") + "/api/chat"
        payload = {
            "model": self.ollama_model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": OLLAMA_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": self.ollama_prompt,
                    "images": [image_b64],
                },
            ],
            "format": OLLAMA_RESPONSE_SCHEMA,
            "options": {
                "temperature": 0,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.ollama_timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        message = response_payload.get("message") or {}
        content = str(message.get("content") or response_payload.get("response") or "").strip()
        try:
            parsed = json.loads(content)
            text_value = parsed.get("text")
            if isinstance(text_value, str):
                return text_value.strip()
        except json.JSONDecodeError:
            pass
        return content

    def _ensure_transformers_ready(self) -> None:
        if self.transformers_processor is not None and self.transformers_ocr_model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - dependency failure is environment-specific
            raise RuntimeError(
                "The transformers OCR backend requires the 'transformers' package with Florence-2 support installed."
            ) from exc

        self.transformers_device = "cuda" if self.gpu and torch.cuda.is_available() else "cpu"
        self.transformers_torch_dtype = torch.float16 if self.transformers_device == "cuda" else torch.float32
        self.transformers_ocr_model = Florence2ForConditionalGeneration.from_pretrained(
            self.transformers_model,
            torch_dtype=self.transformers_torch_dtype,
        ).to(self.transformers_device)
        self.transformers_ocr_model.eval()
        self.transformers_processor = AutoProcessor.from_pretrained(
            self.transformers_model,
            use_fast=False,
        )

    def _call_transformers(self, image: Image.Image) -> str:
        import torch

        assert self.transformers_processor is not None
        assert self.transformers_ocr_model is not None

        inputs = self.transformers_processor(
            text=self.transformers_task_prompt,
            images=image,
            return_tensors="pt",
        )
        prepared_inputs = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                prepared_inputs[key] = value
                continue

            value_dtype = getattr(value, "dtype", None)
            if value_dtype is not None and value_dtype.is_floating_point:
                prepared_inputs[key] = value.to(
                    device=self.transformers_device,
                    dtype=self.transformers_torch_dtype,
                )
            else:
                prepared_inputs[key] = value.to(self.transformers_device)

        with torch.inference_mode():
            generated_ids = self.transformers_ocr_model.generate(
                **prepared_inputs,
                do_sample=False,
                max_new_tokens=self.transformers_max_new_tokens,
                num_beams=self.transformers_num_beams,
            )

        generated_text = self.transformers_processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        try:
            parsed = self.transformers_processor.post_process_generation(
                generated_text,
                task=self.transformers_task_prompt,
                image_size=image.size,
            )
        except Exception:
            return generated_text

        return self._extract_transformers_text(parsed, generated_text)

    def _extract_transformers_text(self, parsed, raw_text: str) -> str:
        if isinstance(parsed, str):
            return parsed.strip()

        task_value = None
        if isinstance(parsed, dict):
            task_value = parsed.get(self.transformers_task_prompt)
            if task_value is None and len(parsed) == 1:
                task_value = next(iter(parsed.values()))

        if isinstance(task_value, str):
            return task_value.strip()

        if isinstance(task_value, list):
            parts = [str(item).strip() for item in task_value if str(item).strip()]
            if parts:
                return "\n".join(parts)

        if isinstance(task_value, dict):
            text_value = task_value.get("text")
            if isinstance(text_value, str):
                return text_value.strip()
            labels = task_value.get("labels")
            if isinstance(labels, list):
                parts = [str(item).strip() for item in labels if str(item).strip()]
                if parts:
                    return "\n".join(parts)

        return str(raw_text or "").strip()

    def _clean_vlm_output(self, text: str) -> str:
        cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        cleaned = cleaned.replace("</s>", " ").replace("<s>", " ")
        cleaned = cleaned.replace(self.transformers_task_prompt, " ")
        cleaned = re.sub(r"^\s*```(?:text)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)

        for pattern in LEADING_WRAPPER_PATTERNS:
            cleaned = pattern.sub("", cleaned).strip()

        if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
            cleaned = cleaned[1:-1].strip()

        lines: list[str] = []
        for raw_line in cleaned.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            for phrase in INLINE_UI_PHRASES:
                line = re.sub(re.escape(phrase), "", line, flags=re.IGNORECASE).strip(" -|:")

            line = re.sub(r"(?:\s*\.\s*){2,}", ".", line)
            line = re.sub(r"\s+([,.;:!?])", r"\1", line)
            line = re.sub(r"\s{2,}", " ", line).strip()
            if not line:
                continue

            if re.fullmatch(r"[.\-_|:;,\s]+", line):
                continue

            if any(pattern.match(line) for pattern in UI_NOISE_PATTERNS):
                continue

            if lines and line == lines[-1]:
                continue

            lines.append(line)

        result = "\n".join(lines).strip()
        for pattern in LEADING_WRAPPER_PATTERNS:
            result = pattern.sub("", result).strip()
        return result

    def runtime_details(self) -> dict[str, object]:
        if self.backend == "easyocr":
            details = {}
            if self.easyocr_use_grounding_dino:
                details.update(
                    {
                        "EasyOCR Grounding DINO Enabled": True,
                        "EasyOCR Grounding DINO Model": self.easyocr_grounding_dino_model,
                        "EasyOCR Grounding DINO Prompt": self._normalize_grounding_dino_prompt(
                            self.easyocr_grounding_dino_prompt
                        ),
                        "EasyOCR Grounding Box Threshold": self.easyocr_grounding_box_threshold,
                        "EasyOCR Grounding Text Threshold": self.easyocr_grounding_text_threshold,
                        "EasyOCR Grounding Max Regions": self.easyocr_grounding_max_regions,
                        "EasyOCR Grounding Padding Ratio": self.easyocr_grounding_padding_ratio,
                        "EasyOCR Grounding Device": self.easyocr_grounding_device,
                        "EasyOCR Grounding Regions Found": len(self.last_grounding_dino_regions),
                    }
                )
            return details

        if self.backend == "ollama":
            return {
                "Ollama Model": self.ollama_model,
                "Ollama Host": self.ollama_host,
                "Ollama Timeout Seconds": self.ollama_timeout_seconds,
                "Ollama Cleaning Enabled": self.ollama_clean_output,
            }

        if self.backend == "transformers":
            return {
                "Transformers Model": self.transformers_model,
                "Transformers Task Prompt": self.transformers_task_prompt,
                "Transformers Max New Tokens": self.transformers_max_new_tokens,
                "Transformers Num Beams": self.transformers_num_beams,
                "Transformers Cleaning Enabled": self.transformers_clean_output,
                "Transformers Device": self.transformers_device,
            }

        return {}
