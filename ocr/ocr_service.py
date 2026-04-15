import re

import easyocr
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

class OCRService:
    def __init__(self, languages=None, gpu=False):
        if languages is None:
            languages = ["en"]
        self.requested_languages = list(languages)
        self.active_languages = []
        self.load_error = None
        self.reader = None

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

    def extract_text(self, image_path: str, skip_preprocess: bool = False) -> str:
        if self.reader is None:
            return ""

        try:
            original_image = self._load_image_rgb(image_path)
        except Exception:
            return ""

        original_text = self._safe_readtext(np.array(original_image))

        if skip_preprocess:
            return original_text.strip()

        processed_image = self._preprocess_for_ocr(original_image)
        processed_text = self._safe_readtext(processed_image)

        return self._pick_best_text(original_text, processed_text)

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

    def _pick_best_text(self, original_text: str, processed_text: str) -> str:
        candidates = [original_text.strip(), processed_text.strip()]
        candidates = [text for text in candidates if text]
        if not candidates:
            return ""
        return max(candidates, key=self._text_quality_score)

    def detect_text_regions(
        self,
        image_path: str,
        min_confidence: float = 0.0,
        include_processed: bool = True,
    ):
        if self.reader is None:
            return []

        regions = []

        try:
            original_image = self._load_image_rgb(image_path)
        except Exception:
            return []

        original_results = self.reader.readtext(np.array(original_image), detail=1)
        regions.extend(
            self._results_to_regions(
                original_results,
                scale_x=1.0,
                scale_y=1.0,
                source="original",
                min_confidence=min_confidence,
            )
        )

        if include_processed:
            processed_image = self._preprocess_for_ocr(original_image)
            processed_height, processed_width = processed_image.shape[:2]
            processed_results = self.reader.readtext(processed_image, detail=1)
            regions.extend(
                self._results_to_regions(
                    processed_results,
                    scale_x=processed_width / max(original_image.width, 1),
                    scale_y=processed_height / max(original_image.height, 1),
                    source="processed",
                    min_confidence=min_confidence,
                )
            )

        return regions

    def _results_to_regions(self, results, scale_x: float, scale_y: float, source: str, min_confidence: float):
        regions = []
        for result in results:
            if not isinstance(result, (list, tuple)) or len(result) < 3:
                continue

            polygon, text, confidence = result[:3]
            text = str(text or "").strip()
            confidence = float(confidence or 0.0)
            if not text or confidence < min_confidence:
                continue

            bbox = self._polygon_to_bbox(polygon, scale_x=scale_x, scale_y=scale_y)
            width = max(bbox["x2"] - bbox["x1"], 0)
            height = max(bbox["y2"] - bbox["y1"], 0)
            if width == 0 or height == 0:
                continue

            regions.append(
                {
                    "bbox": [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
                    "text": text,
                    "confidence": confidence,
                    "source": source,
                    "width": width,
                    "height": height,
                    "area": width * height,
                }
            )

        return regions

    def _polygon_to_bbox(self, polygon, scale_x: float, scale_y: float):
        xs = []
        ys = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            xs.append(float(point[0]) / max(scale_x, 1e-8))
            ys.append(float(point[1]) / max(scale_y, 1e-8))

        if not xs or not ys:
            return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

        return {
            "x1": int(round(min(xs))),
            "y1": int(round(min(ys))),
            "x2": int(round(max(xs))),
            "y2": int(round(max(ys))),
        }

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
