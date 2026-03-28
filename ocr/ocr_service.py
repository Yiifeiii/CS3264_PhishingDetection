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

    def extract_text(self, image_path: str) -> str:
        if self.reader is None:
            return ""

        original_text = self._join_results(self.reader.readtext(image_path, detail=0))
        processed_image = self._preprocess_for_ocr(image_path)
        processed_text = self._join_results(self.reader.readtext(processed_image, detail=0))

        return self._pick_best_text(original_text, processed_text)

    def _preprocess_for_ocr(self, image_path: str):
        image = Image.open(image_path).convert("RGB")

        # Upscale small screenshot text so the OCR recognizer gets clearer glyphs.
        width, height = image.size
        image = image.resize((max(width * 2, 1), max(height * 2, 1)), Image.Resampling.LANCZOS)

        grayscale = ImageOps.grayscale(image)
        grayscale = ImageOps.autocontrast(grayscale)
        grayscale = ImageEnhance.Contrast(grayscale).enhance(1.8)
        grayscale = ImageEnhance.Sharpness(grayscale).enhance(2.2)
        grayscale = grayscale.filter(ImageFilter.MedianFilter(size=3))

        return np.array(grayscale)

    def _join_results(self, results) -> str:
        return " ".join(str(item).strip() for item in results if str(item).strip())

    def _pick_best_text(self, original_text: str, processed_text: str) -> str:
        candidates = [original_text.strip(), processed_text.strip()]
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
