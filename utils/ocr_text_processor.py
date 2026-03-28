import re
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianTokenizer


class OCRTextProcessor:
    CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
    MULTISPACE_PATTERN = re.compile(r"\s+")
    TOKEN_PATTERN = re.compile(r"\S+")

    def __init__(self, cfg):
        self.cfg = cfg
        self.policy = getattr(cfg, "OCR_CHINESE_POLICY", "skip")
        self.model_name = getattr(cfg, "CHINESE_TO_ENGLISH_MODEL_NAME", "")
        self.device = getattr(cfg, "DEVICE", "cpu")
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.load_error: Optional[str] = None

        if self.policy == "translate":
            self._load_translation_model()

    def process(self, text: str) -> dict:
        normalized = (text or "").strip()
        if not normalized:
            return {
                "text": "",
                "contains_chinese": False,
                "action": "empty",
                "warning": None,
            }

        contains_chinese = bool(self.CHINESE_PATTERN.search(normalized))
        if not contains_chinese:
            cleaned = self._clean_ocr_text(normalized)
            return {
                "text": cleaned,
                "contains_chinese": False,
                "action": "cleaned",
                "warning": None,
            }

        if self.policy == "skip":
            return {
                "text": "",
                "contains_chinese": True,
                "action": "skipped_chinese",
                "warning": "Chinese text detected and skipped by OCR policy.",
            }

        if self.policy == "strip":
            stripped = self._strip_chinese(normalized)
            stripped = self._clean_ocr_text(stripped)
            if stripped:
                return {
                    "text": stripped,
                    "contains_chinese": True,
                    "action": "stripped_chinese",
                    "warning": "Chinese characters removed before text scoring.",
                }
            return {
                "text": "",
                "contains_chinese": True,
                "action": "strip_empty",
                "warning": "Chinese characters removed, but no usable non-Chinese text remained.",
            }

        translated = self._translate(normalized)
        translated = self._clean_ocr_text(translated or "")
        if translated:
            return {
                "text": translated,
                "contains_chinese": True,
                "action": "translated",
                "warning": None,
            }

        return {
            "text": "",
            "contains_chinese": True,
            "action": "translation_unavailable",
            "warning": self.load_error or "Chinese translation failed.",
        }

    def _load_translation_model(self):
        try:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception:
                # Some Marian checkpoints need the explicit tokenizer class.
                self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.load_error = str(exc)
            self.tokenizer = None
            self.model = None
            self.is_loaded = False

    def _translate(self, text: str) -> Optional[str]:
        if not self.is_loaded or not text:
            return None

        with torch.no_grad():
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            ).to(self.device)
            output = self.model.generate(**encoded, max_new_tokens=256)

        translated = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        if not translated:
            return None

        normalized = translated[0].strip()
        return normalized or None

    def _strip_chinese(self, text: str) -> str:
        without_chinese = self.CHINESE_PATTERN.sub(" ", text)
        return self.MULTISPACE_PATTERN.sub(" ", without_chinese).strip()

    def _clean_ocr_text(self, text: str) -> str:
        if not text:
            return ""

        kept_tokens = []
        for token in self.TOKEN_PATTERN.findall(text):
            cleaned = self._clean_token(token)
            if cleaned:
                kept_tokens.append(cleaned)

        normalized = " ".join(kept_tokens)
        normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
        return self.MULTISPACE_PATTERN.sub(" ", normalized).strip()

    def _clean_token(self, token: str) -> str:
        token = token.strip()
        if not token:
            return ""

        # Keep URLs, emails, and phone-like tokens intact since they are strong phishing signals.
        lowered = token.lower()
        if any(marker in lowered for marker in ("http://", "https://", "www.", "@")):
            return token
        if re.fullmatch(r"[+\d][\d\s-]{5,}", token):
            return token

        cleaned = re.sub(r"^[^A-Za-z0-9$+]+|[^A-Za-z0-9$+]+$", "", token)
        if not cleaned:
            return ""

        alnum_count = sum(char.isalnum() for char in cleaned)
        alpha_count = sum(char.isalpha() for char in cleaned)
        digit_count = sum(char.isdigit() for char in cleaned)
        symbol_count = len(cleaned) - alnum_count

        if alnum_count == 0:
            return ""
        if len(cleaned) <= 2 and alpha_count == 0:
            return ""
        if len(cleaned) <= 3 and alpha_count <= 1 and digit_count >= 1:
            return ""
        if symbol_count > max(2, alnum_count):
            return ""

        return cleaned
