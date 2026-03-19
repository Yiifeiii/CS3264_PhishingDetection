import re
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TextModelService:
    def __init__(self, model_name: str, device: str, positive_class_index: Optional[int] = None):
        self.model_name = model_name
        self.device = device
        self.positive_class_index = positive_class_index
        self.tokenizer = None
        self.model = None
        self.id2label = {}
        self.is_loaded = False
        self.load_error = None
        self._load_model()

    def _load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.id2label = getattr(self.model.config, "id2label", {}) or {}
            self.is_loaded = True
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.load_error = str(exc)
            self.tokenizer = None
            self.model = None
            self.id2label = {}
            self.is_loaded = False

    def predict_phishing_probability(self, text: str) -> Optional[float]:
        normalized = (text or "").strip()
        if not self.is_loaded or not normalized:
            return None

        with torch.no_grad():
            encoded = self.tokenizer(
                normalized,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            ).to(self.device)
            logits = self.model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=0).cpu().tolist()

        if not probs:
            return None

        phishing_index = self._resolve_phishing_index(len(probs))
        if phishing_index is None:
            phishing_index = 1 if len(probs) >= 2 else 0

        prob = float(probs[phishing_index])
        return max(0.0, min(prob, 1.0))

    def _resolve_phishing_index(self, class_count: int) -> Optional[int]:
        if self.positive_class_index is not None and 0 <= self.positive_class_index < class_count:
            return self.positive_class_index

        if not self.id2label:
            return None

        phishing_tokens = ("phish", "spam", "scam", "malicious", "fraud", "suspicious", "1", "positive")
        safe_tokens = ("safe", "legit", "ham", "benign", "non", "0", "negative")

        best_idx = None
        best_score = -1

        for idx in range(class_count):
            label = str(self.id2label.get(idx, "")).lower()
            label_norm = re.sub(r"[^a-z0-9]+", " ", label).strip()
            if not label_norm:
                continue

            score = 0
            if any(tok in label_norm for tok in phishing_tokens):
                score += 2
            if any(tok in label_norm for tok in safe_tokens):
                score -= 2

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx if best_score > 0 else None
