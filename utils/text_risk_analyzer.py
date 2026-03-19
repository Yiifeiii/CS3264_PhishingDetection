import re

from utils.text_model_service import TextModelService


class TextRiskAnalyzer:
    URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(?:com|net|org|io|co|ru|cn|xyz)\b)")
    EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
    PHONE_PATTERN = re.compile(r"(\+\d{1,3}[\s-]?\d{4,14}|\b\d{8,14}\b)")
    MONEY_PATTERN = re.compile(r"(\$\s?\d+(?:[\.,]\d{1,2})?|\b\d+(?:[\.,]\d{1,2})?\s?(?:usd|sgd|rm)\b)", re.IGNORECASE)

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = TextModelService(
            cfg.TEXT_PHISHING_MODEL_NAME,
            cfg.DEVICE,
            cfg.TEXT_MODEL_POSITIVE_CLASS_INDEX,
        )

    def analyze(self, text: str):
        normalized = (text or "").strip()
        lowered = normalized.lower()
        denoised = self._denoise_for_keywords(lowered)

        matched_groups = {}
        total_hits = 0

        for group_name, keywords in self.cfg.PHISHING_KEYWORDS.items():
            hits = [keyword for keyword in keywords if keyword in lowered or keyword in denoised]
            if hits:
                matched_groups[group_name] = hits
                total_hits += len(hits)

        urls = self.URL_PATTERN.findall(normalized)
        emails = self.EMAIL_PATTERN.findall(normalized)
        phone_numbers = self.PHONE_PATTERN.findall(normalized)
        money_mentions = self.MONEY_PATTERN.findall(normalized)

        rule_score = 0.0
        rule_score += min(total_hits * self.cfg.KEYWORD_HIT_WEIGHT, 0.6)

        if urls:
            rule_score += self.cfg.URL_PRESENT_WEIGHT
        if emails:
            rule_score += self.cfg.EMAIL_PRESENT_WEIGHT
        if phone_numbers:
            rule_score += self.cfg.PHONE_PRESENT_WEIGHT
        if money_mentions:
            rule_score += self.cfg.MONEY_PRESENT_WEIGHT

        signal_families = sum([
            bool(matched_groups),
            bool(urls),
            bool(emails),
            bool(phone_numbers),
            bool(money_mentions),
        ])
        if signal_families >= 3:
            rule_score += self.cfg.MULTI_SIGNAL_BONUS
        if ("credential_request" in matched_groups) and (urls or emails):
            rule_score += self.cfg.LINK_CREDENTIAL_BONUS

        if not normalized:
            rule_score *= self.cfg.EMPTY_TEXT_PENALTY

        rule_score = min(rule_score, 1.0)

        model_chunks = self._split_text_for_model(normalized)
        chunk_scores = []
        for chunk in model_chunks:
            prob = self.model.predict_phishing_probability(chunk)
            if prob is not None:
                chunk_scores.append((chunk, prob))

        if chunk_scores:
            best_chunk, model_score_raw = max(chunk_scores, key=lambda x: x[1])
        else:
            best_chunk, model_score_raw = (None, None)

        model_score = self._calibrate_model_score(model_score_raw)

        if model_score is None:
            score = rule_score
        else:
            # Model signal is always included; rules are not used as a hard gate.
            score = (
                self.cfg.TEXT_RULE_WEIGHT * rule_score
                + self.cfg.TEXT_MODEL_WEIGHT * model_score
            )
        score = min(score, 1.0)

        reasons = []
        if matched_groups:
            reasons.append("phishing language detected")
        if urls:
            reasons.append("URL detected in image text")
        if emails:
            reasons.append("email address detected in image text")
        if phone_numbers:
            reasons.append("phone number detected in image text")
        if money_mentions:
            reasons.append("money-related language detected")
        if signal_families >= 3:
            reasons.append("multiple independent phishing signal types detected")
        if model_score_raw is not None:
            reasons.append(
                f"Cybersectony model probability: {model_score_raw:.2f} "
                f"(calibrated: {model_score:.2f})"
            )
            if best_chunk:
                reasons.append(f"model strongest text span: '{best_chunk[:90]}'")
        elif self.model.is_loaded:
            reasons.append("text model loaded but no usable text for inference")
        else:
            reasons.append("text model unavailable; heuristic-only text scoring")
        warnings = []
        if model_score is not None and model_score >= 0.6 and rule_score < 0.18:
            warnings.append("model indicates phishing but rule signals are weak")
        if model_score is not None and model_score < 0.2 and rule_score >= 0.4:
            warnings.append("rule signals are strong but model confidence is low")
        if warnings:
            reasons.extend([f"warning: {w}" for w in warnings])
        if not reasons:
            reasons.append("no obvious phishing text signals found")

        return {
            "score": round(score, 4),
            "rule_score": round(rule_score, 4),
            "model_score": None if model_score is None else round(model_score, 4),
            "model_score_raw": None if model_score_raw is None else round(model_score_raw, 4),
            "model_loaded": self.model.is_loaded,
            "model_chunks_evaluated": len(chunk_scores),
            "model_best_chunk": best_chunk,
            "warnings": warnings,
            "keyword_groups": matched_groups,
            "urls": urls,
            "emails": emails,
            "phone_numbers": phone_numbers,
            "money_mentions": money_mentions,
            "reasons": reasons,
        }

    def _denoise_for_keywords(self, text: str) -> str:
        char_map = str.maketrans({
            "0": "o",
            "1": "i",
            "!": "i",
            "|": "l",
            "$": "s",
            "5": "s",
            "@": "a",
            "3": "e",
            "7": "t",
        })
        mapped = text.translate(char_map)
        cleaned = re.sub(r"[^a-z0-9\s]", " ", mapped)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _calibrate_model_score(self, score):
        if score is None:
            return None
        threshold = self.cfg.MODEL_POSITIVE_THRESHOLD
        if score <= threshold:
            return 0.0
        return min((score - threshold) / (1.0 - threshold), 1.0)

    def _split_text_for_model(self, text: str):
        normalized = (text or "").strip()
        if not normalized:
            return []

        chunks = [normalized]

        # Split into sentence-like segments first.
        segments = [
            seg.strip()
            for seg in re.split(r"[.!?;\n]+", normalized)
            if seg.strip()
        ]
        chunks.extend(segments)

        # Add token windows so short phishing spans inside long OCR strings are not diluted.
        tokens = normalized.split()
        window = 16
        stride = 8
        if len(tokens) > window:
            for i in range(0, len(tokens), stride):
                piece = " ".join(tokens[i:i + window]).strip()
                if piece:
                    chunks.append(piece)

        # Deduplicate while preserving order.
        deduped = []
        seen = set()
        for c in chunks:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        return deduped
