import re


class TextRiskAnalyzer:
    URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(?:com|net|org|io|co|ru|cn|xyz)\b)")
    EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
    PHONE_PATTERN = re.compile(r"(\+\d{1,3}[\s-]?\d{4,14}|\b\d{8,14}\b)")
    MONEY_PATTERN = re.compile(r"(\$\s?\d+(?:[\.,]\d{1,2})?|\b\d+(?:[\.,]\d{1,2})?\s?(?:usd|sgd|rm)\b)", re.IGNORECASE)

    def __init__(self, cfg):
        self.cfg = cfg

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

        score = 0.0
        score += min(total_hits * self.cfg.KEYWORD_HIT_WEIGHT, 0.6)

        if urls:
            score += self.cfg.URL_PRESENT_WEIGHT
        if emails:
            score += self.cfg.EMAIL_PRESENT_WEIGHT
        if phone_numbers:
            score += self.cfg.PHONE_PRESENT_WEIGHT
        if money_mentions:
            score += self.cfg.MONEY_PRESENT_WEIGHT

        signal_families = sum([
            bool(matched_groups),
            bool(urls),
            bool(emails),
            bool(phone_numbers),
            bool(money_mentions),
        ])
        if signal_families >= 3:
            score += self.cfg.MULTI_SIGNAL_BONUS
        if ("credential_request" in matched_groups) and (urls or emails):
            score += self.cfg.LINK_CREDENTIAL_BONUS

        if not normalized:
            score *= self.cfg.EMPTY_TEXT_PENALTY

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
        if not reasons:
            reasons.append("no obvious phishing text signals found")

        return {
            "score": round(score, 4),
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
