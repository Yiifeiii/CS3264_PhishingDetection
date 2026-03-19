class RiskFusionService:
    def __init__(self, cfg):
        self.cfg = cfg

    def combine(self, image_result, text_result):
        image_score = self._image_risk_score(image_result)
        text_score = float(text_result.get("score", 0.0))

        image_weight = self.cfg.IMAGE_SCORE_WEIGHT
        text_weight = self.cfg.TEXT_SCORE_WEIGHT
        if text_score < 0.05:
            image_weight = 0.55
            text_weight = 0.45

        final_score = (
            image_weight * image_score
            + text_weight * text_score
        )

        # Safety guardrail: strong image manipulation confidence should not become low risk
        # simply because OCR text is neutral or sparse.
        if image_score >= 0.8 and text_score < 0.15:
            final_score = max(final_score, self.cfg.MEDIUM_RISK_THRESHOLD)

        if final_score >= self.cfg.HIGH_RISK_THRESHOLD:
            level = "high"
        elif final_score >= self.cfg.MEDIUM_RISK_THRESHOLD:
            level = "medium"
        else:
            level = "low"

        reasons = list(text_result.get("reasons", []))
        if image_score >= 0.6:
            reasons.append("image model indicates possible manipulation")
        elif image_score <= 0.3:
            reasons.append("image model indicates low manipulation risk")
        if image_score >= 0.8 and text_score < 0.15:
            reasons.append("risk elevated due to highly suspicious image with weak text evidence")

        return {
            "risk_level": level,
            "risk_score": round(final_score, 4),
            "image_score": round(image_score, 4),
            "text_score": round(text_score, 4),
            "reasons": reasons,
        }

    def _image_risk_score(self, image_result):
        label = str(image_result.get("prediction", "")).lower()
        confidence = float(image_result.get("confidence", 0.0))

        suspicious_labels = ("fake", "deepfake", "manipulated", "synthetic", "edited")
        safe_labels = ("real", "authentic", "genuine", "original")

        if any(token in label for token in suspicious_labels):
            return confidence
        if any(token in label for token in safe_labels):
            return 1.0 - confidence
        return 0.5
