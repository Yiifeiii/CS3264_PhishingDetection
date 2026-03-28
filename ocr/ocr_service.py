import easyocr

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
        results = self.reader.readtext(image_path, detail=0)
        return " ".join(results)
