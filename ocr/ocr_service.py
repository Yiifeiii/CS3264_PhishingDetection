import easyocr

class OCRService:
    def __init__(self, languages=None, gpu=False):
        if languages is None:
            languages = ["en"]
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def extract_text(self, image_path: str) -> str:
        results = self.reader.readtext(image_path, detail=0)
        return " ".join(results)