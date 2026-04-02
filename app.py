from pathlib import Path

from utils.config import Config
from models.deepfake_model import DeepfakeModel
from preprocess.image_preprocessor import ImagePreprocessor
from utils.inference_service import InferenceService
from ocr.ocr_service import OCRService
from utils.ocr_text_processor import OCRTextProcessor
from utils.text_risk_analyzer import TextRiskAnalyzer
from utils.risk_fusion_service import RiskFusionService


class App:
    def __init__(self):
        self.cfg = Config()

        self.model = DeepfakeModel(
            self.cfg.DEEPFAKE_MODEL_NAME,
            self.cfg.DEVICE
        )

        self.infer = InferenceService(self.model)
        self.preprocessor = ImagePreprocessor()
        self.ocr = OCRService(list(self.cfg.OCR_LANGUAGES), gpu=(self.cfg.DEVICE == "cuda"))
        self.ocr_text_processor = OCRTextProcessor(self.cfg)
        self.text_analyzer = TextRiskAnalyzer(self.cfg)
        self.risk_fusion = RiskFusionService(self.cfg)

    def run_single(self, path):
        image = self.preprocessor.load_image(path)

        image_result = self.infer.predict(image)
        raw_text = self.ocr.extract_text(path)
        processed_text = self.ocr_text_processor.process(raw_text)
        text = processed_text["text"]
        text_result = self.text_analyzer.analyze(text)
        if processed_text.get("warning"):
            text_result.setdefault("warnings", []).append(processed_text["warning"])
        text_result["ocr_processing"] = processed_text
        fused_result = self.risk_fusion.combine(image_result, text_result)

        print(f"\nImage: {path}")
        print("OCR Languages:", self.ocr.active_languages)
        print("OCR Load Warning:", self.ocr.load_error)
        print("Chinese Policy:", self.cfg.OCR_CHINESE_POLICY)
        print("English Text Model Loaded:", self.text_analyzer.model.is_loaded)
        print("Chinese Text Model Loaded:", bool(self.text_analyzer.chinese_model and self.text_analyzer.chinese_model.is_loaded))
        print("Chinese Text Model Warning:", None if self.text_analyzer.chinese_model is None else self.text_analyzer.chinese_model.load_error)
        print("Translator Loaded:", self.ocr_text_processor.is_loaded)
        print("Translator Load Warning:", self.ocr_text_processor.load_error)
        print("Image Model:", image_result)
        print("OCR Raw Text:", raw_text)
        print("OCR Processed Text:", text)
        print("Text Signals:", text_result)
        print("Phishing Risk:", fused_result)

    def run(self):
        image_dir = Path(self.cfg.RAW_IMAGE_DIR)
        allowed_exts = {ext.lower() for ext in self.cfg.SUPPORTED_IMAGE_EXTENSIONS}

        image_paths = sorted(
            str(p) for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in allowed_exts
        )

        if not image_paths:
            print(f"No images found in {image_dir}")
            return

        for path in image_paths:
            self.run_single(path)


if __name__ == "__main__":
    app = App()
    app.run()