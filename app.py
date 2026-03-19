from pathlib import Path

from utils.config import Config
from models.deepfake_model import DeepfakeModel
from preprocess.image_preprocessor import ImagePreprocessor
from utils.inference_service import InferenceService
from ocr.ocr_service import OCRService
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
        self.ocr = OCRService()
        self.text_analyzer = TextRiskAnalyzer(self.cfg)
        self.risk_fusion = RiskFusionService(self.cfg)

    def run_single(self, path):
        image = self.preprocessor.load_image(path)

        image_result = self.infer.predict(image)
        text = self.ocr.extract_text(path)
        text_result = self.text_analyzer.analyze(text)
        fused_result = self.risk_fusion.combine(image_result, text_result)

        print(f"\nImage: {path}")
        print("Image Model:", image_result)
        print("OCR Text:", text)
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
