from utils.config import Config
from models.deepfake_model import DeepfakeModel
from preprocess.image_preprocessor import ImagePreprocessor
from utils.inference_service import InferenceService
from ocr.ocr_service import OCRService


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

    def run_single(self, path):
        image = self.preprocessor.load_image(path)

        result = self.infer.predict(image)
        text = self.ocr.extract_text(path)

        print(f"\nImage: {path}")
        print("Prediction:", result)
        print("OCR Text:", text)

    def run(self):
        self.run_single(self.cfg.REAL_IMAGE)
        self.run_single(self.cfg.FAKE_IMAGE)


if __name__ == "__main__":
    app = App()
    app.run()