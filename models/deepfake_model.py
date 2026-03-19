from transformers import AutoImageProcessor, AutoModelForImageClassification

class DeepfakeModel:
    def __init__(self, model_name: str, device: str):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)
        return outputs.logits