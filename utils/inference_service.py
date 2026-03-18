import torch

class InferenceService:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.processor = model_wrapper.processor
        self.device = model_wrapper.device
        self.labels = self.model.config.id2label

    def predict(self, image):
        logits = self.model_wrapper.predict(image)

        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs))

        return {
            "prediction": self.labels[pred],
            "confidence": float(probs[pred])
        }