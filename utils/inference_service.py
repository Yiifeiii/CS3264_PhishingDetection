import torch
from torchvision import transforms


class InferenceService:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self.labels = model_wrapper.id2label
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

    def predict(self, image):
        if isinstance(image, torch.Tensor):
            image_tensor = image
        else:
            image_tensor = self.transform(image).unsqueeze(0)

        logits = self.model_wrapper.predict(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())
        return {
            "prediction": self.labels[pred],
            "confidence": float(probs[pred]),
            "probabilities": {
                self.labels[i]: float(probs[i]) for i in range(len(probs))
            },
        }
