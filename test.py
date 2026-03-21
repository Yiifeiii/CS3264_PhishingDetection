import torch
import timm
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import sys

class FaceForgeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.xception = timm.create_model(
            "legacy_xception",
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )

        in_features = self.xception.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.xception(x)
        x = self.classifier(x)
        return x


# =========================
# LOAD MODEL
# =========================
# Load model
model = FaceForgeNet()
checkpoint = torch.load('models/detector_best.pth', map_location='cpu')
state_dict = checkpoint["model_state_dict"]
model.load_state_dict(state_dict, strict=True)

model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Inference
def detect_deepfake(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
    
    label = "REAL" if prediction == 0 else "FAKE"
    return label, confidence

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/image2.jpg"
    label, confidence = detect_deepfake(image_path)
    print(f"Image: {image_path}")
    print(f"Prediction: {label} (Confidence: {confidence:.2%})")
