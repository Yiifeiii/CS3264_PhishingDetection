import torch
import torch.nn as nn
import timm


class FaceForgeNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Use legacy_xception to match timm's current naming
        self.xception = timm.create_model(
            "legacy_xception",
            pretrained=False,
            num_classes=0,      # remove default classifier
            global_pool="avg"
        )

        in_features = self.xception.num_features  # usually 2048 for xception

        # Match checkpoint structure:
        # classifier.1.weight and classifier.4.weight suggest:
        # [Dropout, Linear, ReLU, Dropout, Linear]
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.xception(x)
        logits = self.classifier(features)
        return logits


class DeepfakeModel:
    def __init__(self, model_path: str, device: str):
        self.device = device
        self.model = FaceForgeNet(num_classes=2)

        checkpoint = torch.load(model_path, map_location=device)

        # Handle common checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Load checkpoint
        self.model.load_state_dict(state_dict, strict=True)

        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            0: "real",
            1: "fake"
        }
        self.label2id = {label: index for index, label in self.id2label.items()}

    def predict(self, image_tensor):
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)

        return logits
