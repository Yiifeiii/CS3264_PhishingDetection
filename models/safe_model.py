from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFE_ROOT = REPO_ROOT / "external" / "SAFE"


def prepare_safe_input(
    image: Image.Image,
    input_size: int = 256,
    transform_mode: str = "crop",
) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")

    if transform_mode == "crop":
        transform = transforms.Compose(
            [
                transforms.CenterCrop((input_size, input_size)),
                transforms.ToTensor(),
            ]
        )
    elif transform_mode == "resize_BILINEAR":
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (input_size, input_size),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )
    elif transform_mode == "resize_NEAREST":
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (input_size, input_size),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )
    elif transform_mode == "source":
        transform = transforms.ToTensor()
    else:
        raise ValueError(f"Unsupported SAFE transform_mode: {transform_mode}")

    return transform(image)


def build_safe_result(probabilities: torch.Tensor) -> dict:
    fake_probability = float(probabilities[1])
    real_probability = float(probabilities[0])
    prediction = "fake" if fake_probability >= real_probability else "real"
    confidence = fake_probability if prediction == "fake" else real_probability

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "real": real_probability,
            "fake": fake_probability,
        },
    }


def _import_safe_model():
    if not SAFE_ROOT.exists():
        raise FileNotFoundError(
            f"SAFE repo not found at {SAFE_ROOT}. "
            "Clone the official SAFE repository into external/SAFE first."
        )

    safe_path = str(SAFE_ROOT)
    if safe_path not in sys.path:
        sys.path.insert(0, safe_path)

    from models.resnet import resnet50  # pylint: disable=import-outside-toplevel

    return resnet50


class SafeModel:
    backend = "safe"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        input_size: int = 256,
        transform_mode: str = "crop",
    ):
        self.device = self._resolve_device(device)
        self.input_size = input_size
        self.transform_mode = transform_mode
        self.id2label = {
            0: "real",
            1: "fake",
        }
        self.label2id = {label: idx for idx, label in self.id2label.items()}

        resnet50 = _import_safe_model()
        self.model = resnet50(num_classes=2)

        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested_device

    def predict_image(self, image: Image.Image | str | Path) -> dict:
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        else:
            pil_image = image

        image_tensor = prepare_safe_input(
            pil_image,
            input_size=self.input_size,
            transform_mode=self.transform_mode,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu()

        return build_safe_result(probabilities)
