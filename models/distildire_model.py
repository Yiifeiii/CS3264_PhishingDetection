from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DISTILDIRE_ROOT = REPO_ROOT / "external" / "DistilDIRE"


def prepare_distildire_input(image: Image.Image, image_size: int = 256) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = TF.to_tensor(image) * 2 - 1
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop((image_size, image_size)),
        ]
    )
    return transform(tensor)


def build_distildire_result(fake_probability: float, threshold: float) -> dict:
    prediction = "fake" if fake_probability >= threshold else "real"
    confidence = fake_probability if prediction == "fake" else 1 - fake_probability
    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "probabilities": {
            "real": float(1 - fake_probability),
            "fake": float(fake_probability),
        },
        "fake_threshold": float(threshold),
    }


def _import_distildire_modules():
    if not DISTILDIRE_ROOT.exists():
        raise FileNotFoundError(
            f"DistilDIRE repo not found at {DISTILDIRE_ROOT}. "
            "Clone the official repo into external/DistilDIRE first."
        )

    distildire_path = str(DISTILDIRE_ROOT)
    if distildire_path not in sys.path:
        sys.path.insert(0, distildire_path)

    from guided_diffusion.compute_dire_eps import (  # pylint: disable=import-outside-toplevel
        create_dicts_for_static_init,
        dire_get_first_step_noise,
    )
    from guided_diffusion.guided_diffusion.script_util import (  # pylint: disable=import-outside-toplevel
        create_model_and_diffusion,
        dict_parse,
        model_and_diffusion_defaults,
    )
    from networks.distill_model import DistilDIRE  # pylint: disable=import-outside-toplevel

    return (
        DistilDIRE,
        create_dicts_for_static_init,
        dire_get_first_step_noise,
        create_model_and_diffusion,
        dict_parse,
        model_and_diffusion_defaults,
    )


class DistilDireModel:
    backend = "distildire"

    def __init__(
        self,
        model_path: str,
        adm_model_path: str,
        device: str = "cpu",
        fake_threshold: float = 0.2,
        image_size: int = 256,
    ):
        self.device = self._resolve_device(device)
        self.fake_threshold = fake_threshold
        self.image_size = image_size
        self.id2label = {
            0: "real",
            1: "fake",
        }
        self.label2id = {label: idx for idx, label in self.id2label.items()}
        (
            distildire_cls,
            create_dicts_for_static_init,
            self._dire_get_first_step_noise,
            create_model_and_diffusion,
            dict_parse,
            model_and_diffusion_defaults,
        ) = _import_distildire_modules()

        self.model = distildire_cls(self.device).to(self.device)
        self._load_student_weights(model_path)
        self.model.eval()

        self.args = create_dicts_for_static_init()
        self.args["timestep_respacing"] = "ddim20"
        self.args["model_path"] = adm_model_path
        self.args["image_size"] = image_size
        self.adm_model, self.diffusion = create_model_and_diffusion(
            **dict_parse(self.args, model_and_diffusion_defaults().keys())
        )
        self.adm_model.load_state_dict(torch.load(adm_model_path, map_location="cpu"))
        self.adm_model.to(self.device)
        self.adm_model.eval()

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested_device

    def _load_student_weights(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

    def preprocess_image(self, image: Image.Image | str | Path) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        else:
            pil_image = image
        return prepare_distildire_input(
            pil_image,
            image_size=self.image_size,
        )

    def compute_eps(self, image_batch: torch.Tensor) -> torch.Tensor:
        return self._dire_get_first_step_noise(
            image_batch,
            self.adm_model,
            self.diffusion,
            self.args,
            self.device,
        )

    def forward_logits(
        self,
        image_batch: torch.Tensor,
        eps_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_batch = image_batch.to(self.device)
        if eps_batch is None:
            eps_batch = self.compute_eps(image_batch)
        output = self.model(image_batch, eps_batch)
        return output["logit"].view(-1)

    def predict_image(self, image: Image.Image | str | Path) -> dict:
        image_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            fake_probability = float(self.forward_logits(image_tensor).sigmoid().item())

        return build_distildire_result(fake_probability, self.fake_threshold)

    def predict(self, image: Image.Image | str | Path) -> dict:
        return self.predict_image(image)
