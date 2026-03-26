from models.deepfake_model import DeepfakeModel
from models.distildire_model import DistilDireModel
from models.safe_model import SafeModel


SUPPORTED_MODELS = ("faceforge", "safe", "distildire")


def create_model(model_name: str, cfg):
    if model_name == "faceforge":
        return DeepfakeModel(
            model_path=cfg.DEEPFAKE_MODEL_PATH,
            device=cfg.DEVICE,
        )

    if model_name == "safe":
        return SafeModel(
            model_path=cfg.SAFE_MODEL_PATH,
            device=cfg.DEVICE,
            input_size=cfg.SAFE_INPUT_SIZE,
            transform_mode=cfg.SAFE_TRANSFORM_MODE,
        )

    if model_name == "distildire":
        return DistilDireModel(
            model_path=cfg.DISTILDIRE_MODEL_PATH,
            adm_model_path=cfg.DISTILDIRE_ADM_MODEL_PATH,
            device=cfg.DEVICE,
            fake_threshold=cfg.DISTILDIRE_FAKE_THRESHOLD,
            image_size=cfg.DISTILDIRE_IMAGE_SIZE,
        )

    raise ValueError(f"Unsupported model backend: {model_name}")
