import os
from pathlib import Path

import torch

# ===== Paths =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    if not value:
        return default

    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


DATA_DIR = PROJECT_ROOT / "data"
SIGLIP_DATA_DIR = _path_from_env("SIGLIP_DATA_DIR", DATA_DIR / "sglip")
OUTPUT_DIR = _path_from_env("SIGLIP_OUTPUT_DIR", PROJECT_ROOT / "outputs")
EMBED_DIR = OUTPUT_DIR / "embeddings"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

for p in [OUTPUT_DIR, EMBED_DIR, MODEL_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ===== Data =====
TRAIN_DIR = SIGLIP_DATA_DIR / "train"
VAL_DIR = SIGLIP_DATA_DIR / "val"
TEST_DIR = SIGLIP_DATA_DIR / "test"

CLASS_NAMES = ["real", "fake"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ===== SigLIP2 =====
MODEL_NAME = os.getenv("SIGLIP_MODEL_NAME", "google/siglip2-base-patch16-224")


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = os.getenv("SIGLIP_DEVICE", detect_device())
BATCH_SIZE = 16
NUM_WORKERS = int(os.getenv("SIGLIP_NUM_WORKERS", "4"))

# ===== Output =====
TRAIN_EMBED_FILE = EMBED_DIR / "train_embeddings.npz"
VAL_EMBED_FILE = EMBED_DIR / "val_embeddings.npz"
TEST_EMBED_FILE = EMBED_DIR / "test_embeddings.npz"

LOGREG_MODEL_FILE = MODEL_DIR / "logreg.joblib"
LGBM_MODEL_FILE = MODEL_DIR / "lightgbm.joblib"
XGB_MODEL_FILE = MODEL_DIR / "xgboost.joblib"

MODEL_FILE_BY_NAME = {
    "logreg": LOGREG_MODEL_FILE,
    "lightgbm": LGBM_MODEL_FILE,
    "xgboost": XGB_MODEL_FILE,
}
