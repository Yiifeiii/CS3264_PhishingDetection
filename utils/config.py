import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    DEEPFAKE_MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

    # paths
    REAL_IMAGE = "data/raw/image1.jpg"
    FAKE_IMAGE = "data/raw/image3.jpg"
