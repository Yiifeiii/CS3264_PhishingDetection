import torch


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    DEEPFAKE_MODEL_PATH = "models/detector_best.pth"

    # paths
    REAL_IMAGE = "data/raw/IMG_5437.JPG"
    FAKE_IMAGE = "data/raw/image2.jpg"
