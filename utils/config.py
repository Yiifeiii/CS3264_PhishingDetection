import torch


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # model selection
    MODEL_BACKEND = "safe"

    # FaceForge
    DEEPFAKE_MODEL_PATH = "models/detector_best.pth"

    # SAFE
    SAFE_MODEL_PATH = "external/SAFE/checkpoint/checkpoint-best.pth"
    SAFE_INPUT_SIZE = 256
    SAFE_TRANSFORM_MODE = "crop"
    SAFE_FINETUNE_TRAIN_PATH = "data/datasets/train_ForenSynths/train"
    SAFE_FINETUNE_VAL_PATH = "data/datasets/train_ForenSynths/val"
    SAFE_FINETUNE_OUTPUT_DIR = "results/SAFE"

    # DistilDIRE
    DISTILDIRE_MODEL_PATH = "models/distildire/celebahq-distil-dire-34e.pth"
    DISTILDIRE_ADM_MODEL_PATH = "models/distildire/256x256_diffusion_uncond.pt"
    DISTILDIRE_IMAGE_SIZE = 256
    DISTILDIRE_FAKE_THRESHOLD = 0.2

    # paths
    REAL_IMAGE = "data/raw/real/image1.jpg"
    FAKE_IMAGE = "data/raw/fake/image2.jpg"
    BENCHMARK_DATASET_ROOT = "data/raw"
