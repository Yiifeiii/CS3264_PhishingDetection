import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    MODEL_NAME, DEVICE, BATCH_SIZE, NUM_WORKERS,
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    CLASS_NAMES, CLASS_TO_ID,
    TRAIN_EMBED_FILE, VAL_EMBED_FILE, TEST_EMBED_FILE
)
from dataset import collect_samples, ImagePathDataset
from feature_utils import get_normalized_image_features
from hf_utils import load_siglip_processor_and_model


def collate_fn(batch):
    images, labels, paths = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(paths)


def load_siglip():
    return load_siglip_processor_and_model(MODEL_NAME, DEVICE)


@torch.no_grad()
def extract_split_embeddings(split_dir, output_file, processor, model):
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}. "
            f"Expected folders like {split_dir / CLASS_NAMES[0]} and {split_dir / CLASS_NAMES[1]}."
        )

    samples = collect_samples(split_dir, CLASS_TO_ID)
    if not samples:
        raise ValueError(
            f"No supported images found under {split_dir}. "
            "Expected files ending in .jpg, .jpeg, .png, .webp, or .bmp."
        )

    dataset = ImagePathDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    all_embeddings = []
    all_labels = []
    all_paths = []

    for images, labels, paths in tqdm(loader, desc=f"Extracting {split_dir.name}"):
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        image_features = get_normalized_image_features(model, inputs)

        all_embeddings.append(image_features.cpu().numpy())
        all_labels.append(labels.numpy())
        all_paths.extend(paths)

    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)
    paths_arr = np.array(all_paths)

    np.savez_compressed(output_file, X=X, y=y, paths=paths_arr)
    print(f"Saved: {output_file}")
    print(f"Embeddings shape: {X.shape}")
    print(f"Labels shape: {y.shape}")


if __name__ == "__main__":
    processor, model = load_siglip()
    extract_split_embeddings(TRAIN_DIR, TRAIN_EMBED_FILE, processor, model)
    extract_split_embeddings(VAL_DIR, VAL_EMBED_FILE, processor, model)
    extract_split_embeddings(TEST_DIR, TEST_EMBED_FILE, processor, model)
