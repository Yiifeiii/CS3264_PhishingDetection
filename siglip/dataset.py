from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset
from utils.image_loading import load_image_rgb

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}


def collect_samples(root_dir: Path, class_to_id: dict) -> List[Tuple[str, int]]:
    samples = []
    for class_name, class_id in class_to_id.items():
        class_dir = root_dir / class_name
        if not class_dir.exists():
            continue
        for path in class_dir.rglob("*"):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                samples.append((str(path), class_id))
    return sorted(samples)


class ImagePathDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = load_image_rgb(image_path)
        return image, label, image_path
