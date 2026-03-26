import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from models.safe_model import SAFE_ROOT, SafeModel
from utils.benchmarking import DatasetSample
from utils.dataset_splits import balanced_limit, build_train_val_samples, count_by_label


def _import_safe_transforms():
    safe_path = str(SAFE_ROOT)
    if safe_path not in sys.path:
        sys.path.insert(0, safe_path)

    from data.datasets import Get_Transforms  # pylint: disable=import-outside-toplevel

    return Get_Transforms


def make_dataset_args(
    input_size: int,
    transform_mode: str,
):
    return SimpleNamespace(
        input_size=input_size,
        transform_mode=transform_mode,
        jpeg_factor=None,
        blur_sigma=None,
        mask_ratio=None,
        mask_patch_size=None,
    )


class SafeFolderDataset(Dataset):
    def __init__(
        self,
        samples: list[DatasetSample],
        transform,
        limit: int | None = None,
    ):
        self.samples = balanced_limit(samples, limit)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        target = 1 if sample.label == "fake" else 0
        return tensor, torch.tensor(target, dtype=torch.long)


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    probabilities = []
    targets_all = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            correct += int((preds == targets).sum().item())
            total += int(targets.numel())
            probabilities.extend(probs.cpu().tolist())
            targets_all.extend(targets.cpu().tolist())

    accuracy = correct / total if total else 0.0
    ap = average_precision_score(targets_all, probabilities) if targets_all else 0.0
    return accuracy, ap


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def run_safe_finetune(args) -> None:
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples, split_info = build_train_val_samples(
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    dataset_args = make_dataset_args(
        input_size=args.input_size,
        transform_mode=args.transform_mode,
    )
    get_transforms = _import_safe_transforms()
    train_transform, val_transform = get_transforms(dataset_args)

    train_dataset = SafeFolderDataset(
        samples=train_samples,
        transform=train_transform,
        limit=args.num_train,
    )
    val_dataset = SafeFolderDataset(
        samples=val_samples,
        transform=val_transform,
        limit=args.num_val,
    )
    split_info["effective_train_counts"] = count_by_label(train_dataset.samples)
    split_info["effective_val_counts"] = count_by_label(val_dataset.samples)
    split_info["effective_train_size"] = len(train_dataset)
    split_info["effective_val_size"] = len(val_dataset)
    split_info["train_paths"] = [str(sample.path) for sample in train_dataset.samples]
    split_info["val_paths"] = [str(sample.path) for sample in val_dataset.samples]

    wrapper = SafeModel(
        model_path=args.pretrained_path,
        device=args.device,
        input_size=args.input_size,
        transform_mode=args.transform_mode,
    )
    model = wrapper.model
    device = wrapper.device

    pin_memory = device.startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc1")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = -1.0
    history = []

    with open(output_dir / "split.json", "w", encoding="utf-8") as handle:
        json.dump(split_info, handle, indent=2)

    print(
        f"Using device: {device}\n"
        f"Split mode: {split_info['mode']}\n"
        f"Train samples: {len(train_dataset)} ({split_info['effective_train_counts']})\n"
        f"Val samples: {len(val_dataset)} ({split_info['effective_val_counts']})"
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = int(targets.numel())
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = running_loss / seen if seen else 0.0
        val_accuracy, val_ap = evaluate(model, val_loader, device)
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "val_ap": val_ap,
        }
        history.append(metrics)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f} "
            f"val_ap={val_ap:.4f}"
        )

        save_checkpoint(output_dir / "last.pth", model, optimizer, epoch + 1, metrics)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(output_dir / "best.pth", model, optimizer, epoch + 1, metrics)

    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Saved checkpoints to: {output_dir}")
