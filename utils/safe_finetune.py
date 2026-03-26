import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from models.safe_model import SAFE_ROOT, SafeModel


def _import_safe_dataset():
    safe_path = str(SAFE_ROOT)
    if safe_path not in sys.path:
        sys.path.insert(0, safe_path)

    from data.datasets import TrainDataset  # pylint: disable=import-outside-toplevel

    return TrainDataset


def make_dataset_args(
    train_data_path: str,
    val_data_path: str,
    input_size: int,
    transform_mode: str,
    num_train: int | None,
):
    return SimpleNamespace(
        input_size=input_size,
        transform_mode=transform_mode,
        data_path=train_data_path,
        eval_data_path=val_data_path,
        num_train=num_train if num_train is not None else 10_000_000_000,
        jpeg_factor=None,
        blur_sigma=None,
        mask_ratio=None,
        mask_patch_size=None,
    )


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

    dataset_args = make_dataset_args(
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        input_size=args.input_size,
        transform_mode=args.transform_mode,
        num_train=args.num_train,
    )

    train_dataset_cls = _import_safe_dataset()
    train_dataset = train_dataset_cls(is_train=True, args=dataset_args)
    val_dataset = train_dataset_cls(is_train=False, args=dataset_args)

    pin_memory = args.device != "cpu"
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

    wrapper = SafeModel(
        model_path=args.pretrained_path,
        device=args.device,
        input_size=args.input_size,
        transform_mode=args.transform_mode,
    )
    model = wrapper.model

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

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, targets in train_loader:
            images = images.to(args.device)
            targets = targets.to(args.device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = int(targets.numel())
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = running_loss / seen if seen else 0.0
        val_accuracy, val_ap = evaluate(model, val_loader, args.device)
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
