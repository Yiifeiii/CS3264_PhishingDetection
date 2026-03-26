import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from models.distildire_model import (
    DistilDireModel,
    build_distildire_result,
    prepare_distildire_input,
)
from utils.benchmarking import DatasetSample, summarize_predictions
from utils.dataset_splits import balanced_limit, build_train_val_samples, count_by_label


class DistilDireFolderDataset(Dataset):
    def __init__(
        self,
        samples: list[DatasetSample],
        image_size: int,
        limit: int | None = None,
    ):
        self.samples = balanced_limit(samples, limit)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = prepare_distildire_input(image, image_size=self.image_size)
        label = 1.0 if sample.label == "fake" else 0.0
        return tensor, torch.tensor(label, dtype=torch.float32), str(sample.path)


def _compute_average_precision(
    targets: list[int],
    probabilities: list[float],
) -> float:
    if len(set(targets)) < 2:
        return 0.0
    return float(average_precision_score(targets, probabilities))


def evaluate_distildire(
    wrapper: DistilDireModel,
    data_loader: DataLoader,
    fake_threshold: float,
) -> dict:
    wrapper.model.eval()
    rows = []
    targets_all = []
    probabilities_all = []

    with torch.no_grad():
        for images, targets, paths in data_loader:
            images = images.to(wrapper.device)
            targets = targets.to(wrapper.device)

            eps = wrapper.compute_eps(images)
            logits = wrapper.forward_logits(images, eps)
            probabilities = torch.sigmoid(logits).cpu().tolist()
            targets_cpu = targets.int().cpu().tolist()

            for path, target, probability in zip(paths, targets_cpu, probabilities):
                label = "fake" if target == 1 else "real"
                result = build_distildire_result(probability, fake_threshold)
                rows.append(
                    {
                        "path": path,
                        "label": label,
                        "prediction": result["prediction"],
                        "confidence": result["confidence"],
                        "probabilities": result["probabilities"],
                    }
                )
                targets_all.append(target)
                probabilities_all.append(probability)

    summary = summarize_predictions(rows)
    summary["ap"] = _compute_average_precision(targets_all, probabilities_all)
    return summary


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    metrics: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def run_distildire_finetune(args) -> None:
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples, split_info = build_train_val_samples(
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_dataset = DistilDireFolderDataset(
        samples=train_samples,
        image_size=args.input_size,
        limit=args.num_train,
    )
    val_dataset = DistilDireFolderDataset(
        samples=val_samples,
        image_size=args.input_size,
        limit=args.num_val,
    )
    split_info["effective_train_counts"] = count_by_label(train_dataset.samples)
    split_info["effective_val_counts"] = count_by_label(val_dataset.samples)
    split_info["effective_train_size"] = len(train_dataset)
    split_info["effective_val_size"] = len(val_dataset)
    split_info["train_paths"] = [str(sample.path) for sample in train_dataset.samples]
    split_info["val_paths"] = [str(sample.path) for sample in val_dataset.samples]

    wrapper = DistilDireModel(
        model_path=args.pretrained_path,
        adm_model_path=args.adm_model_path,
        device=args.device,
        fake_threshold=args.fake_threshold,
        image_size=args.input_size,
    )
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
        for name, parameter in wrapper.model.named_parameters():
            parameter.requires_grad = name.startswith("student_head")

    trainable_params = [parameter for parameter in wrapper.model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for DistilDIRE fine-tuning.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    best_score = (-1.0, -1.0)
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
        wrapper.model.train()
        running_loss = 0.0
        seen = 0

        for images, targets, _paths in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                eps = wrapper.compute_eps(images)
            logits = wrapper.forward_logits(images, eps)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = int(targets.numel())
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = running_loss / seen if seen else 0.0
        val_summary = evaluate_distildire(
            wrapper=wrapper,
            data_loader=val_loader,
            fake_threshold=args.fake_threshold,
        )

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_summary["accuracy"],
            "val_precision": val_summary["precision"],
            "val_recall": val_summary["recall"],
            "val_f1": val_summary["f1"],
            "val_ap": val_summary["ap"],
            "val_tp": val_summary["tp"],
            "val_tn": val_summary["tn"],
            "val_fp": val_summary["fp"],
            "val_fn": val_summary["fn"],
        }
        history.append(metrics)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_accuracy={val_summary['accuracy']:.4f} "
            f"val_f1={val_summary['f1']:.4f} "
            f"val_ap={val_summary['ap']:.4f}"
        )

        save_checkpoint(output_dir / "last.pth", wrapper.model, optimizer, epoch + 1, metrics)

        score = (val_summary["f1"], val_summary["accuracy"])
        if score > best_score:
            best_score = score
            save_checkpoint(output_dir / "best.pth", wrapper.model, optimizer, epoch + 1, metrics)

    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    best_f1, best_accuracy = best_score
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Saved checkpoints to: {output_dir}")
