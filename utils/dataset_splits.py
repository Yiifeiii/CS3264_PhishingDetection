from collections import defaultdict
from pathlib import Path
import random

from utils.benchmarking import DatasetSample, discover_labeled_images


def load_labeled_samples(dataset_root: str) -> list[DatasetSample]:
    samples = discover_labeled_images(dataset_root)
    if not samples:
        raise ValueError(
            f"No labeled images found under {dataset_root}. "
            "Expected paths containing real/fake or 0_real/1_fake."
        )
    return samples


def balanced_limit(
    samples: list[DatasetSample],
    limit: int | None,
) -> list[DatasetSample]:
    if limit is None or limit >= len(samples):
        return samples

    buckets = defaultdict(list)
    for sample in samples:
        buckets[sample.label].append(sample)

    limited = []
    while len(limited) < limit:
        added_any = False
        for label in ("real", "fake"):
            if not buckets[label]:
                continue
            limited.append(buckets[label].pop(0))
            added_any = True
            if len(limited) == limit:
                break
        if not added_any:
            break

    return limited


def count_by_label(samples: list[DatasetSample]) -> dict[str, int]:
    counts = {"real": 0, "fake": 0}
    for sample in samples:
        counts[sample.label] += 1
    return counts


def stratified_train_val_split(
    samples: list[DatasetSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[DatasetSample], list[DatasetSample]]:
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}.")

    train_samples = []
    val_samples = []

    for label in ("real", "fake"):
        label_samples = [sample for sample in samples if sample.label == label]
        if len(label_samples) < 2:
            raise ValueError(
                f"Need at least 2 samples for label '{label}' to create a train/val split."
            )

        rng = random.Random(f"{seed}:{label}")
        shuffled = label_samples[:]
        rng.shuffle(shuffled)

        val_count = int(round(len(shuffled) * val_ratio))
        val_count = max(1, min(len(shuffled) - 1, val_count))

        val_samples.extend(shuffled[:val_count])
        train_samples.extend(shuffled[val_count:])

    train_samples.sort(key=lambda sample: str(sample.path))
    val_samples.sort(key=lambda sample: str(sample.path))
    return train_samples, val_samples


def build_train_val_samples(
    train_data_path: str,
    val_data_path: str,
    val_ratio: float,
    seed: int,
) -> tuple[list[DatasetSample], list[DatasetSample], dict]:
    if Path(train_data_path).resolve() == Path(val_data_path).resolve():
        all_samples = load_labeled_samples(train_data_path)
        train_samples, val_samples = stratified_train_val_split(
            all_samples,
            val_ratio=val_ratio,
            seed=seed,
        )
        return train_samples, val_samples, {
            "mode": "auto_split_same_root",
            "source_root": str(Path(train_data_path)),
            "val_ratio": val_ratio,
            "seed": seed,
            "total_samples": len(all_samples),
            "pre_limit_train_counts": count_by_label(train_samples),
            "pre_limit_val_counts": count_by_label(val_samples),
        }

    train_samples = load_labeled_samples(train_data_path)
    val_samples = load_labeled_samples(val_data_path)
    return train_samples, val_samples, {
        "mode": "separate_roots",
        "train_root": str(Path(train_data_path)),
        "val_root": str(Path(val_data_path)),
        "pre_limit_train_counts": count_by_label(train_samples),
        "pre_limit_val_counts": count_by_label(val_samples),
    }
