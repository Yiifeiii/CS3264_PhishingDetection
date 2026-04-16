import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGLIP_ROOT = PROJECT_ROOT / "siglip"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SIGLIP_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hf_utils import load_siglip_processor_and_model  # noqa: E402
from feature_utils import get_normalized_image_features  # noqa: E402
from classifier_utils import build_classifier, predict_positive_scores  # noqa: E402
from ocr.ocr_service import OCRService  # noqa: E402
from run_siglip_grounding_dino_crop_stream import (  # noqa: E402
    DEFAULT_PROMPT_LABELS,
    build_grounding_dino_candidates,
    detect_grounding_dino_regions,
    load_grounding_dino_processor_and_model,
)
from run_siglip_ocr_crop_stream import generate_crop_candidates, safe_slug  # noqa: E402


CLASS_NAME_TO_ID = {"real": 0, "fake": 1}
CLASSIFIER_NAME_ALIASES = {
    "lr": "logreg",
    "logreg": "logreg",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "contrastive": "contrastive",
}
CLASSIFIER_DISPLAY_NAMES = {
    "logreg": "Logistic Regression",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "contrastive": "Contrastive Projection",
}
PRIORITY_TOKEN_WEIGHTS = {
    "telegram": 6.0,
    "whatsapp": 6.0,
    "outlook": 6.0,
    "office365": 6.0,
    "microsoft": 5.0,
    "onedrive": 4.0,
    "paypal": 4.0,
    "facebook": 4.0,
    "instagram": 4.0,
    "linkedin": 4.0,
    "singtel": 7.0,
    "dbs": 7.0,
    "ocbc": 7.0,
    "uob": 7.0,
    "grab": 6.0,
    "govtech": 7.0,
    "singpass": 7.0,
    "posb": 7.0,
    "citibank": 5.0,
}
SINGAPORE_DOMAIN_BONUS = 4.0
GROUNDING_SIGNAL_LABEL_WEIGHTS = {
    "password field": 7.0,
    "login form": 6.5,
    "sign in form": 6.5,
    "qr code": 6.0,
    "barcode": 5.5,
    "notification": 5.0,
    "pop-up": 5.0,
    "dialog box": 4.5,
    "text field": 4.0,
    "button": 3.0,
    "card": 2.5,
    "logo": 2.0,
}
GROUNDING_SIGNAL_LABEL_REASONS = {
    "password field": "Credential-entry UI is a strong phishing signal.",
    "login form": "A login form often carries the credential theft surface.",
    "sign in form": "A sign-in form often carries the credential theft surface.",
    "qr code": "QR prompts are commonly used in modern phishing flows.",
    "barcode": "Machine-readable code can redirect victims away from the visible page.",
    "notification": "Urgent notifications are common social-engineering bait.",
    "pop-up": "Pop-ups often carry urgency, blocking, or forced-click bait.",
    "dialog box": "Dialog-style overlays often hide the core phishing prompt.",
    "text field": "Free-text input fields can capture account or contact data.",
    "button": "Buttons often trigger the phishing action after social-engineering text.",
    "card": "Card-like panels often group the phishing CTA with brand cues.",
    "logo": "Brand logos reinforce impersonation but are weaker than credential fields.",
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def detect_device(device: str = "auto"):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def canonicalize_classifier_name(model_name: str):
    canonical = CLASSIFIER_NAME_ALIASES.get(str(model_name).strip().lower())
    if canonical is None:
        valid = ", ".join(sorted(CLASSIFIER_NAME_ALIASES))
        raise ValueError(f"Unsupported classifier `{model_name}`. Expected one of: {valid}")
    return canonical


def classifier_display_name(model_name: str):
    canonical = canonicalize_classifier_name(model_name)
    return CLASSIFIER_DISPLAY_NAMES[canonical]


def tokenize_source_name(source_name: str):
    return [token for token in re.split(r"[^a-z0-9]+", source_name.lower()) if token]


def compute_priority(source_name: str):
    raw_name = source_name.lower()
    tokens = set(tokenize_source_name(source_name))
    matched_tokens = sorted(token for token in tokens if token in PRIORITY_TOKEN_WEIGHTS)
    score = sum(PRIORITY_TOKEN_WEIGHTS[token] for token in matched_tokens)
    if raw_name.endswith(".sg") or ".com.sg" in raw_name:
        matched_tokens.append("sg_domain")
        score += SINGAPORE_DOMAIN_BONUS
    return matched_tokens, float(score)


def collect_items(source_root: Path, class_name: str):
    items = []
    for image_path in sorted(source_root.glob("*/shot.png")):
        source_name = image_path.parent.name
        matched_tokens, priority_score = compute_priority(source_name)
        items.append(
            {
                "class_name": class_name,
                "label_id": CLASS_NAME_TO_ID[class_name],
                "image_path": image_path.resolve(),
                "source_name": source_name,
                "matched_tokens": matched_tokens,
                "priority_score": priority_score,
                "sampling_weight": 1.0 + priority_score,
            }
        )
    if not items:
        raise ValueError(f"No shot.png files found under {source_root}")
    return items


def _weighted_sample_without_replacement(items, sample_size: int, rng: np.random.Generator):
    if sample_size <= 0:
        return []
    if sample_size >= len(items):
        return list(items)

    weights = np.asarray(
        [max(float(item.get("sampling_weight", 1.0)), 1e-6) for item in items],
        dtype=float,
    )
    uniforms = np.clip(rng.random(len(items)), 1e-12, 1.0)
    keys = np.log(uniforms) / weights
    selected_indices = np.argpartition(keys, -sample_size)[-sample_size:]
    selected_indices = selected_indices[np.argsort(keys[selected_indices])[::-1]]
    return [items[int(idx)] for idx in selected_indices]


def _select_class_split(
    items,
    train_size: int,
    test_size: int,
    priority_fraction: float,
    rng: np.random.Generator,
):
    total_size = train_size + test_size
    if len(items) < total_size:
        raise ValueError(f"Need {total_size} items but only found {len(items)}")

    priority_items = [item for item in items if item["priority_score"] > 0]
    background_items = [item for item in items if item["priority_score"] <= 0]
    priority_target = min(len(priority_items), int(round(total_size * priority_fraction)))

    selected = []
    selected_paths = set()
    if priority_target > 0:
        priority_selected = _weighted_sample_without_replacement(priority_items, priority_target, rng)
        selected.extend(priority_selected)
        selected_paths.update(item["image_path"] for item in priority_selected)

    remaining_needed = total_size - len(selected)
    if remaining_needed > 0:
        if len(background_items) >= remaining_needed:
            background_selected = _weighted_sample_without_replacement(background_items, remaining_needed, rng)
            selected.extend(background_selected)
            selected_paths.update(item["image_path"] for item in background_selected)
        else:
            selected.extend(background_items)
            selected_paths.update(item["image_path"] for item in background_items)
            extra_needed = total_size - len(selected)
            priority_pool = [item for item in priority_items if item["image_path"] not in selected_paths]
            extra_selected = _weighted_sample_without_replacement(priority_pool, extra_needed, rng)
            selected.extend(extra_selected)
            selected_paths.update(item["image_path"] for item in extra_selected)

    if len(selected) != total_size:
        raise RuntimeError(f"Split sampling failed; expected {total_size}, got {len(selected)}")

    selected = list(selected)
    rng.shuffle(selected)
    train_items = selected[:train_size]
    test_items = selected[train_size:]
    return train_items, test_items


def build_balanced_split(
    real_root: Path,
    fake_root: Path,
    train_per_class: int,
    test_per_class: int,
    seed: int,
    priority_fraction: float,
):
    rng = np.random.default_rng(seed)

    real_items = collect_items(real_root, "real")
    fake_items = collect_items(fake_root, "fake")

    train_real, test_real = _select_class_split(
        real_items,
        train_size=train_per_class,
        test_size=test_per_class,
        priority_fraction=priority_fraction,
        rng=rng,
    )
    train_fake, test_fake = _select_class_split(
        fake_items,
        train_size=train_per_class,
        test_size=test_per_class,
        priority_fraction=priority_fraction,
        rng=rng,
    )

    manifest = []
    for split_name, rows in (
        ("train", train_real + train_fake),
        ("test", test_real + test_fake),
    ):
        for row in rows:
            manifest.append(
                {
                    "split": split_name,
                    "class_name": row["class_name"],
                    "label_id": row["label_id"],
                    "image_path": str(row["image_path"]),
                    "source_name": row["source_name"],
                    "priority_score": row["priority_score"],
                    "matched_tokens": "|".join(row["matched_tokens"]),
                }
            )

    manifest.sort(key=lambda row: (row["split"], row["class_name"], row["source_name"]))
    return manifest


def split_rows(manifest, split_name: str):
    return [row for row in manifest if row["split"] == split_name]


def split_summary(manifest):
    summary = {"splits": {}}
    for split_name in ("train", "test"):
        rows = split_rows(manifest, split_name)
        summary["splits"][split_name] = {
            "rows": len(rows),
            "real": sum(row["label_id"] == 0 for row in rows),
            "fake": sum(row["label_id"] == 1 for row in rows),
            "prioritized_rows": sum(float(row["priority_score"]) > 0 for row in rows),
            "priority_examples": [
                row["source_name"] for row in rows if float(row["priority_score"]) > 0
            ][:20],
        }
    return summary


@torch.no_grad()
def embed_image_paths(
    image_paths,
    processor,
    model,
    device: str,
    batch_size: int,
    stage_name: str | None = None,
    log_every: int = 0,
    logger=None,
):
    total = len(image_paths)
    last_logged = 0
    features = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch_images = []
        try:
            for image_path in batch_paths:
                batch_images.append(Image.open(image_path).convert("RGB"))
            inputs = processor(images=batch_images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            batch_features = get_normalized_image_features(model, inputs)
            features.append(batch_features.detach().cpu().numpy())
        finally:
            for image in batch_images:
                image.close()

        processed = min(start + len(batch_paths), total)
        if logger and log_every > 0 and (processed - last_logged >= log_every or processed == total):
            logger(f"{stage_name or 'global'}: embedded {processed}/{total} images")
            last_logged = processed

    return np.concatenate(features, axis=0)


@torch.no_grad()
def embed_pil_images(images, processor, model, device: str, batch_size: int):
    if not images:
        raise ValueError("No images were provided for embedding.")

    features = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        batch_features = get_normalized_image_features(model, inputs)
        features.append(batch_features.detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def _bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def pool_crop_features(crop_features, crop_weights, pooling: str):
    if pooling == "max":
        pooled_feature = np.max(crop_features, axis=0)
    else:
        crop_weights = np.asarray(crop_weights, dtype=np.float32)
        pooled_feature = np.average(crop_features, axis=0, weights=crop_weights)

    pooled_feature = pooled_feature / max(np.linalg.norm(pooled_feature), 1e-12)
    return pooled_feature.astype(np.float32)


def l2_normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (X / norms).astype(np.float32)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        return self.net(x)


def supervised_contrastive_loss(projections, labels, temperature: float):
    projections = F.normalize(projections, dim=1)
    logits = projections @ projections.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    batch_size = labels.shape[0]
    self_mask = torch.eye(batch_size, device=labels.device, dtype=torch.bool)
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~self_mask)

    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return None

    loss = -(positive_mask * log_prob).sum(dim=1) / positive_count.clamp(min=1)
    return loss[valid].mean()


class ContrastiveProjectionClassifier:
    def __init__(
        self,
        seed: int,
        hidden_dim: int,
        projection_dim: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        temperature: float,
        device: str,
        log_every: int = 0,
        logger=None,
    ):
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.device = detect_device(device)
        self.log_every = log_every
        self.logger = logger

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X).astype(np.float32)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=min(self.batch_size, len(y)),
            shuffle=True,
            drop_last=False,
        )

        self.projector_ = ProjectionHead(
            input_dim=X.shape[1],
            hidden_dim=self.hidden_dim,
            projection_dim=self.projection_dim,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.projector_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.training_history_ = []
        self.projector_.train()
        for epoch in range(1, self.epochs + 1):
            running_loss = 0.0
            steps = 0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                projections = self.projector_(batch_X)
                loss = supervised_contrastive_loss(
                    projections,
                    batch_y,
                    temperature=self.temperature,
                )
                if loss is None:
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                steps += 1

            mean_loss = running_loss / max(steps, 1)
            self.training_history_.append({"epoch": epoch, "loss": mean_loss})
            if self.logger and self.log_every > 0 and (epoch == 1 or epoch % self.log_every == 0 or epoch == self.epochs):
                self.logger(f"contrastive epoch {epoch}/{self.epochs}: loss={mean_loss:.4f}")

        train_projected = self.transform(X)
        centroids = []
        for class_id in (0, 1):
            centroid = np.mean(train_projected[y == class_id], axis=0)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
            centroids.append(centroid)
        self.class_centroids_ = np.stack(centroids, axis=0).astype(np.float32)
        return self

    def transform(self, X):
        X_scaled = self.scaler_.transform(X).astype(np.float32)
        self.projector_.eval()
        with torch.no_grad():
            tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            projected = self.projector_(tensor)
            projected = F.normalize(projected, dim=1)
        return projected.detach().cpu().numpy().astype(np.float32)

    def predict_proba(self, X):
        projected = self.transform(X)
        logits = projected @ self.class_centroids_.T / self.temperature
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.maximum(np.sum(exp_logits, axis=1, keepdims=True), 1e-12)
        return probs.astype(np.float32)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class AttentionFusionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.global_proj = nn.Linear(input_dim, hidden_dim)
        self.crop_proj = nn.Linear(input_dim, hidden_dim)
        self.modality_embed = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.classifier = nn.Linear(input_dim, 1)

    def encode(self, global_x, crop_x):
        tokens = torch.stack(
            [
                self.global_proj(global_x),
                self.crop_proj(crop_x),
            ],
            dim=1,
        )
        tokens = tokens + self.modality_embed.unsqueeze(0)
        query = self.query.expand(tokens.shape[0], -1, -1)
        attn_out, attn_weights = self.attn(query, tokens, tokens, need_weights=True)
        token_mean = tokens.mean(dim=1, keepdim=True)
        fused = self.post_norm(attn_out + token_mean)
        fused = fused + self.ffn(fused)
        fused_embedding = self.out_proj(fused.squeeze(1))
        fused_embedding = F.normalize(fused_embedding, dim=-1)
        return fused_embedding, attn_weights.squeeze(1)

    def forward(self, global_x, crop_x):
        fused_embedding, attn_weights = self.encode(global_x, crop_x)
        logits = self.classifier(fused_embedding).squeeze(-1)
        return logits, fused_embedding, attn_weights


def train_attention_encoder(
    X_global,
    X_crop,
    y_train,
    *,
    seed: int,
    val_fraction: float,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: str,
    log_every: int = 0,
    logger=None,
):
    resolved_device = detect_device(device)
    input_dim = X_global.shape[1]
    class_counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=2)
    if len(y_train) < 6 or np.min(class_counts[class_counts > 0], initial=0) < 2:
        train_idx = np.arange(len(y_train))
        val_idx = np.arange(len(y_train))
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(y_train)),
            test_size=val_fraction,
            stratify=y_train,
            random_state=seed,
        )

    Xg_subtrain = torch.tensor(X_global[train_idx], dtype=torch.float32)
    Xc_subtrain = torch.tensor(X_crop[train_idx], dtype=torch.float32)
    y_subtrain = torch.tensor(y_train[train_idx], dtype=torch.float32)
    Xg_val = torch.tensor(X_global[val_idx], dtype=torch.float32)
    Xc_val = torch.tensor(X_crop[val_idx], dtype=torch.float32)
    y_val = y_train[val_idx]

    loader = DataLoader(
        TensorDataset(Xg_subtrain, Xc_subtrain, y_subtrain),
        batch_size=min(batch_size, len(train_idx)),
        shuffle=True,
        drop_last=False,
    )

    model = AttentionFusionEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    ).to(resolved_device)

    positives = float(np.sum(y_train[train_idx] == 1))
    negatives = float(np.sum(y_train[train_idx] == 0))
    pos_weight = torch.tensor([max(negatives / max(positives, 1.0), 1.0)], dtype=torch.float32, device=resolved_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_steps = 0
        for batch_g, batch_c, batch_y in loader:
            batch_g = batch_g.to(resolved_device)
            batch_c = batch_c.to(resolved_device)
            batch_y = batch_y.to(resolved_device)

            optimizer.zero_grad()
            logits, _, _ = model(batch_g, batch_c)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_steps += 1

        model.eval()
        with torch.no_grad():
            val_logits, _, _ = model(Xg_val.to(resolved_device), Xc_val.to(resolved_device))
            val_probs = torch.sigmoid(val_logits).detach().cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(int)
        val_metrics = compute_metrics(y_val, val_preds, val_probs)
        mean_train_loss = train_loss / max(train_steps, 1)
        row = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_roc_auc": float(val_metrics.get("roc_auc", 0.0)),
        }
        history.append(row)

        score_tuple = (
            row["val_f1"],
            row["val_roc_auc"],
            row["val_accuracy"],
            -epoch,
        )
        if best is None or score_tuple > best["score_tuple"]:
            best = {
                "score_tuple": score_tuple,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "epoch": epoch,
                "metrics": row,
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if logger and log_every > 0 and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            logger(
                f"attention epoch {epoch}/{epochs}: "
                f"train_loss={mean_train_loss:.4f}, "
                f"val_f1={row['val_f1']:.4f}, "
                f"val_auc={row['val_roc_auc']:.4f}"
            )

        if bad_epochs >= patience:
            if logger:
                logger(f"attention early stop at epoch {epoch}; best epoch was {best['epoch']}")
            break

    model.load_state_dict(best["state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, history, best, resolved_device


def encode_attention_fusion(model, X_global, X_crop, device: str, batch_size: int):
    model.eval()
    fused_embeddings = []
    attn_weights = []
    with torch.no_grad():
        for start in range(0, len(X_global), batch_size):
            end = start + batch_size
            batch_g = torch.tensor(X_global[start:end], dtype=torch.float32, device=device)
            batch_c = torch.tensor(X_crop[start:end], dtype=torch.float32, device=device)
            _, fused, weights = model(batch_g, batch_c)
            fused_embeddings.append(fused.detach().cpu().numpy())
            attn_weights.append(weights.detach().cpu().numpy())

    return (
        np.concatenate(fused_embeddings, axis=0).astype(np.float32),
        np.concatenate(attn_weights, axis=0).astype(np.float32),
    )


def _normalize_label(text: str):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _resolve_grounding_signal_profile(texts):
    normalized_texts = [_normalize_label(text) for text in texts if _normalize_label(text)]
    best_label = "detected_region"
    best_weight = 1.0
    best_reason = "A detected UI region may contain the phishing prompt or CTA."
    for text in normalized_texts:
        for known_label, weight in GROUNDING_SIGNAL_LABEL_WEIGHTS.items():
            if text == known_label or text in known_label or known_label in text:
                if weight > best_weight:
                    best_label = known_label
                    best_weight = weight
                    best_reason = GROUNDING_SIGNAL_LABEL_REASONS[known_label]
        if best_label == "detected_region" and text:
            best_label = text
    return best_label, float(best_weight), best_reason


def build_ocr_crop_embedding(
    image_path: Path,
    ocr: OCRService,
    processor,
    model,
    device: str,
    batch_size: int,
    max_crops: int,
    min_confidence: float,
    include_processed: bool,
    padding_ratio: float,
    candidate_mode: str,
    pooling: str = "avg",
):
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
        image_width, image_height = image.size

        try:
            regions = ocr.detect_text_regions(
                str(image_path),
                min_confidence=min_confidence,
                include_processed=include_processed,
            )
            candidates, _ = generate_crop_candidates(
                regions,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=padding_ratio,
                candidate_mode=candidate_mode,
            )
        except Exception:
            candidates = [
                {
                    "candidate_type": "fallback_full_image",
                    "bbox": [0, 0, image_width, image_height],
                    "ocr_confidence_mean": 0.0,
                    "score_hint": 0.0,
                }
            ]

        if max_crops > 0:
            candidates = candidates[:max_crops]

        crop_images = []
        crop_weights = []
        try:
            for candidate in candidates:
                bbox = tuple(candidate["bbox"])
                crop_images.append(image.crop(bbox).copy())
                area_ratio = _bbox_area(candidate["bbox"]) / max(image_width * image_height, 1)
                crop_weights.append(
                    max(
                        float(candidate.get("score_hint", 0.0))
                        + float(candidate.get("ocr_confidence_mean", 0.0))
                        + area_ratio,
                        0.05,
                    )
                )

            crop_features = embed_pil_images(
                crop_images,
                processor=processor,
                model=model,
                device=device,
                batch_size=batch_size,
            )
        finally:
            for crop_image in crop_images:
                crop_image.close()

    candidate_types = "|".join(candidate.get("candidate_type", "unknown") for candidate in candidates)
    return pool_crop_features(crop_features, crop_weights, pooling=pooling), {
        "crop_count": len(candidates),
        "candidate_types": candidate_types,
    }


def build_crop_feature_matrix(
    rows,
    ocr: OCRService,
    processor,
    model,
    device: str,
    batch_size: int,
    max_crops: int,
    min_confidence: float,
    include_processed: bool,
    padding_ratio: float,
    candidate_mode: str,
    pooling: str = "avg",
    stage_name: str | None = None,
    log_every: int = 0,
    logger=None,
):
    features = []
    crop_rows = []
    total = len(rows)
    last_logged = 0
    for idx, row in enumerate(rows, start=1):
        embedding, crop_meta = build_ocr_crop_embedding(
            Path(row["image_path"]),
            ocr=ocr,
            processor=processor,
            model=model,
            device=device,
            batch_size=batch_size,
            max_crops=max_crops,
            min_confidence=min_confidence,
            include_processed=include_processed,
            padding_ratio=padding_ratio,
            candidate_mode=candidate_mode,
            pooling=pooling,
        )
        features.append(embedding)
        crop_rows.append(
            {
                "split": row["split"],
                "class_name": row["class_name"],
                "label_id": row["label_id"],
                "image_path": row["image_path"],
                "source_name": row["source_name"],
                "priority_score": row["priority_score"],
                "matched_tokens": row.get("matched_tokens", ""),
                "crop_count": crop_meta["crop_count"],
                "candidate_types": crop_meta["candidate_types"],
            }
        )

        if logger and log_every > 0 and (idx - last_logged >= log_every or idx == total):
            logger(f"{stage_name or 'ocr'}: processed {idx}/{total} images")
            last_logged = idx

    return np.stack(features, axis=0), crop_rows


def parse_grounding_prompt_labels(raw_labels: str):
    labels = [part.strip().lower() for part in raw_labels.split(",") if part.strip()]
    return labels or list(DEFAULT_PROMPT_LABELS)


def load_grounding_dino(model_name: str, device: str):
    resolved_device = detect_device(device)
    processor, model = load_grounding_dino_processor_and_model(model_name, resolved_device)
    return processor, model, resolved_device


def build_grounding_dino_crop_embedding(
    row,
    detector_processor,
    detector_model,
    siglip_processor,
    siglip_model,
    device: str,
    batch_size: int,
    prompt_labels,
    box_threshold: float,
    text_threshold: float,
    padding_ratio: float,
    max_crops: int,
    pooling: str = "avg",
    signal_output_root: Path | None = None,
):
    image_path = Path(row["image_path"])
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
        image_width, image_height = image.size

        try:
            detections = detect_grounding_dino_regions(
                image=image,
                processor=detector_processor,
                model=detector_model,
                device=device,
                prompt_labels=prompt_labels,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            candidates = build_grounding_dino_candidates(
                detections,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=padding_ratio,
            )
        except Exception:
            candidates = [
                {
                    "candidate_type": "fallback_full_image",
                    "bbox": [0, 0, image_width, image_height],
                    "texts": [],
                    "ocr_confidence_mean": 0.0,
                    "score_hint": 0.0,
                }
            ]

        if max_crops > 0:
            candidates = candidates[:max_crops]

        crop_images = []
        crop_weights = []
        signal_scores = []
        candidate_labels = []
        candidate_reasons = []
        try:
            for candidate in candidates:
                bbox = tuple(candidate["bbox"])
                crop_images.append(image.crop(bbox).copy())

                label, label_weight, reason = _resolve_grounding_signal_profile(candidate.get("texts", []))
                area_ratio = _bbox_area(candidate["bbox"]) / max(image_width * image_height, 1)
                detector_confidence = float(candidate.get("ocr_confidence_mean", 0.0))
                signal_score = label_weight + detector_confidence + area_ratio * 2.0
                pooling_weight = max(
                    float(candidate.get("score_hint", 0.0))
                    + label_weight
                    + detector_confidence
                    + area_ratio,
                    0.05,
                )

                candidate_labels.append(label)
                candidate_reasons.append(reason)
                signal_scores.append(signal_score)
                crop_weights.append(pooling_weight)

            crop_features = embed_pil_images(
                crop_images,
                processor=siglip_processor,
                model=siglip_model,
                device=device,
                batch_size=batch_size,
            )

            selected_signal_index = int(np.argmax(np.asarray(signal_scores, dtype=np.float32)))
            selected_signal_path = ""
            if signal_output_root is not None:
                selected_signal_dir = (
                    signal_output_root
                    / row["split"]
                    / row["class_name"]
                )
                ensure_dir(selected_signal_dir)
                selected_signal_path = str(
                    (selected_signal_dir / f"{safe_slug(image_path)}__signal.png").resolve()
                )
                crop_images[selected_signal_index].save(selected_signal_path)
        finally:
            for crop_image in crop_images:
                crop_image.close()

    selected_candidate = candidates[selected_signal_index]
    candidate_types = "|".join(candidate.get("candidate_type", "unknown") for candidate in candidates)
    return pool_crop_features(crop_features, crop_weights, pooling=pooling), {
        "crop_count": len(candidates),
        "candidate_types": candidate_types,
        "selected_signal_index": selected_signal_index + 1,
        "selected_signal_label": candidate_labels[selected_signal_index],
        "selected_signal_reason": candidate_reasons[selected_signal_index],
        "selected_signal_score": float(signal_scores[selected_signal_index]),
        "selected_signal_bbox_x1": int(selected_candidate["bbox"][0]),
        "selected_signal_bbox_y1": int(selected_candidate["bbox"][1]),
        "selected_signal_bbox_x2": int(selected_candidate["bbox"][2]),
        "selected_signal_bbox_y2": int(selected_candidate["bbox"][3]),
        "selected_signal_crop_path": selected_signal_path,
        "prompt_labels": "|".join(prompt_labels),
    }


def build_grounding_dino_feature_matrix(
    rows,
    detector_processor,
    detector_model,
    siglip_processor,
    siglip_model,
    device: str,
    batch_size: int,
    prompt_labels,
    box_threshold: float,
    text_threshold: float,
    padding_ratio: float,
    max_crops: int,
    pooling: str = "avg",
    signal_output_root: Path | None = None,
    stage_name: str | None = None,
    log_every: int = 0,
    logger=None,
):
    features = []
    crop_rows = []
    total = len(rows)
    last_logged = 0
    for idx, row in enumerate(rows, start=1):
        embedding, crop_meta = build_grounding_dino_crop_embedding(
            row=row,
            detector_processor=detector_processor,
            detector_model=detector_model,
            siglip_processor=siglip_processor,
            siglip_model=siglip_model,
            device=device,
            batch_size=batch_size,
            prompt_labels=prompt_labels,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            padding_ratio=padding_ratio,
            max_crops=max_crops,
            pooling=pooling,
            signal_output_root=signal_output_root,
        )
        features.append(embedding)
        crop_rows.append(
            {
                "split": row["split"],
                "class_name": row["class_name"],
                "label_id": row["label_id"],
                "image_path": row["image_path"],
                "source_name": row["source_name"],
                "priority_score": row["priority_score"],
                "matched_tokens": row.get("matched_tokens", ""),
                "crop_count": crop_meta["crop_count"],
                "candidate_types": crop_meta["candidate_types"],
                "selected_signal_index": crop_meta["selected_signal_index"],
                "selected_signal_label": crop_meta["selected_signal_label"],
                "selected_signal_reason": crop_meta["selected_signal_reason"],
                "selected_signal_score": crop_meta["selected_signal_score"],
                "selected_signal_bbox_x1": crop_meta["selected_signal_bbox_x1"],
                "selected_signal_bbox_y1": crop_meta["selected_signal_bbox_y1"],
                "selected_signal_bbox_x2": crop_meta["selected_signal_bbox_x2"],
                "selected_signal_bbox_y2": crop_meta["selected_signal_bbox_y2"],
                "selected_signal_crop_path": crop_meta["selected_signal_crop_path"],
                "prompt_labels": crop_meta["prompt_labels"],
            }
        )

        if logger and log_every > 0 and (idx - last_logged >= log_every or idx == total):
            logger(f"{stage_name or 'grounding_dino'}: processed {idx}/{total} images")
            last_logged = idx

    return np.stack(features, axis=0), crop_rows


def train_classifier(
    model_name: str,
    X_train,
    y_train,
    *,
    seed: int = 42,
    contrastive_hidden_dim: int = 256,
    contrastive_projection_dim: int = 128,
    contrastive_epochs: int = 60,
    contrastive_batch_size: int = 64,
    contrastive_learning_rate: float = 1e-3,
    contrastive_weight_decay: float = 1e-4,
    contrastive_temperature: float = 0.10,
    contrastive_device: str = "auto",
    log_every: int = 0,
    logger=None,
):
    canonical = canonicalize_classifier_name(model_name)
    if canonical == "contrastive":
        clf = ContrastiveProjectionClassifier(
            seed=seed,
            hidden_dim=contrastive_hidden_dim,
            projection_dim=contrastive_projection_dim,
            epochs=contrastive_epochs,
            batch_size=contrastive_batch_size,
            learning_rate=contrastive_learning_rate,
            weight_decay=contrastive_weight_decay,
            temperature=contrastive_temperature,
            device=contrastive_device,
            log_every=log_every,
            logger=logger,
        )
    else:
        clf = build_classifier(canonical, y_train)
    clf.fit(X_train, y_train)
    return clf, canonical


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def evaluate_classifier(clf, X_test, y_test, rows):
    y_prob = predict_positive_scores(clf, X_test)
    if y_prob is not None:
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = clf.predict(X_test)
        y_prob = np.asarray(y_pred, dtype=np.float32)
    metrics = compute_metrics(y_test, y_pred, y_prob)

    prediction_rows = []
    for row, prob, pred in zip(rows, y_prob, y_pred):
        prediction_rows.append(
            {
                "split": row["split"],
                "class_name": row["class_name"],
                "label_id": row["label_id"],
                "pred_label_id": int(pred),
                "fake_probability": float(prob),
                "correct": int(int(pred) == int(row["label_id"])),
                "image_path": row["image_path"],
                "source_name": row["source_name"],
                "priority_score": row.get("priority_score", ""),
                "matched_tokens": row.get("matched_tokens", ""),
                "crop_count": row.get("crop_count", ""),
                "candidate_types": row.get("candidate_types", ""),
                "selected_signal_label": row.get("selected_signal_label", ""),
                "selected_signal_reason": row.get("selected_signal_reason", ""),
                "selected_signal_score": row.get("selected_signal_score", ""),
                "selected_signal_bbox_x1": row.get("selected_signal_bbox_x1", ""),
                "selected_signal_bbox_y1": row.get("selected_signal_bbox_y1", ""),
                "selected_signal_bbox_x2": row.get("selected_signal_bbox_x2", ""),
                "selected_signal_bbox_y2": row.get("selected_signal_bbox_y2", ""),
                "selected_signal_crop_path": row.get("selected_signal_crop_path", ""),
                "global_attention_weight": row.get("global_attention_weight", ""),
                "crop_attention_weight": row.get("crop_attention_weight", ""),
            }
        )
    return metrics, prediction_rows


def load_siglip(model_name: str, device: str):
    resolved_device = detect_device(device)
    processor, model = load_siglip_processor_and_model(model_name, resolved_device)
    return processor, model, resolved_device


def save_json(path: Path, payload):
    ensure_dir(path.parent)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def save_text(path: Path, text: str):
    ensure_dir(path.parent)
    with open(path, "w") as handle:
        handle.write(text)


def save_csv(path: Path, rows):
    ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        with open(path, "w", newline="") as handle:
            handle.write("")
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: Path, X, y, paths):
    ensure_dir(path.parent)
    np.savez_compressed(path, X=X, y=y, paths=np.asarray(paths))


def format_metrics_markdown(title: str, metrics, extra_lines=None):
    lines = [f"# {title}", "", "## Metrics"]
    for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if metric_name in metrics:
            lines.append(f"- {metric_name}: {metrics[metric_name]:.4f}")
    lines.append(f"- confusion_matrix: {metrics['confusion_matrix']}")
    if extra_lines:
        lines.extend(["", "## Notes", *extra_lines])
    lines.append("")
    return "\n".join(lines)


def summarize_crop_rows(crop_rows):
    crop_counts = [int(row["crop_count"]) for row in crop_rows]
    if not crop_counts:
        return {"mean_crop_count": 0.0, "median_crop_count": 0.0, "max_crop_count": 0}
    return {
        "mean_crop_count": float(np.mean(crop_counts)),
        "median_crop_count": float(np.median(crop_counts)),
        "max_crop_count": int(np.max(crop_counts)),
    }


def build_comparison(
    global_metrics,
    crop_metrics,
    split_summary_payload,
    crop_summary,
    device: str,
    segmentation_key: str,
    global_key: str = "global_siglip_lr",
    fusion_metrics=None,
    fusion_key: str | None = None,
):
    comparison = {
        "device": device,
        "split_summary": split_summary_payload,
        global_key: global_metrics,
        segmentation_key: crop_metrics,
        "segmentation_method": segmentation_key,
        "crop_summary": crop_summary,
        "delta_accuracy": float(crop_metrics["accuracy"] - global_metrics["accuracy"]),
        "delta_f1": float(crop_metrics["f1"] - global_metrics["f1"]),
        "delta_roc_auc": float(
            crop_metrics.get("roc_auc", math.nan) - global_metrics.get("roc_auc", math.nan)
        ),
    }
    if fusion_metrics is not None and fusion_key is not None:
        comparison[fusion_key] = fusion_metrics
        comparison["fusion_method"] = fusion_key
        comparison["delta_fusion_vs_global_accuracy"] = float(
            fusion_metrics["accuracy"] - global_metrics["accuracy"]
        )
        comparison["delta_fusion_vs_global_f1"] = float(
            fusion_metrics["f1"] - global_metrics["f1"]
        )
        comparison["delta_fusion_vs_global_roc_auc"] = float(
            fusion_metrics.get("roc_auc", math.nan) - global_metrics.get("roc_auc", math.nan)
        )
        comparison["delta_fusion_vs_crop_accuracy"] = float(
            fusion_metrics["accuracy"] - crop_metrics["accuracy"]
        )
        comparison["delta_fusion_vs_crop_f1"] = float(
            fusion_metrics["f1"] - crop_metrics["f1"]
        )
        comparison["delta_fusion_vs_crop_roc_auc"] = float(
            fusion_metrics.get("roc_auc", math.nan) - crop_metrics.get("roc_auc", math.nan)
        )
    return comparison
