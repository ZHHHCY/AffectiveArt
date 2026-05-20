"""Train CLIP-based style and emotion classifiers on EmoArt."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel

from src.dataset import EmoArtDataset, collate_fn
from src.models.classifiers import EmotionClassifier, StyleClassifier
from src.utils import ensure_dir, get_project_root, load_config, set_seed, setup_logging


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights."""
    counts = Counter(labels)
    weights = []
    total = len(labels)
    for c in range(num_classes):
        n = counts.get(c, 1)
        weights.append(total / (num_classes * n))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[float, float]:
    """Return (accuracy, avg_loss)."""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for batch in loader:
        pixels = batch["clip_pixel_values"].to(device)
        if isinstance(model, StyleClassifier):
            targets = batch["style_id"].to(device)
        else:
            targets = batch["emotion_id"].to(device)
        logits = model(pixels)
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        if criterion is not None:
            loss_sum += criterion(logits, targets).item() * targets.size(0)
    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1) if criterion else 0.0
    return acc, avg_loss


def train_one_classifier(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict,
    style2id: dict,
    emotion2id: dict,
    label_key: str,
) -> None:
    """Train a single classifier and save best checkpoint."""
    logger = setup_logging(f"train_{name}")
    out_dir = ensure_dir(Path(cfg["training"]["output_dir"]))
    num_epochs = cfg["training"]["num_epochs"]
    lr = cfg["training"]["learning_rate"]

    # Class weights for emotion imbalance
    if cfg["training"].get("use_class_weights", False) and label_key == "emotion_id":
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch["emotion_id"].tolist())
        num_classes = len(emotion2id)
        weights = compute_class_weights(all_labels, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"{name} epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            pixels = batch["clip_pixel_values"].to(device)
            targets = batch[label_key].to(device)
            optimizer.zero_grad()
            logits = model(pixels)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        val_acc, val_loss = evaluate(model, val_loader, device, criterion)
        logger.info(
            f"Epoch {epoch+1}: train_loss={running_loss/len(train_loader):.4f} "
            f"val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "style2id": style2id,
                "emotion2id": emotion2id,
                "config": cfg,
            }
            path = out_dir / f"{name}_best.pt"
            torch.save(ckpt, path)
            logger.info(f"Saved best {name} checkpoint -> {path} (acc={val_acc:.4f})")

        if (epoch + 1) % cfg["training"].get("save_every_epochs", 5) == 0:
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                out_dir / f"{name}_epoch{epoch+1}.pt",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train style/emotion classifiers")
    parser.add_argument("--config", type=str, default="configs/train_classifiers.yaml")
    parser.add_argument("--task", type=str, default=None, choices=["style", "emotion", "both"])
    args = parser.parse_args()

    cfg = load_config(get_project_root() / args.config)
    if args.task:
        cfg["training"]["task"] = args.task

    set_seed(cfg["training"]["seed"])
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    root = get_project_root()
    ann_path = root / cfg["data"]["annotation_path"]
    image_root = root / cfg["data"]["image_root"]

    train_ds = EmoArtDataset(
        annotation_path=str(ann_path),
        image_root=str(image_root),
        split="train",
        resolution=cfg["data"]["resolution"],
        center_crop=cfg["data"].get("center_crop", True),
        val_frac=cfg["data"]["val_frac"],
        test_frac=cfg["data"]["test_frac"],
        split_seed=cfg["data"]["split_seed"],
        build_prompt=False,
    )
    val_ds = EmoArtDataset(
        annotation_path=str(ann_path),
        image_root=str(image_root),
        split="val",
        resolution=cfg["data"]["resolution"],
        center_crop=cfg["data"].get("center_crop", True),
        val_frac=cfg["data"]["val_frac"],
        test_frac=cfg["data"]["test_frac"],
        split_seed=cfg["data"]["split_seed"],
        build_prompt=False,
    )

    # Weighted sampler for emotion training
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        collate_fn=collate_fn,
    )

    clip_model = CLIPModel.from_pretrained(cfg["model"]["clip_model"])
    num_styles = len(train_ds.style2id)
    num_emotions = len(train_ds.emotion2id)
    logger.info(f"Styles: {num_styles}, Emotions: {num_emotions}")

    task = cfg["training"]["task"]
    if task in ("style", "both"):
        style_clf = StyleClassifier(
            clip_model,
            num_classes=num_styles,
            hidden_dim=cfg["model"]["mlp_hidden_dim"],
            dropout=cfg["model"]["dropout"],
            freeze_clip=cfg["model"].get("freeze_clip", True),
        ).to(device)
        train_one_classifier(
            "style",
            style_clf,
            train_loader,
            val_loader,
            device,
            cfg,
            train_ds.style2id,
            train_ds.emotion2id,
            "style_id",
        )

    if task in ("emotion", "both"):
        # Reload CLIP for separate emotion model to avoid shared head interference
        clip_model_e = CLIPModel.from_pretrained(cfg["model"]["clip_model"])
        emotion_clf = EmotionClassifier(
            clip_model_e,
            num_classes=num_emotions,
            hidden_dim=cfg["model"]["mlp_hidden_dim"],
            dropout=cfg["model"]["dropout"],
            freeze_clip=cfg["model"].get("freeze_clip", True),
        ).to(device)
        train_one_classifier(
            "emotion",
            emotion_clf,
            train_loader,
            val_loader,
            device,
            cfg,
            train_ds.style2id,
            train_ds.emotion2id,
            "emotion_id",
        )

    logger.info("Classifier training complete.")


if __name__ == "__main__":
    main()
