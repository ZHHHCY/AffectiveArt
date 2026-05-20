"""Shared utilities for the emotion art generation pipeline."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image


def get_project_root() -> Path:
    """Return emotion_art_generation/ directory."""
    return Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path | None, base: Path | None = None) -> Path:
    """Resolve a path relative to project root or explicit base."""
    if path is None:
        raise ValueError("Path cannot be None")
    p = Path(path)
    if p.is_absolute():
        return p
    root = base or get_project_root()
    return (root / p).resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Provide a valid path, e.g. configs/train_lora.yaml"
        )
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    # Resolve project_root
    if cfg.get("project_root") is None:
        cfg["project_root"] = str(get_project_root().parent)
    return cfg


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(name: str = "emotion_art", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_annotations(annotation_path: str | Path) -> list[dict[str, Any]]:
    """Load annotation JSON with helpful error messages."""
    path = resolve_path(annotation_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {path}\n"
            "Expected EmoArt annotations at Images/annotation_5k.json.\n"
            "Run scripts/prepare_dataset.py to validate paths."
        )
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"Annotation file is empty or invalid: {path}")
    return records


def extract_style_from_record(record: dict[str, Any]) -> str:
    """Extract painting style label from request_id."""
    request_id = record.get("request_id", "")
    if "_request-" in request_id:
        return request_id.rsplit("_request-", 1)[0]
    raise KeyError(f"Cannot parse style from request_id: {request_id}")


def extract_emotion_from_record(record: dict[str, Any]) -> str:
    """Extract dominant emotion from annotation."""
    try:
        return record["description"]["third_section"]["dominant_emotion"]
    except KeyError as e:
        raise KeyError(f"Missing dominant_emotion in record {record.get('request_id')}") from e


def get_style_emotion_maps(
    annotation_path: str | Path,
) -> tuple[dict[str, int], dict[str, int], dict[int, str], dict[int, str]]:
    """
    Build deterministic label mappings from annotations.

    Returns:
        style2id, emotion2id, id2style, id2emotion
    """
    records = load_annotations(annotation_path)
    styles = sorted({extract_style_from_record(r) for r in records})
    emotions = sorted({extract_emotion_from_record(r) for r in records})
    style2id = {s: i for i, s in enumerate(styles)}
    emotion2id = {e: i for i, e in enumerate(emotions)}
    id2style = {i: s for s, i in style2id.items()}
    id2emotion = {i: e for e, i in emotion2id.items()}
    return style2id, emotion2id, id2style, id2emotion


def normalize_image_path(image_path: str) -> str:
    """Convert Windows-style paths to forward slashes."""
    return image_path.replace("\\", "/")


def resolve_image_path(
    record: dict[str, Any],
    image_root: str | Path,
) -> Path:
    """Resolve absolute path to image file from annotation record."""
    rel = normalize_image_path(record["image_path"])
    root = resolve_path(image_root)
    full = (root / rel).resolve()
    if not full.exists():
        raise FileNotFoundError(
            f"Image not found: {full}\n"
            f"  annotation image_path: {record.get('image_path')}\n"
            f"  image_root: {root}\n"
            "Check that EmoArt images are extracted under the image_root directory."
        )
    return full


def save_image_grid(
    images: list[Image.Image],
    path: str | Path,
    nrow: int = 4,
) -> None:
    """Save a grid of PIL images."""
    import torchvision.utils as vutils

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = []
    for img in images:
        if isinstance(img, Image.Image):
            import torchvision.transforms as T

            t = T.ToTensor()(img)
        else:
            t = img
        tensors.append(t)
    grid = vutils.make_grid(torch.stack(tensors), nrow=nrow, padding=2)
    vutils.save_image(grid, str(path))


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
