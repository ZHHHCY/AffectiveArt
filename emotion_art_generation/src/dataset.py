"""EmoArt-5k dataset loader with stratified splits."""

from __future__ import annotations

from typing import Any, Literal

import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

from src.prompt_builder import PromptBuilder
from src.utils import (
    extract_emotion_from_record,
    extract_style_from_record,
    get_style_emotion_maps,
    load_annotations,
    normalize_image_path,
    resolve_image_path,
)


def parse_record(
    record: dict[str, Any],
    style2id: dict[str, int],
    emotion2id: dict[str, int],
) -> dict[str, Any]:
    """Parse a single annotation record into a flat sample dict."""
    style = extract_style_from_record(record)
    emotion = extract_emotion_from_record(record)
    desc = record["description"]
    first = desc["first_section"]
    second = desc["second_section"]
    third = desc["third_section"]
    attrs = second.get("visual_attributes", {})

    return {
        "image_path": normalize_image_path(record["image_path"]),
        "request_id": record.get("request_id", ""),
        "style": style,
        "style_id": style2id[style],
        "emotion": emotion,
        "emotion_id": emotion2id[emotion],
        "content": first.get("description", ""),
        "brushstroke": attrs.get("brushstroke", ""),
        "color": attrs.get("color", ""),
        "composition": attrs.get("composition", ""),
        "light_and_shadow": attrs.get("light_and_shadow", ""),
        "line_quality": attrs.get("line_quality", ""),
        "emotional_impact": second.get("emotional_impact", ""),
        "valence": third.get("emotional_valence", ""),
        "arousal": third.get("emotional_arousal_level", ""),
        "healing_effects": third.get("healing_effects", []),
    }


def build_splits(
    records: list[dict[str, Any]],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Stratified train/val/test split by painting style.

    Returns:
        train_indices, val_indices, test_indices
    """
    n = len(records)
    styles = [extract_style_from_record(r) for r in records]
    indices = list(range(n))

    # First split: train+val vs test
    test_size = test_frac
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(indices, styles))

    # Second split: train vs val from train_val
    val_relative = val_frac / (1.0 - test_frac) if test_frac < 1.0 else val_frac
    train_val_styles = [styles[i] for i in train_val_idx]
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed + 1)
    train_sub, val_sub = next(sss_val.split(list(range(len(train_val_idx))), train_val_styles))

    train_idx = [train_val_idx[i] for i in train_sub]
    val_idx = [train_val_idx[i] for i in val_sub]
    test_idx = list(test_idx)

    return train_idx, val_idx, test_idx


class EmoArtDataset(Dataset):
    """
    EmoArt dataset for diffusion fine-tuning and classifier training.

    Each sample returns a dict with image tensors, labels, and metadata.
    """

    def __init__(
        self,
        annotation_path: str,
        image_root: str,
        split: Literal["train", "val", "test"] = "train",
        resolution: int = 512,
        center_crop: bool = True,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        split_seed: int = 42,
        return_pil: bool = False,
        build_prompt: bool = True,
        indices: list[int] | None = None,
    ) -> None:
        self.image_root = image_root
        self.split = split
        self.resolution = resolution
        self.return_pil = return_pil
        self.build_prompt = build_prompt
        self.prompt_builder = PromptBuilder()

        records = load_annotations(annotation_path)
        self.style2id, self.emotion2id, self.id2style, self.id2emotion = get_style_emotion_maps(
            annotation_path
        )

        train_idx, val_idx, test_idx = build_splits(
            records, val_frac=val_frac, test_frac=test_frac, seed=split_seed
        )
        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        chosen = indices if indices is not None else split_map[split]
        self.records = [records[i] for i in chosen]
        self.samples = [parse_record(r, self.style2id, self.emotion2id) for r in self.records]

        # Image transforms
        transform_list = []
        if center_crop:
            transform_list.append(
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
            )
            transform_list.append(transforms.CenterCrop(resolution))
        else:
            transform_list.append(
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR)
            )
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.transform = transforms.Compose(transform_list)

        # CLIP classifier transform (224)
        clip_list = [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
        self.clip_transform = transforms.Compose(clip_list)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, record: dict[str, Any]) -> Image.Image:
        path = resolve_image_path(record, self.image_root)
        return Image.open(path).convert("RGB")

    def get_attribute_text(self, sample: dict[str, Any]) -> str:
        """Combine visual attribute fields for attribute alignment loss."""
        parts = [
            sample.get("brushstroke", ""),
            sample.get("color", ""),
            sample.get("composition", ""),
            sample.get("light_and_shadow", ""),
            sample.get("line_quality", ""),
            sample.get("emotional_impact", ""),
        ]
        return " ".join(p for p in parts if p).strip()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        sample = self.samples[idx].copy()

        image = self._load_image(record)
        sample["pixel_values"] = self.transform(image)
        sample["clip_pixel_values"] = self.clip_transform(image)
        sample["attribute_text"] = self.get_attribute_text(sample)

        if self.build_prompt:
            sample["prompt"] = self.prompt_builder.build_train_prompt(sample)
        else:
            sample["prompt"] = ""

        if self.return_pil:
            sample["image_pil"] = image

        # Tensor labels for training
        sample["style_id"] = torch.tensor(sample["style_id"], dtype=torch.long)
        sample["emotion_id"] = torch.tensor(sample["emotion_id"], dtype=torch.long)

        return sample


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate batch for DataLoader."""
    out: dict[str, Any] = {}
    keys_tensor = ["pixel_values", "clip_pixel_values", "style_id", "emotion_id"]
    keys_list = [
        "prompt",
        "content",
        "style",
        "emotion",
        "attribute_text",
        "image_path",
        "brushstroke",
        "color",
        "composition",
        "light_and_shadow",
        "line_quality",
        "emotional_impact",
        "valence",
        "arousal",
    ]
    for k in keys_tensor:
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    for k in keys_list:
        if k in batch[0]:
            out[k] = [b[k] for b in batch]
    return out
