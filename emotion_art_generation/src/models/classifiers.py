"""CLIP-based style and emotion classifiers."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPVisionModel


class MLPHead(nn.Module):
    """Two-layer MLP classification head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CLIPClassifierBase(nn.Module):
    """Base classifier with frozen CLIP vision encoder."""

    def __init__(
        self,
        clip_model: CLIPModel | CLIPVisionModel,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        freeze_clip: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(clip_model, CLIPModel):
            self.vision = clip_model.vision_model
            embed_dim = clip_model.config.projection_dim
            # Use pooled output before projection for richer features
            self._use_clip_model = True
            self.clip_full = clip_model
        else:
            self.vision = clip_model
            embed_dim = clip_model.config.hidden_size
            self._use_clip_model = False
            self.clip_full = None

        if freeze_clip:
            for p in self.vision.parameters():
                p.requires_grad = False

        self.head = MLPHead(embed_dim, hidden_dim, num_classes, dropout)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._use_clip_model and self.clip_full is not None:
            vision_out = self.vision(pixel_values=pixel_values)
            pooled = vision_out.pooler_output
            image_features = self.clip_full.visual_projection(pooled)
        else:
            vision_out = self.vision(pixel_values=pixel_values)
            pooled = vision_out.pooler_output
            image_features = pooled
        return image_features

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.encode_image(pixel_values)
        return self.head(features)

    def predict_proba(self, pixel_values: torch.Tensor) -> torch.Tensor:
        logits = self.forward(pixel_values)
        return torch.softmax(logits, dim=-1)


class StyleClassifier(_CLIPClassifierBase):
    """Predict painting style from image."""

    def __init__(
        self,
        clip_model: CLIPModel | CLIPVisionModel,
        num_classes: int = 56,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        freeze_clip: bool = True,
    ) -> None:
        super().__init__(clip_model, hidden_dim, num_classes, dropout, freeze_clip)


class EmotionClassifier(_CLIPClassifierBase):
    """Predict dominant emotion from image."""

    def __init__(
        self,
        clip_model: CLIPModel | CLIPVisionModel,
        num_classes: int = 12,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        freeze_clip: bool = True,
    ) -> None:
        super().__init__(clip_model, hidden_dim, num_classes, dropout, freeze_clip)


def load_classifiers_from_checkpoint(
    checkpoint_dir: str,
    clip_model_name: str,
    num_style_classes: int,
    num_emotion_classes: int,
    hidden_dim: int = 512,
    dropout: float = 0.3,
    device: str | torch.device = "cpu",
) -> tuple[StyleClassifier, EmotionClassifier, dict[str, int], dict[str, int]]:
    """Load trained style and emotion classifiers."""
    from pathlib import Path

    device = torch.device(device)
    clip = CLIPModel.from_pretrained(clip_model_name)
    style_clf = StyleClassifier(clip, num_style_classes, hidden_dim, dropout)
    emotion_clf = EmotionClassifier(clip, num_emotion_classes, hidden_dim, dropout)

    ckpt_dir = Path(checkpoint_dir)
    style_path = ckpt_dir / "style_best.pt"
    emotion_path = ckpt_dir / "emotion_best.pt"
    if not style_path.exists():
        raise FileNotFoundError(f"Style classifier checkpoint not found: {style_path}")
    if not emotion_path.exists():
        raise FileNotFoundError(f"Emotion classifier checkpoint not found: {emotion_path}")

    style_ckpt = torch.load(style_path, map_location=device, weights_only=False)
    emotion_ckpt = torch.load(emotion_path, map_location=device, weights_only=False)
    style_clf.load_state_dict(style_ckpt["model_state_dict"])
    emotion_clf.load_state_dict(emotion_ckpt["model_state_dict"])
    style_clf.eval().to(device)
    emotion_clf.eval().to(device)

    return (
        style_clf,
        emotion_clf,
        style_ckpt.get("style2id", {}),
        emotion_ckpt.get("emotion2id", {}),
    )
