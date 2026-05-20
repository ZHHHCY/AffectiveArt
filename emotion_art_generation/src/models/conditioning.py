"""Style-emotion conditioning modules for diffusion training."""

from __future__ import annotations

import torch
import torch.nn as nn


class StyleEmotionConditioner(nn.Module):
    """
    Learnable style and emotion embeddings combined with text encoder outputs.

    Modes:
        additive: text_embeds + scale_s * e_style + scale_e * e_emotion
        concat: project [text; e_style; e_emotion] back to text dim
    """

    def __init__(
        self,
        num_styles: int,
        num_emotions: int,
        embed_dim: int = 768,
        mode: str = "additive",
        style_scale: float = 1.0,
        emotion_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.style_scale = style_scale
        self.emotion_scale = emotion_scale
        self.style_embed = nn.Embedding(num_styles, embed_dim)
        self.emotion_embed = nn.Embedding(num_emotions, embed_dim)

        if mode == "concat":
            self.projection = nn.Linear(embed_dim * 3, embed_dim)
        elif mode != "additive":
            raise ValueError(f"Unknown conditioning mode: {mode}")

        self._init_embeddings()

    def _init_embeddings(self) -> None:
        nn.init.normal_(self.style_embed.weight, std=0.02)
        nn.init.normal_(self.emotion_embed.weight, std=0.02)

    def get_style_embedding(self, style_ids: torch.Tensor) -> torch.Tensor:
        return self.style_embed(style_ids)

    def get_emotion_embedding(self, emotion_ids: torch.Tensor) -> torch.Tensor:
        return self.emotion_embed(emotion_ids)

    def forward(
        self,
        text_embeds: torch.Tensor,
        style_ids: torch.Tensor,
        emotion_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            text_embeds: [B, seq_len, D] or [B, D]
            style_ids: [B]
            emotion_ids: [B]

        Returns:
            conditioned_embeds, e_style, e_emotion (per-sample, [B, D])
        """
        e_style = self.get_style_embedding(style_ids)
        e_emotion = self.get_emotion_embedding(emotion_ids)

        if text_embeds.dim() == 3:
            # Broadcast style/emotion across sequence length
            e_style_exp = e_style.unsqueeze(1)
            e_emotion_exp = e_emotion.unsqueeze(1)
        else:
            e_style_exp = e_style
            e_emotion_exp = e_emotion

        if self.mode == "additive":
            conditioned = (
                text_embeds
                + self.style_scale * e_style_exp
                + self.emotion_scale * e_emotion_exp
            )
        else:
            if text_embeds.dim() == 3:
                # Use first token representation for concat then broadcast
                text_vec = text_embeds[:, 0, :]
            else:
                text_vec = text_embeds
            fused = torch.cat([text_vec, e_style, e_emotion], dim=-1)
            projected = self.projection(fused)
            if text_embeds.dim() == 3:
                conditioned = text_embeds.clone()
                conditioned[:, 0, :] = projected
            else:
                conditioned = projected

        return conditioned, e_style, e_emotion

    def extra_tokens(
        self,
        style_ids: torch.Tensor,
        emotion_ids: torch.Tensor,
    ) -> list[str]:
        """Optional special token strings for logging."""
        return [f"<style_{i}>" for i in style_ids.tolist()] + [
            f"<emotion_{i}>" for i in emotion_ids.tolist()
        ]
