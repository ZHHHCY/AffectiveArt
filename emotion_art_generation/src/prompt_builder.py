"""Prompt templates for training and inference."""

from __future__ import annotations

import random
from typing import Any


# Visual modifiers keyed by dominant emotion for richer inference prompts
EMOTION_VISUAL_MODIFIERS: dict[str, str] = {
    "Calm": "soft lighting, muted colors, low contrast, and peaceful composition",
    "Contentment": "warm balanced tones, gentle brushwork, and harmonious layout",
    "Happy": "bright saturated colors, lively contrast, and uplifting composition",
    "Glad": "radiant highlights, vivid palette, and joyful spatial rhythm",
    "Excited": "dynamic angles, high contrast, energetic brushstrokes, and vivid colors",
    "Aroused": "intense saturation, dramatic lighting, and bold compositional tension",
    "Sad": "desaturated cool tones, soft shadows, and melancholic spacing",
    "Frustrated": "harsh edges, clashing tones, and tense asymmetric composition",
    "Annoyed": "irritated color clashes, sharp lines, and restless framing",
    "Alarmed": "sharp contrasts, unsettling shadows, and abrupt visual tension",
    "Bored": "flat lighting, repetitive forms, and muted uninspired palette",
    "Tired": "dim low-contrast lighting, heavy shadows, and languid composition",
}


TEMPLATE_A = (
    "A painting of {content}, in {style} style, expressing {emotion}."
)

TEMPLATE_B = (
    "A {style} artwork of {content}, with {brushstroke}, {color}, and {composition}, "
    "expressing {emotion}."
)

TEMPLATE_C = (
    "A painting of {content}, in {style} style, with a {valence} and {arousal} "
    "emotional atmosphere, expressing {emotion}."
)

TRAIN_TEMPLATES = [TEMPLATE_A, TEMPLATE_B, TEMPLATE_C]


class PromptBuilder:
    """Build prompts from EmoArt sample fields."""

    def __init__(self, templates: list[str] | None = None) -> None:
        self.templates = templates or TRAIN_TEMPLATES

    @staticmethod
    def _valence_arousal_text(valence: str, arousal: str) -> tuple[str, str]:
        """Convert valence/arousal labels to readable phrases."""
        valence_map = {"Positive": "positive", "Negative": "negative"}
        arousal_map = {"Low": "low-arousal", "High": "high-arousal"}
        v = valence_map.get(valence, valence.lower() if valence else "neutral")
        a = arousal_map.get(arousal, arousal.lower() if arousal else "moderate")
        return v, a

    def build_train_prompt(self, sample: dict[str, Any], template_idx: int | None = None) -> str:
        """Randomly sample a template and fill fields (training)."""
        template = self.templates[template_idx] if template_idx is not None else random.choice(
            self.templates
        )
        valence, arousal = self._valence_arousal_text(
            sample.get("valence", ""), sample.get("arousal", "")
        )
        return template.format(
            content=sample.get("content", "an artistic scene"),
            style=sample.get("style", "artistic"),
            emotion=sample.get("emotion", "Calm"),
            brushstroke=sample.get("brushstroke", "expressive brushstrokes"),
            color=sample.get("color", "rich colors"),
            composition=sample.get("composition", "balanced composition"),
            valence=valence,
            arousal=arousal,
        )

    def build_content_prompt(self, content: str) -> str:
        """Semantic content-only prompt for AAS content score."""
        return f"A painting of {content}."

    def build_inference_prompt(
        self,
        content: str,
        style: str,
        emotion: str,
        custom_prompt: str | None = None,
    ) -> str:
        """
        Build extended inference prompt with style preservation and emotion modifiers.
        """
        if custom_prompt:
            return custom_prompt

        modifiers = EMOTION_VISUAL_MODIFIERS.get(
            emotion,
            "expressive brushwork and emotionally evocative color choices",
        )
        return (
            f"A painting of {content}, in {style} style, expressing {emotion} emotion. "
            f"Preserve the artistic language of {style} while using visual attributes such as "
            f"{modifiers} to convey {emotion.lower()}."
        )

    def build_style_prompt(self, style: str) -> str:
        return f"A painting in {style} style."

    def build_emotion_prompt(self, emotion: str) -> str:
        modifiers = EMOTION_VISUAL_MODIFIERS.get(emotion, emotion.lower())
        return f"An artwork expressing {emotion}, with {modifiers}."
