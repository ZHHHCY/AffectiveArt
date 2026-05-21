"""Attribute Alignment Score (AAS) evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.models.classifiers import EmotionClassifier, StyleClassifier
from src.models.losses import _clip_feature_tensor


@torch.no_grad()
def _clip_content_score(
    images: list[Image.Image],
    content_texts: list[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
) -> list[float]:
    """CLIP cosine similarity between image and semantic content prompt."""
    image_inputs = clip_processor(images=images, return_tensors="pt")
    text_inputs = clip_processor(text=content_texts, return_tensors="pt", padding=True, truncation=True)
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    img_feat = _clip_feature_tensor(clip_model.get_image_features(**image_inputs))
    txt_feat = _clip_feature_tensor(clip_model.get_text_features(**text_inputs))
    img_feat = F.normalize(img_feat, dim=-1)
    txt_feat = F.normalize(txt_feat, dim=-1)
    sims = (img_feat * txt_feat).sum(dim=-1)
    return sims.cpu().tolist()


@torch.no_grad()
def _classifier_target_prob(
    classifier: torch.nn.Module,
    clip_processor: CLIPProcessor,
    images: list[Image.Image],
    target_ids: list[int],
    device: torch.device,
) -> list[float]:
    """Probability assigned to target class by classifier."""
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    tensors = torch.stack([transform(img) for img in images]).to(device)
    probs = classifier.predict_proba(tensors)
    scores = [probs[i, target_ids[i]].item() for i in range(len(images))]
    return scores


def compute_aas(
    images: list[Image.Image],
    content_texts: list[str],
    style_ids: list[int],
    emotion_ids: list[int],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    style_classifier: StyleClassifier,
    emotion_classifier: EmotionClassifier,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    device: str | torch.device = "cuda",
) -> dict[str, float | list[float]]:
    """
    AAS = alpha * ContentScore + beta * StyleScore + gamma * EmotionScore

    ContentScore: CLIP sim(image, content prompt)
    StyleScore: P(target style | image)
    EmotionScore: P(target emotion | image)
    """
    device = torch.device(device)
    content_scores = _clip_content_score(images, content_texts, clip_model, clip_processor, device)
    style_scores = _classifier_target_prob(
        style_classifier, clip_processor, images, style_ids, device
    )
    emotion_scores = _classifier_target_prob(
        emotion_classifier, clip_processor, images, emotion_ids, device
    )

    per_sample = [
        alpha * c + beta * s + gamma * e
        for c, s, e in zip(content_scores, style_scores, emotion_scores)
    ]

    return {
        "aas_mean": float(sum(per_sample) / max(len(per_sample), 1)),
        "content_mean": float(sum(content_scores) / max(len(content_scores), 1)),
        "style_mean": float(sum(style_scores) / max(len(style_scores), 1)),
        "emotion_mean": float(sum(emotion_scores) / max(len(emotion_scores), 1)),
        "per_sample_aas": per_sample,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }
