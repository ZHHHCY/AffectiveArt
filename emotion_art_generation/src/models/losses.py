"""Auxiliary losses for style-emotion disentangled diffusion training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


def _clip_feature_tensor(features: torch.Tensor) -> torch.Tensor:
    """Extract embedding tensor from CLIP feature output (transformers 4.x/5.x)."""
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features


def orthogonality_loss(
    e_style: torch.Tensor,
    e_emotion: torch.Tensor,
    mode: str = "cosine",
) -> torch.Tensor:
    """
    Encourage style and emotion embeddings to be disentangled.

    Args:
        e_style: [B, D]
        e_emotion: [B, D]
        mode: cosine | frobenius
    """
    if mode == "frobenius":
        # Batch-wise: mean over ||E_s^T E_e||_F^2 style — use stacked matrices
        style_norm = F.normalize(e_style, dim=-1)
        emotion_norm = F.normalize(e_emotion, dim=-1)
        gram = torch.matmul(style_norm.T, emotion_norm)
        return (gram ** 2).mean()
    # Default: per-sample cosine similarity magnitude
    cos = F.cosine_similarity(e_style, e_emotion, dim=-1)
    return cos.abs().mean()


def clip_alignment_loss(
    images: torch.Tensor,
    texts: list[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    """
    1 - cosine_similarity(CLIP_image, CLIP_text) averaged over batch.
    images: [B, 3, H, W] in [0, 1] or [-1, 1]
    """
    # Denormalize if needed (SD VAE output is roughly [-1, 1])
    imgs = images.detach()
    if imgs.min() < 0:
        imgs = (imgs + 1.0) / 2.0
    imgs = imgs.clamp(0, 1)
    imgs_pil = []
    from torchvision.transforms.functional import to_pil_image

    for i in range(imgs.shape[0]):
        imgs_pil.append(to_pil_image(imgs[i].cpu()))

    text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    image_inputs = clip_processor(images=imgs_pil, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

    with torch.no_grad():
        clip_model.eval()
    image_features = _clip_feature_tensor(clip_model.get_image_features(**image_inputs))
    text_features = _clip_feature_tensor(clip_model.get_text_features(**text_inputs))
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    sim = (image_features * text_features).sum(dim=-1)
    return (1.0 - sim).mean()


def attribute_alignment_loss(
    images: torch.Tensor,
    attribute_texts: list[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    """Attribute alignment: CLIP image vs combined attribute text."""
    return clip_alignment_loss(images, attribute_texts, clip_model, clip_processor, device)


def classifier_alignment_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy for style/emotion classifier alignment."""
    return F.cross_entropy(logits, targets)


def style_emotion_clip_proxy_loss(
    images: torch.Tensor,
    style_texts: list[str],
    emotion_texts: list[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast CLIP-based proxy for style and emotion alignment."""
    l_style = clip_alignment_loss(images, style_texts, clip_model, clip_processor, device)
    l_emotion = clip_alignment_loss(images, emotion_texts, clip_model, clip_processor, device)
    return l_style, l_emotion


def predicted_x0_from_noise(
    noise_pred: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Recover predicted x0 from noise prediction (DDPM parameterization)."""
    # alphas_cumprod: [num_train_timesteps]
    alpha_prod = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    beta_prod = 1 - alpha_prod
    pred_x0 = (noisy_latents - beta_prod.sqrt() * noise_pred) / alpha_prod.sqrt()
    return pred_x0
