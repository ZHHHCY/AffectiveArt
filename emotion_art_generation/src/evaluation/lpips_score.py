"""LPIPS perceptual similarity metrics."""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _get_lpips_model(net: str = "alex", device: str = "cuda"):
    import lpips

    model = lpips.LPIPS(net=net)
    model.eval().to(device)
    return model


def _pil_to_tensor(img: Image.Image, size: int = 512) -> torch.Tensor:
    t = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return t(img).unsqueeze(0)


@torch.no_grad()
def compute_lpips_pair(
    img_a: Image.Image,
    img_b: Image.Image,
    lpips_model=None,
    device: str = "cuda",
    net: str = "alex",
) -> float:
    """LPIPS between two PIL images."""
    if lpips_model is None:
        lpips_model = _get_lpips_model(net, device)
    ta = _pil_to_tensor(img_a).to(device)
    tb = _pil_to_tensor(img_b).to(device)
    return float(lpips_model(ta, tb).item())


@torch.no_grad()
def compute_reconstruction_lpips(
    samples: list[dict],
    generate_fn: Callable[[str], Image.Image],
    max_samples: int = 200,
    device: str = "cuda",
    net: str = "alex",
) -> dict[str, float]:
    """
    For each validation sample, generate from prompt and compare to original via LPIPS.

    Args:
        samples: list of dicts with 'prompt' and 'image_pil' or path
        generate_fn: callable(prompt) -> PIL Image
    """
    lpips_model = _get_lpips_model(net, device)
    scores = []
    n = min(len(samples), max_samples)

    for i in range(n):
        s = samples[i]
        prompt = s["prompt"]
        if "image_pil" in s:
            orig = s["image_pil"]
        else:
            from src.utils import resolve_image_path

            orig = Image.open(resolve_image_path(s, s.get("image_root", "."))).convert("RGB")
        gen = generate_fn(prompt)
        score = compute_lpips_pair(orig, gen, lpips_model, device)
        scores.append(score)

    return {
        "mean_lpips": float(np.mean(scores)) if scores else 0.0,
        "std_lpips": float(np.std(scores)) if scores else 0.0,
        "num_samples": len(scores),
    }


@torch.no_grad()
def compute_diversity_lpips(
    prompts: list[str],
    generate_fn: Callable[[str], Image.Image],
    n_per_prompt: int = 4,
    max_prompts: int = 50,
    device: str = "cuda",
    net: str = "alex",
) -> dict[str, float]:
    """
    Generate multiple images per prompt and compute mean pairwise LPIPS (higher = more diverse).
    """
    lpips_model = _get_lpips_model(net, device)
    all_pairwise = []

    for i, prompt in enumerate(prompts[:max_prompts]):
        images = [generate_fn(prompt) for _ in range(n_per_prompt)]
        if len(images) < 2:
            continue
        pairs = list(combinations(range(len(images)), 2))
        for a, b in pairs:
            d = compute_lpips_pair(images[a], images[b], lpips_model, device)
            all_pairwise.append(d)

    return {
        "mean_diversity_lpips": float(np.mean(all_pairwise)) if all_pairwise else 0.0,
        "std_diversity_lpips": float(np.std(all_pairwise)) if all_pairwise else 0.0,
        "num_pairs": len(all_pairwise),
    }
