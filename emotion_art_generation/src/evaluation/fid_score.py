"""FID score computation using clean-fid."""

from __future__ import annotations

from pathlib import Path


def compute_fid(
    real_dir: str | Path,
    fake_dir: str | Path,
    batch_size: int = 50,
    device: str = "cuda",
    num_workers: int = 4,
) -> float:
    """
    Compute Fréchet Inception Distance between real and generated image folders.

    Args:
        real_dir: directory of real reference images
        fake_dir: directory of generated images
        batch_size: batch size for feature extraction
        device: cuda or cpu
        num_workers: dataloader workers

    Returns:
        FID score (lower is better)
    """
    try:
        from cleanfid import fid
    except ImportError as e:
        raise ImportError(
            "clean-fid is required for FID computation. Install with: pip install clean-fid"
        ) from e

    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)
    if not real_dir.exists():
        raise FileNotFoundError(f"Real images directory not found: {real_dir}")
    if not fake_dir.exists():
        raise FileNotFoundError(f"Generated images directory not found: {fake_dir}")

    score = fid.compute_fid(
        str(real_dir),
        str(fake_dir),
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )
    return float(score)
