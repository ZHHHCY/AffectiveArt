"""Full evaluation pipeline: FID, LPIPS, AAS."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.dataset import EmoArtDataset, collate_fn
from src.evaluation.aas_score import compute_aas
from src.evaluation.fid_score import compute_fid
from src.evaluation.lpips_score import compute_diversity_lpips, compute_reconstruction_lpips
from src.inference import load_pipeline_from_checkpoint, generate_image
from src.models.classifiers import load_classifiers_from_checkpoint
from src.prompt_builder import PromptBuilder
from src.utils import ensure_dir, get_project_root, load_config, set_seed, setup_logging


def export_real_images(dataset: EmoArtDataset, out_dir: Path, max_images: int) -> None:
    """Copy real images to a flat directory for FID."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(dataset), max_images)
    for i in range(n):
        record = dataset.records[i]
        from src.utils import resolve_image_path

        src = resolve_image_path(record, dataset.image_root)
        dst = out_dir / f"{i:06d}.jpg"
        if not dst.exists():
            shutil.copy(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated artworks")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(get_project_root() / args.config)
    if args.checkpoint:
        cfg["model"]["checkpoint_path"] = args.checkpoint

    logger = setup_logging()
    set_seed(cfg["generation"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = get_project_root()

    ann_path = root / cfg["data"]["annotation_path"]
    image_root = root / cfg["data"]["image_root"]
    ckpt_path = root / cfg["model"]["checkpoint_path"]
    gen_dir = ensure_dir(root / cfg["generation"]["output_dir"])

    test_ds = EmoArtDataset(
        annotation_path=str(ann_path),
        image_root=str(image_root),
        split="test",
        resolution=cfg["data"]["resolution"],
        val_frac=cfg["data"]["val_frac"],
        test_frac=cfg["data"]["test_frac"],
        split_seed=cfg["data"]["split_seed"],
        return_pil=True,
        build_prompt=True,
    )

    num_samples = min(cfg["generation"]["num_samples"], len(test_ds))
    logger.info(f"Evaluating on {num_samples} test samples")

    pipeline, conditioner, meta = load_pipeline_from_checkpoint(
        str(ckpt_path),
        base_model=cfg["model"]["base"],
        device=device,
    )

    pb = PromptBuilder()
    results: dict = {}

    # Generate images
    generated_paths = []
    samples_meta = []
    for i in tqdm(range(num_samples), desc="Generating"):
        batch_item = test_ds[i]
        prompt = batch_item["prompt"]
        out_path = gen_dir / f"gen_{i:06d}.png"
        img = generate_image(
            pipeline,
            prompt,
            style_id=int(batch_item["style_id"]),
            emotion_id=int(batch_item["emotion_id"]),
            conditioner=conditioner,
            meta=meta,
            num_inference_steps=cfg["generation"]["num_inference_steps"],
            guidance_scale=cfg["generation"]["guidance_scale"],
            seed=cfg["generation"]["seed"] + i,
        )
        img.save(out_path)
        generated_paths.append(out_path)
        samples_meta.append(batch_item)

    # FID
    real_dir = cfg["fid"].get("real_images_dir")
    if real_dir is None:
        real_dir = root / "outputs/eval/real_images"
        export_real_images(test_ds, Path(real_dir), num_samples)
    else:
        real_dir = root / real_dir

    logger.info("Computing FID...")
    fid_score = compute_fid(
        real_dir,
        gen_dir,
        batch_size=cfg["fid"].get("batch_size", 50),
        device=device if torch.cuda.is_available() else "cpu",
    )
    results["fid"] = fid_score
    logger.info(f"FID: {fid_score:.4f}")

    # LPIPS reconstruction
    logger.info("Computing reconstruction LPIPS...")

    def gen_fn(p: str):
        return generate_image(
            pipeline,
            p,
            num_inference_steps=cfg["generation"]["num_inference_steps"],
            guidance_scale=cfg["generation"]["guidance_scale"],
            seed=cfg["generation"]["seed"],
        )

    recon_samples = []
    for i in range(min(cfg["lpips"]["max_reconstruction_samples"], num_samples)):
        item = test_ds[i]
        item["image_pil"] = item.get("image_pil") or test_ds._load_image(test_ds.records[i])
        recon_samples.append(item)

    lpips_recon = compute_reconstruction_lpips(
        recon_samples,
        gen_fn,
        max_samples=cfg["lpips"]["max_reconstruction_samples"],
        device=device,
        net=cfg["lpips"].get("net", "alex"),
    )
    results["lpips_reconstruction"] = lpips_recon

    prompts = [test_ds[i]["prompt"] for i in range(min(50, num_samples))]
    lpips_div = compute_diversity_lpips(
        prompts,
        gen_fn,
        n_per_prompt=cfg["lpips"].get("diversity_n", 4),
        device=device,
        net=cfg["lpips"].get("net", "alex"),
    )
    results["lpips_diversity"] = lpips_div

    # AAS
    logger.info("Computing AAS...")
    classifier_dir = root / cfg["model"]["classifier_dir"]
    style_clf, emotion_clf, style2id, emotion2id = load_classifiers_from_checkpoint(
        str(classifier_dir),
        cfg["model"]["clip_model"],
        num_style_classes=len(test_ds.style2id),
        num_emotion_classes=len(test_ds.emotion2id),
        device=device,
    )
    clip_model = CLIPModel.from_pretrained(cfg["model"]["clip_model"]).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(cfg["model"]["clip_model"])

    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in generated_paths[:num_samples]]
    content_texts = [pb.build_content_prompt(test_ds.samples[i]["content"]) for i in range(len(images))]
    style_ids = [int(test_ds.samples[i]["style_id"]) for i in range(len(images))]
    emotion_ids = [int(test_ds.samples[i]["emotion_id"]) for i in range(len(images))]

    aas = compute_aas(
        images,
        content_texts,
        style_ids,
        emotion_ids,
        clip_model,
        clip_processor,
        style_clf,
        emotion_clf,
        alpha=cfg["aas"]["alpha"],
        beta=cfg["aas"]["beta"],
        gamma=cfg["aas"]["gamma"],
        device=device,
    )
    results["aas"] = aas
    logger.info(f"AAS mean: {aas['aas_mean']:.4f}")

    out_path = ensure_dir(root / Path(cfg["output"]["results_path"]).parent) / Path(
        cfg["output"]["results_path"]
    ).name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
