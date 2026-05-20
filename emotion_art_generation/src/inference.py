"""Inference script for emotion-aware artistic image generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from src.models.conditioning import StyleEmotionConditioner
from src.prompt_builder import PromptBuilder
from src.utils import get_project_root, setup_logging


def load_pipeline_from_checkpoint(
    checkpoint_dir: str,
    base_model: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda",
) -> tuple[StableDiffusionPipeline, StyleEmotionConditioner | None, dict]:
    """
    Load SD pipeline with LoRA weights and optional conditioner.

    Returns:
        pipeline, conditioner, meta dict (style2id, emotion2id)
    """
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Train LoRA first with: bash scripts/train_lora.sh"
        )

    meta_path = ckpt / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    unet_path = ckpt / "unet"
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    if unet_path.exists():
        from peft import PeftModel

        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, str(unet_path))
    pipeline = pipeline.to(device)

    conditioner = None
    cond_path = ckpt / "conditioner.pt"
    if cond_path.exists() and meta.get("style2id") and meta.get("emotion2id"):
        embed_dim = pipeline.text_encoder.config.hidden_size
        conditioner = StyleEmotionConditioner(
            num_styles=len(meta["style2id"]),
            num_emotions=len(meta["emotion2id"]),
            embed_dim=embed_dim,
        )
        state = torch.load(cond_path, map_location=device, weights_only=True)
        conditioner.load_state_dict(state)
        conditioner.eval().to(device)

    return pipeline, conditioner, meta


@torch.no_grad()
def generate_image(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    style_id: int | None = None,
    emotion_id: int | None = None,
    conditioner: StyleEmotionConditioner | None = None,
    meta: dict | None = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
) -> Image.Image:
    """Generate a single image with optional style-emotion conditioning."""
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    if conditioner is not None and style_id is not None and emotion_id is not None:
        # Custom generation with conditioned embeddings
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(pipeline.device)
        text_embeds = text_encoder(input_ids)[0]
        style_t = torch.tensor([style_id], device=pipeline.device)
        emotion_t = torch.tensor([emotion_id], device=pipeline.device)
        text_embeds, _, _ = conditioner(text_embeds, style_t, emotion_t)

        image = pipeline(
            prompt_embeds=text_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
        ).images[0]
    else:
        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
        ).images[0]
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate emotion-aware artwork")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--emotion", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/samples/output.png")
    parser.add_argument("--prompt", type=str, default=None, help="Override auto-built prompt")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = get_project_root()
    ckpt = args.checkpoint if Path(args.checkpoint).is_absolute() else root / args.checkpoint
    out = args.output if Path(args.output).is_absolute() else root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    pipeline, conditioner, meta = load_pipeline_from_checkpoint(
        str(ckpt), base_model=args.base_model, device=device
    )

    pb = PromptBuilder()
    prompt = pb.build_inference_prompt(
        args.content, args.style, args.emotion, custom_prompt=args.prompt
    )
    logger.info(f"Prompt: {prompt}")

    style_id, emotion_id = None, None
    if meta.get("style2id") and meta.get("emotion2id"):
        style2id = meta["style2id"]
        emotion2id = meta["emotion2id"]
        if args.style not in style2id:
            logger.warning(
                f"Style '{args.style}' not in training labels. Known: {list(style2id.keys())[:5]}..."
            )
        if args.emotion not in emotion2id:
            logger.warning(
                f"Emotion '{args.emotion}' not in training labels. Known: {list(emotion2id.keys())}"
            )
        style_id = style2id.get(args.style)
        emotion_id = emotion2id.get(args.emotion)

    image = generate_image(
        pipeline,
        prompt,
        style_id=style_id,
        emotion_id=emotion_id,
        conditioner=conditioner,
        meta=meta,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    image.save(out)
    logger.info(f"Saved -> {out}")


if __name__ == "__main__":
    main()
