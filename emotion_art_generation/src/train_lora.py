"""LoRA fine-tuning for emotion-aware artistic image generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer

from src.dataset import EmoArtDataset, collate_fn
from src.models.conditioning import StyleEmotionConditioner
from src.models.losses import (
    attribute_alignment_loss,
    orthogonality_loss,
    predicted_x0_from_noise,
    style_emotion_clip_proxy_loss,
)
from src.prompt_builder import PromptBuilder
from src.utils import ensure_dir, get_project_root, load_config, setup_logging


def apply_ablation(cfg: dict, ablation_name: str | None) -> dict:
    """Apply ablation preset overrides to config."""
    if not ablation_name:
        return cfg
    presets = cfg.get("ablation", {})
    if ablation_name not in presets:
        raise ValueError(f"Unknown ablation: {ablation_name}. Choose from {list(presets)}")
    preset = presets[ablation_name]
    for key, val in preset.items():
        if key in cfg.get("model", {}):
            cfg["model"][key] = val
        elif key in cfg.get("loss", {}):
            cfg["loss"][key] = val
    return cfg


def encode_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompts: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Encode prompts to text embeddings [B, seq, dim]."""
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        text_embeds = text_encoder(input_ids)[0]
    return text_embeds


def save_checkpoint(
    output_dir: Path,
    unet,
    conditioner: StyleEmotionConditioner | None,
    global_step: int,
    cfg: dict,
    style2id: dict,
    emotion2id: dict,
    is_final: bool = False,
) -> None:
    """Save LoRA weights and conditioner state."""
    name = "checkpoint-final" if is_final else f"checkpoint-{global_step}"
    ckpt_dir = output_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(ckpt_dir / "unet")
    meta = {
        "global_step": global_step,
        "style2id": style2id,
        "emotion2id": emotion2id,
        "config": cfg,
    }
    if conditioner is not None:
        torch.save(conditioner.state_dict(), ckpt_dir / "conditioner.pt")
    with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA with style-emotion disentanglement")
    parser.add_argument("--config", type=str, default="configs/train_lora.yaml")
    parser.add_argument("--ablation", type=str, default=None, help="Ablation preset name")
    args = parser.parse_args()

    cfg = load_config(get_project_root() / args.config)
    cfg = apply_ablation(cfg, args.ablation)
    logger = setup_logging()

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        mixed_precision=cfg["training"].get("mixed_precision", "fp16"),
        log_with=None,
    )
    accelerate_set_seed(cfg["training"]["seed"])
    device = accelerator.device

    root = get_project_root()
    ann_path = root / cfg["data"]["annotation_path"]
    image_root = root / cfg["data"]["image_root"]
    output_dir = ensure_dir(root / cfg["training"]["output_dir"])

    # Dataset
    train_ds = EmoArtDataset(
        annotation_path=str(ann_path),
        image_root=str(image_root),
        split="train",
        resolution=cfg["data"]["resolution"],
        center_crop=cfg["data"].get("center_crop", True),
        val_frac=cfg["data"]["val_frac"],
        test_frac=cfg["data"]["test_frac"],
        split_seed=cfg["data"]["split_seed"],
        build_prompt=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model_id = cfg["model"]["sdxl_base"] if cfg["model"].get("use_sdxl") else cfg["model"]["base"]
    logger.info(f"Loading base model: {model_id}")

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    # text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    # vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    # LoRA
    lora_config = LoraConfig(
        r=cfg["model"]["lora_rank"],
        lora_alpha=cfg["model"]["lora_alpha"],
        target_modules=cfg["model"]["lora_target_modules"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Style-emotion conditioner
    conditioner = None
    if cfg["model"].get("use_style_emotion_tokens", True):
        embed_dim = text_encoder.config.hidden_size
        conditioner = StyleEmotionConditioner(
            num_styles=len(train_ds.style2id),
            num_emotions=len(train_ds.emotion2id),
            embed_dim=embed_dim,
            mode=cfg["model"].get("conditioning_mode", "additive"),
            style_scale=cfg["model"].get("style_scale", 1.0),
            emotion_scale=cfg["model"].get("emotion_scale", 1.0),
        )

    # Optional CLIP for proxy losses
    clip_model = None
    clip_processor = None
    loss_cfg = cfg["loss"]
    need_clip = (
        loss_cfg.get("use_clip_alignment", True)
        and (
            loss_cfg.get("enable_style_loss")
            or loss_cfg.get("enable_emotion_loss")
            or loss_cfg.get("enable_attr_loss")
        )
    )
    if need_clip:
        clip_name = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(clip_name)
        clip_processor = CLIPProcessor.from_pretrained(clip_name)
        clip_model.requires_grad_(False)
        clip_model.eval()

    prompt_builder = PromptBuilder()

    # Optimizer params
    params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if conditioner is not None:
        params += list(conditioner.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["training"]["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    num_update_steps_per_epoch = len(train_loader) // cfg["training"]["gradient_accumulation_steps"]
    max_train_steps = cfg["training"]["num_epochs"] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        cfg["training"].get("lr_scheduler", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=cfg["training"].get("warmup_steps", 500),
        num_training_steps=max_train_steps,
    )

    if conditioner is not None:
        unet, conditioner, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            unet, conditioner, optimizer, train_loader, lr_scheduler
        )
    else:
        unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_loader, lr_scheduler
        )

    if clip_model is not None:
        clip_model = clip_model.to(device)

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    global_step = 0

    for epoch in range(cfg["training"]["num_epochs"]):
        unet.train()
        if conditioner is not None:
            conditioner.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process)

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                style_ids = batch["style_id"]
                emotion_ids = batch["emotion_id"]
                prompts = batch["prompt"]

                # VAE encode
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text conditioning
                text_embeds = encode_prompt(tokenizer, text_encoder, prompts, latents.device)
                e_style, e_emotion = None, None
                if conditioner is not None:
                    text_embeds, e_style, e_emotion = conditioner(
                        text_embeds, style_ids, emotion_ids
                    )

                model_pred = unet(noisy_latents, timesteps, text_embeds).sample
                loss_diffusion = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                total_loss = loss_diffusion
                loss_log = {"diffusion": loss_diffusion.item()}

                # Orthogonality loss
                if (
                    conditioner is not None
                    and loss_cfg.get("enable_orth_loss", True)
                    and e_style is not None
                ):
                    l_orth = orthogonality_loss(e_style, e_emotion)
                    total_loss = total_loss + loss_cfg["lambda_orth"] * l_orth
                    loss_log["orth"] = l_orth.item()

                # Auxiliary losses on predicted x0 (periodic)
                aux_every = loss_cfg.get("aux_loss_every", 4)
                compute_aux = global_step % aux_every == 0
                if compute_aux and clip_model is not None:
                    with torch.no_grad():
                        pred_latents = predicted_x0_from_noise(
                            model_pred, noisy_latents, timesteps, alphas_cumprod
                        )
                        pred_latents = pred_latents / vae.config.scaling_factor
                        pred_images = vae.decode(pred_latents).sample

                    style_texts = [
                        prompt_builder.build_style_prompt(s) for s in batch["style"]
                    ]
                    emotion_texts = [
                        prompt_builder.build_emotion_prompt(e) for e in batch["emotion"]
                    ]

                    if loss_cfg.get("enable_style_loss") or loss_cfg.get("enable_emotion_loss"):
                        l_style_proxy, l_emotion_proxy = style_emotion_clip_proxy_loss(
                            pred_images,
                            style_texts,
                            emotion_texts,
                            clip_model,
                            clip_processor,
                            device,
                        )
                        if loss_cfg.get("enable_style_loss"):
                            total_loss = total_loss + loss_cfg["lambda_style"] * l_style_proxy
                            loss_log["style"] = l_style_proxy.item()
                        if loss_cfg.get("enable_emotion_loss"):
                            total_loss = total_loss + loss_cfg["lambda_emotion"] * l_emotion_proxy
                            loss_log["emotion"] = l_emotion_proxy.item()

                    if loss_cfg.get("enable_attr_loss"):
                        l_attr = attribute_alignment_loss(
                            pred_images,
                            batch["attribute_text"],
                            clip_model,
                            clip_processor,
                            device,
                        )
                        total_loss = total_loss + loss_cfg["lambda_attr"] * l_attr
                        loss_log["attr"] = l_attr.item()

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, cfg["training"].get("max_grad_norm", 1.0))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                epoch_loss += total_loss.item()
                if global_step % cfg["training"].get("logging_steps", 50) == 0:
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_log.items()})

                if global_step % cfg["training"].get("save_steps", 1000) == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(unet)
                        cond_unwrapped = (
                            accelerator.unwrap_model(conditioner) if conditioner else None
                        )
                        save_checkpoint(
                            output_dir,
                            unwrapped,
                            cond_unwrapped,
                            global_step,
                            cfg,
                            train_ds.style2id,
                            train_ds.emotion2id,
                        )
                        logger.info(f"Saved checkpoint at step {global_step}")

        logger.info(f"Epoch {epoch+1} avg loss: {epoch_loss / max(len(train_loader), 1):.4f}")

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet)
        cond_unwrapped = accelerator.unwrap_model(conditioner) if conditioner else None
        save_checkpoint(
            output_dir,
            unwrapped,
            cond_unwrapped,
            global_step,
            cfg,
            train_ds.style2id,
            train_ds.emotion2id,
            is_final=True,
        )
        logger.info(f"Training complete. Final checkpoint -> {output_dir / 'checkpoint-final'}")


if __name__ == "__main__":
    main()
