# Emotion-Aware Artistic Image Generation (Track 1)

Generate artwork images from multimodal prompts combining **semantic content**, **painting style**, and **target emotion**, using Stable Diffusion LoRA fine-tuning with style-emotion disentanglement on the EmoArt-5k dataset.

## Project goal

Given a prompt with:
1. Semantic content (what to paint)
2. Artistic movement / style (how it looks artistically)
3. Desired emotional state (how it should feel)

the model generates an image satisfying all three conditions.

## Architecture

- **Backbone**: Stable Diffusion v1.5 (configurable to SDXL)
- **Fine-tuning**: LoRA on U-Net cross-attention (`to_q`, `to_v`)
- **Conditioning**: Learnable style & emotion embedding tables fused with CLIP text embeddings
- **Disentanglement**: Orthogonality loss between style/emotion embeddings
- **Alignment**: CLIP-based style/emotion/attribute proxy losses (+ optional classifier CE)
- **Evaluation**: FID, LPIPS (reconstruction + diversity), AAS (Attribute Alignment Score)

## Dataset

EmoArt-5k: ~5,528 artworks, 56 styles, 12 emotion classes. See [data/README.md](data/README.md).

Annotation file: `../Images/annotation_5k.json` (relative to project root).

## Installation

```bash
cd emotion_art_generation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+, CUDA recommended for training.

## Quick start

### 1. Validate dataset

```bash
python scripts/prepare_dataset.py
```

### 2. Train classifiers (for AAS & optional alignment losses)

```bash
bash scripts/train_classifiers.sh
# Outputs: outputs/classifiers/style_best.pt, emotion_best.pt
```

### 3. Train LoRA

```bash
bash scripts/train_lora.sh
# Outputs: outputs/lora/checkpoint-final/
```

### 4. Inference

```bash
python src/inference.py \
  --checkpoint outputs/lora/checkpoint-final \
  --content "a lonely boat on a quiet river at sunset" \
  --style "Expressionism" \
  --emotion "Calm" \
  --output outputs/samples/calm_expressionism_boat.png
```

Or use the shell wrapper:

```bash
bash scripts/infer.sh
```

### 5. Evaluation

```bash
bash scripts/evaluate.sh
# Results: outputs/eval/results.json
```

## Project structure

```
emotion_art_generation/
├── configs/           # YAML configs
├── data/              # Dataset documentation
├── src/
│   ├── dataset.py
│   ├── prompt_builder.py
│   ├── train_classifiers.py
│   ├── train_lora.py
│   ├── inference.py
│   ├── models/        # classifiers, conditioning, losses
│   └── evaluation/    # FID, LPIPS, AAS
├── scripts/           # Shell wrappers
└── outputs/           # Checkpoints & samples
```

## Prompt templates

**Training** (randomly sampled per image):

- **A**: `A painting of {content}, in {style} style, expressing {emotion}.`
- **B**: `A {style} artwork of {content}, with {brushstroke}, {color}, and {composition}, expressing {emotion}.`
- **C**: `A painting of {content}, in {style} style, with a {valence} and {arousal} emotional atmosphere, expressing {emotion}.`

**Inference** (extended prompt with visual modifiers):

```
A painting of {content}, in {style} style, expressing {emotion} emotion.
Preserve the artistic language of {style} while using visual attributes such as
{modifiers} to convey {emotion}.
```

## Training loss

```
L_total = L_diffusion
        + λ_style   * L_style
        + λ_emotion * L_emotion
        + λ_orth    * L_orth
        + λ_attr    * L_attr
```

| Loss | Description |
|------|-------------|
| `L_diffusion` | MSE between predicted and target noise |
| `L_orth` | \|cos(e_style, e_emotion)\| mean — disentanglement |
| `L_style` | CLIP sim(image, style text) or classifier CE |
| `L_emotion` | CLIP sim(image, emotion text) or classifier CE |
| `L_attr` | 1 − CLIP sim(image, attribute text) |

Lambdas configured in `configs/train_lora.yaml`.

## Evaluation metrics

### FID
Fréchet Inception Distance between generated and real test images (lower is better).

### LPIPS
- **Reconstruction**: generate from val prompt → compare to original
- **Diversity**: pairwise LPIPS among multiple samples per prompt (higher → more diverse)

### AAS (Attribute Alignment Score)

```
AAS = α * ContentScore + β * StyleScore + γ * EmotionScore
```

Default: α=0.4, β=0.3, γ=0.3

- **ContentScore**: CLIP similarity(image, content prompt)
- **StyleScore**: P(target style \| image) from StyleClassifier
- **EmotionScore**: P(target emotion \| image) from EmotionClassifier

## Ablation studies

Run ablations via environment variable:

```bash
# Baseline: LoRA only, no style/emotion tokens or aux losses
ABLATION=baseline_lora_only bash scripts/train_lora.sh

# + style adapter
ABLATION=style_adapter bash scripts/train_lora.sh

# + emotion adapter
ABLATION=emotion_adapter bash scripts/train_lora.sh

# + orthogonality loss
ABLATION=with_orth bash scripts/train_lora.sh

# + attribute loss
ABLATION=with_attr bash scripts/train_lora.sh

# Full model
ABLATION=full_model bash scripts/train_lora.sh
```

Presets defined under `ablation:` in `configs/train_lora.yaml`.

| Experiment | Style tokens | Emotion tokens | L_orth | L_attr | L_style | L_emotion |
|------------|-------------|----------------|--------|--------|---------|-----------|
| baseline_lora_only | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| style_adapter | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| emotion_adapter | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| with_orth | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| with_attr | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| full_model | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Expected outputs

| Stage | Output |
|-------|--------|
| Classifiers | `outputs/classifiers/style_best.pt`, `emotion_best.pt` |
| LoRA training | `outputs/lora/checkpoint-{step}/`, `checkpoint-final/` |
| Inference | `outputs/samples/*.png` |
| Evaluation | `outputs/eval/results.json`, `outputs/eval/generated/` |

## Configuration

All paths are relative to `emotion_art_generation/` unless absolute. Key settings:

- `configs/train_lora.yaml` — LoRA training, loss weights, ablations
- `configs/train_classifiers.yaml` — classifier training
- `configs/eval.yaml` — evaluation

## Notes

- **Class imbalance**: Emotion labels are skewed (Calm ~52%). Enable `use_class_weights: true` for emotion classifier.
- **Aux loss cost**: Style/emotion/attribute losses run every `aux_loss_every` steps on predicted x0 (not full sampling).
- **GPU memory**: Default batch size 4 with grad accum 4. Reduce batch size if OOM.

## License

Research / competition use. EmoArt dataset terms apply to the underlying images and annotations.
