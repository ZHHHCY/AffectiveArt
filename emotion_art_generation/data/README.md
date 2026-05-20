# EmoArt-5k Data Layout

Place the EmoArt dataset so paths in `configs/*.yaml` resolve correctly.

## Expected structure

```
acmmmmmmm/                          # workspace root (image_root = "..")
├── Images/
│   ├── annotation_5k.json          # annotations (5,528 records)
│   ├── Abstract Art/
│   │   └── *.jpg
│   ├── Expressionism/
│   │   └── *.jpg
│   └── ... (56 style folders)
└── emotion_art_generation/         # this project
```

## Annotation format

Each entry in `annotation_5k.json`:

```json
{
  "request_id": "Shin-hanga_request-93",
  "image_path": "Images\\Shin-hanga\\0043826_....jpg",
  "description": {
    "first_section": { "description": "..." },
    "second_section": {
      "visual_attributes": {
        "brushstroke": "...",
        "color": "...",
        "composition": "...",
        "light_and_shadow": "...",
        "line_quality": "..."
      },
      "emotional_impact": "..."
    },
    "third_section": {
      "dominant_emotion": "Calm",
      "emotional_valence": "Positive",
      "emotional_arousal_level": "Low",
      "healing_effects": ["Relieve Stress"]
    }
  }
}
```

## Labels

- **Style**: parsed from `request_id` (prefix before `_request-`)
- **Emotion**: `description.third_section.dominant_emotion` (12 classes)

## Validation

From project root:

```bash
python scripts/prepare_dataset.py
```
