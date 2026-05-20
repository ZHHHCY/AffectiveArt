#!/usr/bin/env python3
"""Validate EmoArt dataset and print split statistics."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import build_splits, parse_record
from src.utils import (
    extract_style_from_record,
    get_style_emotion_maps,
    load_annotations,
    resolve_image_path,
    setup_logging,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate EmoArt-5k dataset")
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="../Images/annotation_5k.json",
    )
    parser.add_argument("--image_root", type=str, default="..")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logging("prepare_dataset")
    ann_path = PROJECT_ROOT / args.annotation_path
    image_root = PROJECT_ROOT / args.image_root

    try:
        records = load_annotations(ann_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    style2id, emotion2id, _, _ = get_style_emotion_maps(ann_path)
    logger.info(f"Loaded {len(records)} annotations")
    logger.info(f"Styles: {len(style2id)}, Emotions: {len(emotion2id)}")

    missing = []
    for rec in records:
        try:
            resolve_image_path(rec, image_root)
        except FileNotFoundError:
            missing.append(rec.get("image_path", "unknown"))

    if missing:
        logger.warning(f"Missing images: {len(missing)} / {len(records)}")
        missing_file = PROJECT_ROOT / "outputs" / "missing_images.txt"
        missing_file.parent.mkdir(parents=True, exist_ok=True)
        missing_file.write_text("\n".join(missing[:500]), encoding="utf-8")
        logger.warning(f"First 500 missing paths written to {missing_file}")
    else:
        logger.info("All images found.")

    train_idx, val_idx, test_idx = build_splits(
        records, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    logger.info(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_styles = Counter(extract_style_from_record(records[i]) for i in train_idx)
    train_emotions = Counter(
        parse_record(records[i], style2id, emotion2id)["emotion"] for i in train_idx
    )
    logger.info(f"Train styles (top 5): {train_styles.most_common(5)}")
    logger.info(f"Train emotions: {dict(train_emotions)}")

    rare = [e for e, c in train_emotions.items() if c < 5]
    if rare:
        logger.warning(
            f"Rare emotion classes (<5 train samples): {rare}. "
            "Consider class weights or merging for classifier training."
        )

    logger.info("Dataset validation complete.")


if __name__ == "__main__":
    main()
