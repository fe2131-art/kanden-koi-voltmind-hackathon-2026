"""Multi-seed batch video generation.

Runs all scenes in gen_prompt.yaml with 10 different seeds each.
Total output: 3 scenes × 10 seeds = 30 videos.

Usage:
    python video_generation/generate_multi_seed.py
    python video_generation/generate_multi_seed.py --config configs/gen_prompt.yaml
    python video_generation/generate_multi_seed.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from src.util.logger import setup_logger
from video_generation.generate import build_pipeline, generate_video, load_gen_config

logger = setup_logger("generate_multi_seed", level=logging.INFO)

# 10 seeds applied to every scene
SEEDS = [0, 56, 100, 200, 314, 500, 777, 999, 1234, 2025]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all scenes × 10 seeds (30 videos total)."
    )
    parser.add_argument(
        "--config",
        default="configs/gen_prompt.yaml",
        help="Path to gen_prompt.yaml (default: configs/gen_prompt.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generation plan without running inference.",
    )
    args = parser.parse_args()

    cfg = load_gen_config(args.config)

    if cfg.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
        logger.info("CUDA_VISIBLE_DEVICES=%s", cfg.cuda_visible_devices)

    scenes = cfg.prompt_items
    total = len(scenes) * len(SEEDS)
    logger.info(
        "Plan: %d scene(s) × %d seeds = %d videos",
        len(scenes),
        len(SEEDS),
        total,
    )
    for scene in scenes:
        for seed in SEEDS:
            logger.info("  [%s] seed=%d", scene.get("id"), seed)

    if args.dry_run:
        logger.info("Dry run complete. Exiting without inference.")
        return

    pipe = build_pipeline(cfg)

    all_saved: list[Path] = []
    count = 0
    for scene in scenes:
        for seed in SEEDS:
            count += 1
            item = copy.deepcopy(scene)
            item["overrides"] = item.get("overrides") or {}
            item["overrides"]["seed"] = seed
            # Embed seed in id so filenames are unique per seed
            item["id"] = f"{scene['id']}_s{seed}"

            logger.info(
                "[%d/%d] Generating [%s] seed=%d", count, total, scene["id"], seed
            )
            saved = generate_video(pipe, cfg, item)
            all_saved.extend(saved)

    logger.info("Generation complete. %d file(s) written:", len(all_saved))
    for p in all_saved:
        logger.info("  %s", p)


if __name__ == "__main__":
    main()
