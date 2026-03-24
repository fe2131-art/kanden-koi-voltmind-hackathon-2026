"""LTX-2 video generation script using diffusers."""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Disable XetHub download protocol (causes "Background writer channel closed" on some systems)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import yaml

from src.util.logger import setup_logger

logger = setup_logger("video_generation", level=logging.INFO)


@dataclass
class VideoGenerationConfig:
    """Typed configuration parsed from gen_prompt.yaml."""

    # model
    repo_id: str
    local_path: Optional[str]
    dtype: str
    device: str
    cuda_visible_devices: Optional[str]
    enable_model_cpu_offload: bool
    enable_vae_slicing: bool
    enable_vae_tiling: bool

    # generation defaults
    width: int
    height: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    seed: Optional[int]
    num_videos_per_prompt: int
    generate_audio: bool

    # prompts
    negative_prompt: str
    prompt_items: list[dict[str, Any]]

    # output
    output_dir: Path
    filename_template: str
    fps: int
    codec: str
    quality: int
    save_frames: bool


def load_gen_config(config_path: str = "configs/gen_prompt.yaml") -> VideoGenerationConfig:
    """Load and validate gen_prompt.yaml, returning a typed config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    m = raw.get("model", {})
    g = raw.get("generation", {})
    p = raw.get("prompts", {})
    o = raw.get("output", {})

    # Validate num_frames constraint: (num_frames - 1) % 8 == 0
    num_frames = g.get("num_frames", 121)
    if (num_frames - 1) % 8 != 0:
        raise ValueError(
            f"num_frames={num_frames} is invalid for LTX-2. "
            "Must satisfy (num_frames - 1) % 8 == 0 (e.g. 9, 17, 25, 121)."
        )

    return VideoGenerationConfig(
        repo_id=m.get("repo_id", "Lightricks/LTX-2"),
        local_path=m.get("local_path"),
        dtype=m.get("dtype", "bfloat16"),
        device=m.get("device", "cuda"),
        cuda_visible_devices=m.get("cuda_visible_devices"),
        enable_model_cpu_offload=m.get("enable_model_cpu_offload", False),
        enable_vae_slicing=m.get("enable_vae_slicing", True),
        enable_vae_tiling=m.get("enable_vae_tiling", False),
        width=g.get("width", 768),
        height=g.get("height", 512),
        num_frames=num_frames,
        num_inference_steps=g.get("num_inference_steps", 40),
        guidance_scale=g.get("guidance_scale", 3.0),
        seed=g.get("seed"),
        num_videos_per_prompt=g.get("num_videos_per_prompt", 1),
        generate_audio=g.get("generate_audio", False),
        negative_prompt=p.get("negative_prompt", ""),
        prompt_items=p.get("items", []),
        output_dir=Path(o.get("dir", "data/videos")),
        filename_template=o.get("filename_template", "{id}_{seed}_{timestamp}.mp4"),
        fps=o.get("fps", 24),
        codec=o.get("codec", "libx264"),
        quality=o.get("quality", 8),
        save_frames=o.get("save_frames", False),
    )


def build_pipeline(cfg: VideoGenerationConfig):
    """Instantiate and configure the LTX2Pipeline."""
    from diffusers import LTX2Pipeline  # lazy import

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

    model_source = cfg.local_path if cfg.local_path else cfg.repo_id
    logger.info("Loading LTX-2 pipeline from: %s", model_source)

    pipe = LTX2Pipeline.from_pretrained(model_source, torch_dtype=torch_dtype)

    if cfg.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
        logger.info("Model CPU offload enabled.")
    else:
        pipe.to(cfg.device)
        logger.info("Pipeline moved to device: %s", cfg.device)

    if cfg.enable_vae_slicing:
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        else:
            logger.warning("enable_vae_slicing is not supported by this pipeline; skipping.")
    if cfg.enable_vae_tiling:
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        else:
            logger.warning("enable_vae_tiling is not supported by this pipeline; skipping.")

    return pipe


def generate_video(
    pipe,
    cfg: VideoGenerationConfig,
    prompt_item: dict[str, Any],
) -> list[Path]:
    """
    Generate video(s) for a single prompt_item entry.
    Returns list of saved MP4 paths.
    """
    item_id = prompt_item.get("id", "unnamed")
    prompt_text = prompt_item.get("prompt", "")
    overrides = prompt_item.get("overrides", {}) or {}

    # Merge per-prompt overrides with generation defaults
    width = overrides.get("width", cfg.width)
    height = overrides.get("height", cfg.height)
    num_frames = overrides.get("num_frames", cfg.num_frames)
    num_inference_steps = overrides.get("num_inference_steps", cfg.num_inference_steps)
    guidance_scale = overrides.get("guidance_scale", cfg.guidance_scale)
    seed = overrides.get("seed", cfg.seed)

    # Validate override num_frames too
    if (num_frames - 1) % 8 != 0:
        raise ValueError(
            f"[{item_id}] Override num_frames={num_frames} invalid. "
            "Must satisfy (num_frames - 1) % 8 == 0."
        )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=cfg.device).manual_seed(seed)

    logger.info(
        "[%s] Generating: frames=%d, size=%dx%d, steps=%d, cfg=%.1f, seed=%s",
        item_id,
        num_frames,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        str(seed),
    )

    # Call pipeline
    result = pipe(
        prompt=prompt_text,
        negative_prompt=cfg.negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_videos_per_prompt=cfg.num_videos_per_prompt,
        generator=generator,
        output_type="np",
        return_dict=False,
    )

    # LTX2Pipeline returns (video, audio) when return_dict=False
    if isinstance(result, (tuple, list)) and len(result) >= 1:
        video_array = result[0]
    else:
        video_array = result

    # Ensure output directory exists
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths: list[Path] = []

    # Handle multiple video outputs
    import numpy as np

    if isinstance(video_array, np.ndarray) and video_array.ndim == 5:
        videos = [video_array[i] for i in range(video_array.shape[0])]
    elif isinstance(video_array, list):
        videos = video_array
    else:
        videos = [video_array]

    for idx, frames in enumerate(videos):
        # frames: np.ndarray (T, H, W, 3), float32 [0, 1] -> uint8 [0, 255]
        frames_uint8 = (frames * 255).clip(0, 255).astype(np.uint8)

        out_seed = seed if seed is not None else "rand"
        filename = cfg.filename_template.format(
            id=f"{item_id}_{idx}" if len(videos) > 1 else item_id,
            seed=out_seed,
            timestamp=timestamp,
        )
        out_path = cfg.output_dir / filename

        _save_mp4(frames_uint8, out_path, cfg.fps, cfg.codec, cfg.quality)
        logger.info("[%s] Saved: %s", item_id, out_path)
        saved_paths.append(out_path)

        if cfg.save_frames:
            _save_frames(frames_uint8, cfg.output_dir, item_id, idx)

    return saved_paths


def _save_mp4(
    frames,
    path: Path,
    fps: int,
    codec: str,
    quality: int,
) -> None:
    """Write numpy frames to an MP4 file using imageio-ffmpeg."""
    import imageio

    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec=codec,
        quality=quality,
        format="ffmpeg",
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def _save_frames(
    frames,
    out_dir: Path,
    item_id: str,
    video_idx: int,
) -> None:
    """Optionally save individual frames to data/frames/ as PNG files."""
    import imageio

    frames_dir = out_dir.parent / "frames" / f"{item_id}_{video_idx}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for t, frame in enumerate(frames):
        frame_path = frames_dir / f"frame_{t:04d}.png"
        imageio.imwrite(str(frame_path), frame)
    logger.info("Saved %d frames to %s", len(frames), frames_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate videos using LTX-2 via diffusers."
    )
    parser.add_argument(
        "--config",
        default="configs/gen_prompt.yaml",
        help="Path to gen_prompt.yaml (default: configs/gen_prompt.yaml)",
    )
    parser.add_argument(
        "--prompt-id",
        default=None,
        help="Run only the prompt with this id (default: run all prompts)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and print plan without running inference.",
    )
    args = parser.parse_args()

    cfg = load_gen_config(args.config)

    # Apply CUDA_VISIBLE_DEVICES before any CUDA/torch initialization
    if cfg.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
        logger.info("CUDA_VISIBLE_DEVICES=%s", cfg.cuda_visible_devices)

    items = cfg.prompt_items
    if args.prompt_id:
        items = [it for it in items if it.get("id") == args.prompt_id]
        if not items:
            logger.error("No prompt found with id=%s", args.prompt_id)
            sys.exit(1)

    logger.info("Video generation plan: %d prompt(s)", len(items))
    for it in items:
        prompt_preview = str(it.get("prompt", ""))[:60]
        logger.info("  - [%s] %s...", it.get("id"), prompt_preview)

    if args.dry_run:
        logger.info("Dry run complete. Exiting without inference.")
        return

    pipe = build_pipeline(cfg)

    all_saved: list[Path] = []
    for item in items:
        saved = generate_video(pipe, cfg, item)
        all_saved.extend(saved)

    logger.info("Generation complete. %d file(s) written:", len(all_saved))
    for p in all_saved:
        logger.info("  %s", p)


if __name__ == "__main__":
    main()
