from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3


def resolve_model_id(
    model_family: str,
    model_size: str,
    model_id: str | None,
) -> str:
    if model_id:
        return model_id

    if model_family == "mono":
        if model_size != "large":
            raise ValueError("mono は large のみ指定できます。")
        return "depth-anything/DA3MONO-LARGE"

    if model_family == "metric":
        if model_size != "large":
            raise ValueError("metric は large のみ指定できます。")
        return "depth-anything/DA3METRIC-LARGE"

    if model_family == "any":
        size_to_id = {
            "small": "depth-anything/DA3-SMALL",
            "base": "depth-anything/DA3-BASE",
            "large": "depth-anything/DA3-LARGE",
            "giant": "depth-anything/DA3-GIANT",
        }
        return size_to_id[model_size]

    raise ValueError(f"unknown model_family: {model_family}")


def depth_to_turbo_rgb(
    depth: np.ndarray,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> np.ndarray:
    """
    深度を公式寄りの見やすいカラーマップ画像に変換。
    近いほど暖色、遠いほど寒色になるよう反転している。
    """
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth)

    if not np.any(valid):
        raise ValueError("Depth map has no finite values.")

    lo = float(np.percentile(depth[valid], percentile_low))
    hi = float(np.percentile(depth[valid], percentile_high))

    if hi <= lo:
        norm = np.zeros_like(depth, dtype=np.float32)
    else:
        clipped = np.clip(depth, lo, hi)
        norm = (clipped - lo) / (hi - lo)

    # 近いものを暖色にしたいので反転
    norm = 1.0 - norm

    gray_u8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(gray_u8, cv2.COLORMAP_TURBO)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return color_rgb


def save_side_by_side(
    rgb: np.ndarray,
    depth_vis_rgb: np.ndarray,
    out_path: Path,
) -> None:
    if rgb.shape[:2] != depth_vis_rgb.shape[:2]:
        depth_vis_rgb = cv2.resize(
            depth_vis_rgb,
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    merged = np.concatenate([rgb, depth_vis_rgb], axis=1)
    Image.fromarray(merged).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 depth visualization smoke test"
    )
    parser.add_argument("--image", type=str, required=True, help="入力画像パス")
    parser.add_argument(
        "--model-family",
        type=str,
        choices=["mono", "metric", "any"],
        default="mono",
        help="mono / metric / any",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "base", "large", "giant"],
        default="large",
        help="any 系は small/base/large/giant、mono/metric は large のみ",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face model id を直接指定したい場合に使用",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="推論時の処理解像度",
    )
    parser.add_argument(
        "--focal-px",
        type=float,
        default=None,
        help="metric モデル時のみ meter 深度を保存",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="scripts/depth_anything_3/data",
        help="出力先ディレクトリ",
    )
    parser.add_argument(
        "--save-official-depth-vis",
        action="store_true",
        help="DA3 の export_format='depth_vis' も保存する",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    model_id = resolve_model_id(args.model_family, args.model_size, args.model_id)
    is_metric = "metric" in model_id.lower()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device      = {device}")
    print(f"[INFO] model_id    = {model_id}")
    print(f"[INFO] outdir      = {outdir}")
    print(f"[INFO] process_res = {args.process_res}")

    model = DepthAnything3.from_pretrained(model_id).to(device)

    if args.save_official_depth_vis:
        prediction = model.inference(
            [str(image_path)],
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
            export_dir=str(outdir),
            export_format="depth_vis",
        )
    else:
        prediction = model.inference(
            [str(image_path)],
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
        )

    depth = prediction.depth[0].astype(np.float32)
    rgb = prediction.processed_images[0].astype(np.uint8)

    print(f"[INFO] depth shape   = {depth.shape}")
    print(f"[INFO] depth dtype   = {depth.dtype}")
    print(f"[INFO] depth min/max = {float(np.min(depth)):.6f} / {float(np.max(depth)):.6f}")

    np.save(outdir / "depth_raw.npy", depth)

    depth_vis_rgb = depth_to_turbo_rgb(depth)
    Image.fromarray(depth_vis_rgb).save(outdir / "depth_vis.png")
    save_side_by_side(rgb, depth_vis_rgb, outdir / "depth_vis_side_by_side.png")

    if is_metric:
        if args.focal_px is not None:
            depth_m = (args.focal_px * depth / 300.0).astype(np.float32)
            np.save(outdir / "depth_m.npy", depth_m)

            depth_m_vis_rgb = depth_to_turbo_rgb(depth_m)
            Image.fromarray(depth_m_vis_rgb).save(outdir / "depth_m_vis.png")
            save_side_by_side(
                rgb,
                depth_m_vis_rgb,
                outdir / "depth_m_vis_side_by_side.png",
            )
            print(
                f"[INFO] metric depth min/max = "
                f"{float(np.min(depth_m)):.6f} / {float(np.max(depth_m)):.6f} [m]"
            )
        else:
            print("[INFO] metric モデルですが --focal-px 未指定のため depth_m.npy は保存しません。")

    print(f"[INFO] saved files under: {outdir}")


if __name__ == "__main__":
    """
    コマンド例:

    1) 単画像の相対深度をきれいに可視化する
    uv run python scripts/depth_anything_3/smoke_test_da3.py \
      --image scripts/depth_anything_3/data/depth_anything_3_demo.png \
      --model-family mono \
      --model-size large \
      --outdir scripts/depth_anything_3/data

    2) 単画像の metric depth を可視化して meter 深度も保存する
    uv run python scripts/depth_anything_3/smoke_test_da3.py \
      --image scripts/depth_anything_3/data/depth_anything_3_demo.png \
      --model-family metric \
      --model-size large \
      --focal-px 1000 \
      --outdir scripts/depth_anything_3/data

    3) any-view 系のサイズを切り替える
    uv run python scripts/depth_anything_3/smoke_test_da3.py \
      --image scripts/depth_anything_3/data/depth_anything_3_demo.png \
      --model-family any \
      --model-size small \
      --outdir scripts/depth_anything_3/data

    uv run python scripts/depth_anything_3/smoke_test_da3.py \
      --image scripts/depth_anything_3/data/depth_anything_3_demo.png \
      --model-family any \
      --model-size large \
      --outdir scripts/depth_anything_3/data

    4) DA3 公式の depth_vis export も一緒に保存する
    uv run python scripts/depth_anything_3/smoke_test_da3.py \
      --image scripts/depth_anything_3/data/depth_anything_3_demo.png \
      --model-family mono \
      --model-size large \
      --outdir scripts/depth_anything_3/data \
      --save-official-depth-vis
    """
    main()