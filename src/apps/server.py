import asyncio
import json
import logging
import time
from pathlib import Path

import websockets

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = OUTPUT_DIR / "perception_results"
MANIFEST_PATH = RESULTS_DIR / "manifest.json"
FRAMES_DIR = RESULTS_DIR / "frames"


def normalize_critical_point(cp: dict) -> dict | None:
    """Convert vision_analysis.critical_points entry to App.jsx format.

    Returns None if normalized_bbox is missing or incomplete.
    """
    nb = cp.get("normalized_bbox") or {}
    if not nb or any(k not in nb for k in ("x_min", "y_min", "x_max", "y_max")):
        return None
    return {
        "description": cp.get("description", ""),
        "severity": cp.get("severity", "unknown"),
        "bbox": [nb["x_min"], nb["y_min"], nb["x_max"], nb["y_max"]],
    }


async def monitor_and_stream(websocket):
    """Monitor data/perception_results/manifest.json and stream new frames to client."""
    logger.info("client connected")
    last_count = 0  # 前回送信した frames 数をトラッキング

    try:
        while True:
            if MANIFEST_PATH.exists():
                # manifest.json からフレーム数を取得（リトライ最大3回）
                manifest = None
                for attempt in range(3):
                    try:
                        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                        break
                    except (json.JSONDecodeError, IOError) as e:
                        if attempt < 2:
                            await asyncio.sleep(0.05)
                        elif attempt == 2:
                            logger.error(
                                f"Error reading manifest (3 attempts): {e}"
                            )

                if manifest is not None:
                    current_count = manifest.get("frame_count", 0)

                    # 新しいフレームのみを送信
                    if current_count > last_count and FRAMES_DIR.exists():
                        # フレームファイルを名前順でソート（000000_xxx.json 形式）
                        json_files = sorted(FRAMES_DIR.glob("*.json"))

                        for frame_idx in range(last_count, current_count):
                            if frame_idx >= len(json_files):
                                break

                            frame_file = json_files[frame_idx]

                            # フレームファイルを読み込み（リトライ最大3回）
                            result = None
                            for attempt in range(3):
                                try:
                                    with open(frame_file, "r", encoding="utf-8") as f:
                                        result = json.load(f)
                                    break
                                except (json.JSONDecodeError, IOError) as e:
                                    if attempt < 2:
                                        await asyncio.sleep(0.05)
                                    else:
                                        logger.error(
                                            f"Error reading {frame_file.name} (3 attempts): {e}"
                                        )

                            if result is None:
                                continue

                            vision_analysis = result.get("vision_analysis") or {}
                            critical_points = [
                                p
                                for p in (
                                    normalize_critical_point(cp)
                                    for cp in vision_analysis.get("critical_points", [])
                                )
                                if p is not None
                            ]

                            frame_timestamp = result.get("timestamp", time.time())
                            video_ts = result.get("video_timestamp")

                            # フレーム番号から RGB フレームパスを推測
                            rgb_frame_name = None
                            depth_image_path = None
                            voice_path = None

                            if video_ts is not None:
                                expected_frame_name = f"frame_{video_ts:.3f}s.jpg"
                                potential_path = OUTPUT_DIR / "frames" / expected_frame_name
                                if potential_path.exists():
                                    rgb_frame_name = expected_frame_name

                            # fallback: フレームディレクトリからインデックスで検索
                            if not rgb_frame_name:
                                rgb_frames_dir = OUTPUT_DIR / "frames"
                                if rgb_frames_dir.exists():
                                    rgb_files = sorted(rgb_frames_dir.glob("frame_*.jpg"))
                                    if frame_idx < len(rgb_files):
                                        rgb_frame_name = rgb_files[frame_idx].name

                            # 深度画像と音声ファイルを検出
                            if rgb_frame_name:
                                potential_depth = OUTPUT_DIR / "depth" / rgb_frame_name
                                if potential_depth.exists():
                                    depth_image_path = f"/depth/{rgb_frame_name}"

                                stem = Path(rgb_frame_name).stem
                                for ext in (".wav", ".mp3"):
                                    candidate = OUTPUT_DIR / "voice" / (stem + ext)
                                    if candidate.exists():
                                        voice_path = f"/voice/{stem}{ext}"
                                        break

                            frame_id = result.get("frame_id", f"frame_{frame_idx}")
                            msg = {
                                "t": video_ts if video_ts is not None else frame_timestamp,
                                "video_timestamp": video_ts,
                                "text": vision_analysis.get(
                                    "scene_description", "Analysis complete"
                                ),
                                "frame_id": frame_id,
                                "assessment": result.get("assessment"),
                                "critical_points": critical_points,
                                "scene_description": vision_analysis.get(
                                    "scene_description", ""
                                ),
                                "depth_image_path": depth_image_path,
                                "voice_path": voice_path,
                            }

                            await websocket.send(json.dumps(msg, ensure_ascii=False))
                            logger.debug(
                                f"  → Frame {frame_id} sent (depth={'✓' if depth_image_path else '✗'}, voice={'✓' if voice_path else '✗'}, t={frame_timestamp:.2f})"
                            )

                        last_count = current_count

            # Poll every 0.1 seconds
            await asyncio.sleep(0.1)

    except websockets.ConnectionClosed:
        logger.info("client disconnected")


async def main():
    async with websockets.serve(monitor_and_stream, "0.0.0.0", 8001):
        logger.info("ws server: ws://localhost:8001")
        logger.info(f"monitoring: {MANIFEST_PATH}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(main())
