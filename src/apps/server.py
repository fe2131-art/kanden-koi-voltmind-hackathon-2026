import asyncio
import json
import logging
import time
from pathlib import Path

import websockets

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data"
PERCEPTION_RESULTS = OUTPUT_DIR / "perception_results.json"


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
    """Monitor data/perception_results.json and stream changes to client."""
    logger.info("client connected")
    last_count = 0  # 前回送信した frames 数をトラッキング

    try:
        while True:
            # Check if perception_results.json has changed
            if PERCEPTION_RESULTS.exists():
                # JSON 読み込みの競合対策：リトライロジック（最大3回、50ms待機）
                data = None
                for attempt in range(3):
                    try:
                        with open(PERCEPTION_RESULTS, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        break
                    except (json.JSONDecodeError, IOError) as e:
                        if attempt < 2:
                            await asyncio.sleep(0.05)
                        elif attempt == 2:
                            logger.error(
                                f"Error reading perception_results.json (3 attempts): {e}"
                            )

                if data is not None:
                    # Extract frames
                    frames = data.get("frames", [])
                    current_count = len(frames)

                    # 新しいフレームのみを送信（増分ストリーミング）
                    if current_count > last_count:
                        for frame_idx, result in enumerate(
                            frames[last_count:], start=last_count
                        ):
                            vision_analysis = result.get("vision_analysis") or {}
                            critical_points = [
                                p
                                for p in (
                                    normalize_critical_point(cp)
                                    for cp in vision_analysis.get("critical_points", [])
                                )
                                if p is not None
                            ]

                            # タイムスタンプを使用（Unix timestamp）
                            frame_timestamp = result.get("timestamp", time.time())
                            video_ts = result.get("video_timestamp")

                            # フレーム番号から RGB フレームパスを推測（配列インデックスをフレーム番号として使用）
                            # frame_0.0s.jpg → frame_1.0s.jpg など
                            rgb_frame_name = None
                            depth_image_path = None
                            voice_path = None

                            # フレームファイルを検索: frame_{frame_idx}.{timestamp}s.jpg
                            # または video_timestamp がある場合は対応する時刻のフレーム
                            if video_ts is not None:
                                # video_timestamp から フレーム番号を推測
                                expected_frame_name = f"frame_{video_ts:.3f}s.jpg"
                                potential_path = (
                                    OUTPUT_DIR / "frames" / expected_frame_name
                                )
                                if potential_path.exists():
                                    rgb_frame_name = expected_frame_name

                            # fallback: フレームディレクトリから一致するファイルを検出
                            if not rgb_frame_name:
                                frames_dir = OUTPUT_DIR / "frames"
                                if frames_dir.exists():
                                    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
                                    if frame_idx < len(frame_files):
                                        rgb_frame_name = frame_files[frame_idx].name

                            # 深度画像と音声ファイルを検出
                            if rgb_frame_name:
                                potential_depth_path = (
                                    OUTPUT_DIR / "depth" / rgb_frame_name
                                )
                                if potential_depth_path.exists():
                                    depth_image_path = f"/depth/{rgb_frame_name}"

                                # voice_path 検出（RGB フレームのステムに対応する音声ファイル）
                                stem = Path(rgb_frame_name).stem  # "frame_0.0s" など
                                for ext in (".wav", ".mp3"):
                                    candidate = OUTPUT_DIR / "voice" / (stem + ext)
                                    if candidate.exists():
                                        voice_path = f"/voice/{stem}{ext}"
                                        break

                            frame_id = result.get("frame_id", f"frame_{frame_idx}")
                            msg = {
                                "t": video_ts
                                if video_ts is not None
                                else frame_timestamp,
                                "video_timestamp": video_ts,  # 動画内秒数（あれば）
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
        logger.info(f"monitoring: {PERCEPTION_RESULTS}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(main())
