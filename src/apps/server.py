import asyncio
import json
import time
from pathlib import Path

import websockets

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data'
PERCEPTION_RESULTS = OUTPUT_DIR / 'perception_results.json'

def normalize_detection(obj: dict) -> dict:
    """Convert safety_agent output to App.jsx detections format."""
    bbox = obj.get("bbox", {})
    return {
        "label": obj.get("label", "unknown"),
        "score": obj.get("confidence", 0.0),
        "bbox": [
            bbox.get("x1", 0),
            bbox.get("y1", 0),
            bbox.get("x2", 1),
            bbox.get("y2", 1),
        ],
    }

async def monitor_and_stream(websocket):
    """Monitor data/perception_results.json and stream changes to client."""
    print("client connected")
    last_count = 0  # 前回送信した frames 数をトラッキング

    try:
        while True:
            # Check if perception_results.json has changed
            if PERCEPTION_RESULTS.exists():
                try:
                    with open(PERCEPTION_RESULTS, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract frames
                    frames = data.get('frames', [])
                    current_count = len(frames)

                    # 新しいフレームのみを送信（増分ストリーミング）
                    if current_count > last_count:
                        for result in frames[last_count:]:
                            objects = result.get('objects', [])
                            normalized_dets = [normalize_detection(obj) for obj in objects]

                            # タイムスタンプを使用（Unix timestamp）
                            frame_timestamp = result.get('timestamp', time.time())
                            video_ts = result.get('video_timestamp')

                            msg = {
                                't': video_ts if video_ts is not None else frame_timestamp,
                                'video_timestamp': video_ts,  # 動画内秒数（あれば）
                                'text': result.get('vision_summary', 'Analysis complete'),
                                'detections': normalized_dets,
                                'frame_id': result.get('frame_id', 'unknown')
                            }

                            await websocket.send(json.dumps(msg, ensure_ascii=False))
                            print(f"  → Frame {result.get('frame_id')} sent (t={frame_timestamp:.2f})")

                        last_count = current_count

                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading perception_results.json: {e}")

            # Poll every 0.5 seconds
            await asyncio.sleep(0.5)

    except websockets.ConnectionClosed:
        print("client disconnected")


async def main():
    async with websockets.serve(monitor_and_stream, "0.0.0.0", 8001):
        print("ws server: ws://localhost:8001")
        print(f"monitoring: {PERCEPTION_RESULTS}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
