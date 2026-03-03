import asyncio
import json
import time
import websockets
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'output'
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
    """Monitor output/perception_results.json and stream changes to client."""
    print("client connected")
    last_mtime = None
    start_time = time.monotonic()

    try:
        while True:
            # Check if perception_results.json has changed
            if PERCEPTION_RESULTS.exists():
                current_mtime = PERCEPTION_RESULTS.stat().st_mtime

                if last_mtime is None or current_mtime > last_mtime:
                    try:
                        with open(PERCEPTION_RESULTS, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Extract perception results
                        perception_results = data.get('perception_results', [])

                        # Stream each result with normalized detections
                        for result in perception_results:
                            objects = result.get('objects', [])
                            normalized_dets = [normalize_detection(obj) for obj in objects]

                            msg = {
                                't': round(time.monotonic() - start_time, 3),
                                'text': result.get('vision_analysis', 'Analysis complete'),
                                'detections': normalized_dets
                            }

                            await websocket.send(json.dumps(msg, ensure_ascii=False))

                        last_mtime = current_mtime

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
