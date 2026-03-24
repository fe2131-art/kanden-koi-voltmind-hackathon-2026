import asyncio
import json
import logging
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import quote as url_quote
from urllib.parse import unquote, urlparse

import websockets

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data"
# 後方互換のためエイリアスを残す
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

STATIC_PORT = 8002


class _AbsolutePathHandler(BaseHTTPRequestHandler):
    """URL パスをそのままファイルシステムの絶対パスとして解釈して配信する。

    ローカル専用（127.0.0.1 バインド）のデモ用 HTTP サーバー。
    例: GET /home/team-005/data/result_1/frames/frame_0.jpg
    """

    def do_GET(self):  # noqa: N802
        path = Path(unquote(self.path))  # %20 → space, %2B → + などデコード
        if not path.is_absolute() or not path.is_file():
            self.send_error(404, "Not Found")
            return
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # suppress access logs
        pass


def _start_static_server():
    HTTPServer(("127.0.0.1", STATIC_PORT), _AbsolutePathHandler).serve_forever()


def normalize_critical_point(cp: dict) -> dict | None:
    """Convert vision_analysis.critical_points entry to App.jsx format.

    Returns None if normalized_bbox is missing or incomplete.
    """
    nb = cp.get("normalized_bbox") or {}
    if not nb or any(k not in nb for k in ("x_min", "y_min", "x_max", "y_max")):
        region_id = cp.get("region_id", "<unknown>")
        logger.debug(
            f"normalize_critical_point: skipped '{region_id}' — normalized_bbox missing or incomplete"
        )
        return None
    return {
        "region_id": cp.get("region_id", ""),
        "description": cp.get("description", ""),
        "severity": cp.get("severity", "unknown"),
        "bbox": [nb["x_min"], nb["y_min"], nb["x_max"], nb["y_max"]],
    }


async def monitor_and_stream(websocket):
    """Monitor data/perception_results/manifest.json and stream new frames to client."""
    # --- WebSocket URL クエリパラムから data ディレクトリを取得 ---
    # parse_qs は + をスペースに変換する（unquote_plus）ためパス中の + が消える。
    # unquote（+ をそのまま保持）で手動パースする。
    parsed_path = urlparse(websocket.request.path)
    raw_data = ""
    for part in parsed_path.query.split("&"):
        if part.startswith("data="):
            raw_data = unquote(part[5:])  # + → + のまま、%XX → 文字
            break
    if raw_data:
        output_dir = Path(raw_data).expanduser().resolve()
        logger.info(f"client connected (data={output_dir})")
    else:
        output_dir = DEFAULT_OUTPUT_DIR
        logger.info("client connected")

    # 指定ディレクトリの存在チェック
    if not output_dir.exists():
        err_msg = json.dumps({"error": f"data directory not found: {output_dir}"}, ensure_ascii=False)
        await websocket.send(err_msg)
        logger.warning(f"data directory not found: {output_dir}")
        return

    results_dir = output_dir / "perception_results"
    manifest_path = results_dir / "manifest.json"
    frames_dir = results_dir / "frames"
    # カスタムディレクトリ時はメディア URL を絶対パス経由の HTTP サーバーで配信
    use_static_server = raw_data != ""
    static_base = f"http://127.0.0.1:{STATIC_PORT}"

    last_count = 0  # 前回送信した frames 数をトラッキング

    try:
        while True:
            if manifest_path.exists():
                # manifest.json からフレーム数を取得（リトライ最大3回）
                manifest = None
                for attempt in range(3):
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                        break
                    except (json.JSONDecodeError, IOError) as e:
                        if attempt < 2:
                            await asyncio.sleep(0.05)
                        else:
                            logger.error(f"Error reading manifest (3 attempts): {e}")

                if manifest is not None:
                    current_count = manifest.get("frame_count", 0)

                    # 新しいフレームのみを送信
                    if current_count > last_count and frames_dir.exists():
                        # フレームファイルを名前順でソート（000000_xxx.json 形式）
                        json_files = sorted(frames_dir.glob("*.json"))

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
                                expected_frame_name = f"frame_{video_ts:.1f}s.jpg"
                                potential_path = (
                                    output_dir / "frames" / expected_frame_name
                                )
                                if potential_path.exists():
                                    rgb_frame_name = expected_frame_name

                            # fallback: フレームディレクトリからインデックスで検索
                            if not rgb_frame_name:
                                rgb_frames_dir = output_dir / "frames"
                                if rgb_frames_dir.exists():
                                    rgb_files = sorted(
                                        rgb_frames_dir.glob("frame_*.jpg")
                                    )
                                    if frame_idx < len(rgb_files):
                                        rgb_frame_name = rgb_files[frame_idx].name

                            # 深度画像・赤外線画像・音声ファイルを検出
                            # カスタムディレクトリ時は絶対パスを HTTP サーバー経由で配信
                            infrared_image_path = None
                            if rgb_frame_name:
                                potential_depth = output_dir / "depth" / rgb_frame_name
                                if potential_depth.exists():
                                    if use_static_server:
                                        depth_image_path = f"{static_base}{url_quote(str(potential_depth))}"
                                    else:
                                        depth_image_path = f"/depth/{rgb_frame_name}"

                                potential_infrared = output_dir / "infrared_frames" / rgb_frame_name
                                if potential_infrared.exists():
                                    if use_static_server:
                                        infrared_image_path = f"{static_base}{url_quote(str(potential_infrared))}"
                                    else:
                                        infrared_image_path = f"/infrared_frames/{rgb_frame_name}"

                                stem = Path(rgb_frame_name).stem
                                for ext in (".wav", ".mp3"):
                                    candidate = output_dir / "voice" / (stem + ext)
                                    if candidate.exists():
                                        if use_static_server:
                                            voice_path = f"{static_base}{url_quote(str(candidate))}"
                                        else:
                                            voice_path = f"/voice/{stem}{ext}"
                                        break

                            frame_id = result.get("frame_id", f"frame_{frame_idx}")
                            msg = {
                                "t": video_ts
                                if video_ts is not None
                                else frame_timestamp,
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
                                "audio_cues": result.get("audio", []),
                                "depth_image_path": depth_image_path,
                                "infrared_image_path": infrared_image_path,
                                "voice_path": voice_path,
                                "belief_state": result.get("belief_state"),
                                "blind_spots": vision_analysis.get("blind_spots", []),
                                "infrared_analysis": result.get("infrared_analysis"),
                                "depth_analysis": result.get("depth_analysis"),
                                "temporal_analysis": result.get("temporal_analysis"),
                                "errors": result.get("errors", []),
                                "processing_time_sec": result.get("processing_time_sec"),
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
    threading.Thread(target=_start_static_server, daemon=True).start()
    logger.info(f"static server: http://127.0.0.1:{STATIC_PORT} (absolute path serving)")
    async with websockets.serve(monitor_and_stream, "127.0.0.1", 8001):
        logger.info("ws server: ws://localhost:8001")
        logger.info(f"default data dir: {DEFAULT_OUTPUT_DIR}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(main())
