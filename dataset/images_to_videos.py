"""
anomaly_samples 以下の各カテゴリディレクトリにある画像群を MP4 動画に変換するスクリプト。
出力先: dataset/videos/<category>.mp4

使い方:
    python images_to_videos.py                          # 全フレームを変換
    python images_to_videos.py --seconds 10             # 先頭10秒分のみ変換
    python images_to_videos.py --start 5 --seconds 10  # 5秒目から10秒分変換
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOTDIR = "/home/team-005/data/hazard-detection/dataset"
ANOMALY_DIR = Path(ROOTDIR) / "anomaly_samples"
OUTPUT_DIR = Path(ROOTDIR) / "videos"
FPS = 15  # フレームレート（必要に応じて変更）


def images_to_video(
    img_dir: Path,
    output_path: Path,
    fps: int = FPS,
    start_seconds: float = 0.0,
    max_seconds: float | None = None,
) -> bool:
    """ffmpeg の concat demuxer で画像シーケンスを MP4 に変換する。"""
    images = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
    if not images:
        print(f"  [SKIP] 画像が見つかりません: {img_dir}")
        return False

    # 開始・終了フレームを計算
    start_frame = int(start_seconds * fps)
    end_frame = start_frame + int(max_seconds * fps) if max_seconds is not None else len(images)
    images = images[start_frame:end_frame]

    if not images:
        print(f"  [SKIP] 指定範囲にフレームがありません: {img_dir}")
        return False

    # concat demuxer 用のファイルリストを一時ファイルに書き出す
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_path = f.name
        duration = 1.0 / fps
        for img in images:
            f.write(f"file '{img}'\n")
            f.write(f"duration {duration}\n")

    end_sec = start_seconds + (max_seconds if max_seconds is not None else len(images) / fps)
    label = f"{start_seconds}s〜{end_sec:.1f}s"
    print(f"  変換中: {img_dir.name} ({len(images)} frames, {label}) → {output_path.name}")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(list_path).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg 失敗:\n{result.stderr[-500:]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="anomaly_samples の画像を MP4 に変換")
    parser.add_argument(
        "--start", type=float, default=0.0,
        help="開始時刻（秒）（デフォルト: 0）",
    )
    parser.add_argument(
        "--seconds", type=float, default=None,
        help="開始時刻から何秒分を動画化するか（省略時は終端まで）",
    )
    parser.add_argument(
        "--fps", type=int, default=FPS,
        help=f"フレームレート（デフォルト: {FPS}）",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    categories = sorted([d for d in ANOMALY_DIR.iterdir() if d.is_dir()])
    print(f"対象カテゴリ数: {len(categories)}")
    if args.start or args.seconds:
        end_label = f"{args.start + args.seconds:.1f}s" if args.seconds else "終端"
        print(f"範囲: {args.start}s 〜 {end_label}")

    success, failed = [], []
    for cat_dir in categories:
        out_file = OUTPUT_DIR / f"{cat_dir.name}.mp4"
        ok = images_to_video(cat_dir, out_file, fps=args.fps, start_seconds=args.start, max_seconds=args.seconds)
        (success if ok else failed).append(cat_dir.name)

    print(f"\n完了: {len(success)} 件成功 / {len(failed)} 件スキップ・失敗")
    if failed:
        print(f"  スキップ: {failed}")
    if success:
        print(f"  出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
