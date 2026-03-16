from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_PORT = 8010
_DEFAULT_TASK_TYPE = "CustomVoice"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS vllm-omni サーバランチャー",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--host",
        default=_DEFAULT_HOST,
        help=f"バインドホスト (default: {_DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_DEFAULT_PORT,
        help=f"リッスンポート (default: {_DEFAULT_PORT})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"TTS モデル ID (default: 環境変数 TTS_MODEL → {_DEFAULT_MODEL})"
        ),
    )
    parser.add_argument(
        "--task-type",
        default=_DEFAULT_TASK_TYPE,
        choices=["CustomVoice", "VoiceDesign", "Base"],
        help=f"TTS タスクタイプ (default: {_DEFAULT_TASK_TYPE})",
    )
    return parser.parse_args()


def main() -> None:
    """
    Qwen3-TTS vllm-omni サーバランチャー。

    vllm-omni の `vllm serve <model> --omni` を起動する薄いラッパー。

    Usage:
        # 引数なしでデフォルト設定のまま起動
        uv run python src/tts/server.py

        # 引数を指定して起動
        uv run python src/tts/server.py --port 8010 --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

        # 直接 vllm-omni CLI で起動する場合
        vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --port 8010 --task-type CustomVoice
    """    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = _parse_args()

    model = args.model or os.environ.get("TTS_MODEL") or _DEFAULT_MODEL

    cmd = [
        "vllm-omni", "serve", model,
        "--omni",
        "--host", args.host,
        "--port", str(args.port),
        "--task-type", args.task_type,
    ]

    logger.info("vllm-omni TTS サーバを起動します: %s", " ".join(cmd))

    os.execvp("vllm-omni", cmd)  # noqa: S606 — 現在プロセスを vllm-omni に置き換える


if __name__ == "__main__":
    main()
