"""
Qwen3-TTS FastAPI サーバ。

起動時に Qwen3TTSModel をロードし、OpenAI 互換の /v1/audio/speech エンドポイントで
音声合成を提供する。vLLM の標準 TTS エンドポイントが持たない instruct / language
パラメータをサポートするため、qwen_tts パッケージを直接ラップする実装になっている。

Usage:
    uv run uvicorn src.tts.server:app --host 0.0.0.0 --port 8010

    # または
    uv run python src/tts/server.py --port 8010
"""

from __future__ import annotations

import argparse
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tts.synthesize import _load_model, synthesize_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# グローバルモデル状態
# ---------------------------------------------------------------------------

_model = None
_model_name: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動時にモデルをロードし、終了時に解放する。"""
    global _model, _model_name

    import torch

    model_name = os.environ.get(
        "TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"TTS サーバ起動: model={model_name} device={device}")

    _model = _load_model(model_name, device)
    _model_name = model_name

    if _model is None:
        logger.error("モデルのロードに失敗しました。合成リクエストは 503 を返します。")
    else:
        logger.info("TTS サーバの準備が完了しました。")

    yield

    _model = None
    logger.info("TTS サーバを停止しました。")


app = FastAPI(title="Qwen3-TTS Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# リクエスト / レスポンス スキーマ
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    input: str
    voice: str = "Vivian"
    language: str = "Japanese"
    instruct: Optional[str] = None
    response_format: str = "wav"
    sample_rate: int = 24000
    # 生成パラメータ
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None


# ---------------------------------------------------------------------------
# エンドポイント
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model": _model_name}


@app.post("/v1/audio/speech")
def create_speech(req: SpeechRequest):
    """テキストを WAV 音声に合成して返す。"""
    if _model is None:
        raise HTTPException(status_code=503, detail="TTS モデルが利用できません。")

    result = synthesize_text(
        text=req.input,
        model=_model,
        voice=req.voice,
        language=req.language,
        instruct=req.instruct,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
    )

    if result is None:
        raise HTTPException(status_code=500, detail="音声合成に失敗しました。")

    waveform, native_sr = result

    # リサンプル
    if req.sample_rate != native_sr:
        import librosa

        waveform = librosa.resample(
            waveform, orig_sr=native_sr, target_sr=req.sample_rate
        )

    # WAV バイト列に変換
    buf = io.BytesIO()
    sf.write(buf, np.asarray(waveform, dtype=np.float32), req.sample_rate, format="WAV")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-TTS FastAPI サーバ")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--model", default=None, help="TTS モデル ID（省略時は TTS_MODEL 環境変数）")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    if args.model:
        os.environ["TTS_MODEL"] = args.model
    uvicorn.run("tts.server:app", host=args.host, port=args.port, reload=False)
