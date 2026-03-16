"""
Qwen3-TTS バッチ音声合成スクリプト。

data/perception_results.json を読み込み、フレームごとに WAV を生成して
data/voice/frame_{timestamp}s.wav に保存する。

Usage:
    # GPU 環境でのフル実行
    uv run python src/tts/synthesize.py

    # ドライラン（テキスト確認のみ、GPU不要）
    uv run python src/tts/synthesize.py --dry-run

    # オプション指定
    uv run python src/tts/synthesize.py \\
        --input  data/perception_results.json \\
        --outdir data/voice \\
        --config configs/default.yaml

将来の拡張:
    frame_to_tts_text() を差し替えることで、行動計画部から渡される
    テキストに切り替えられる設計になっている。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# テキスト生成（ダミー実装 / 将来: 行動計画部から受け取る）
# ---------------------------------------------------------------------------


def frame_to_tts_text(frame: dict) -> str:
    """フレーム辞書から読み上げテキストを生成する。

    現在はダミー実装として assessment の内容をそのまま使用。
    将来的には行動計画部から渡されるテキストに差し替える。

    Args:
        frame: perception_results.json の frames[] の1要素。
               {"video_timestamp": float, "assessment": {"risk_level": str, "reason": str}, ...}

    Returns:
        読み上げ用テキスト文字列。
    """
    ts: float = frame.get("video_timestamp") or 0.0
    assessment: dict = frame.get("assessment") or {}
    risk_level: str = assessment.get("risk_level", "不明")
    reason: str = assessment.get("reason", "")

    text = f"フレーム{ts:.1f}秒。安全レベル: {risk_level}。{reason}"
    return text.strip()


# ---------------------------------------------------------------------------
# WAV ユーティリティ
# ---------------------------------------------------------------------------


def _write_silent_wav(path: Path, duration_s: float = 1.0, sample_rate: int = 12000) -> None:
    """フォールバック用: 無音の WAV ファイルを書き出す。

    モデルロード失敗や合成エラー時に呼ばれ、パイプラインが
    常にファイルを生成できるよう保証する。
    """
    samples = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
    sf.write(str(path), samples, sample_rate)
    logger.warning(f"無音 WAV を書き出しました (fallback): {path}")


def _load_results(input_path: Path) -> list[dict]:
    """perception_results.json からフレームリストを読み込む。"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames: list[dict] = data.get("frames", [])
    if not frames:
        logger.warning(f"フレームが見つかりません: {input_path}")
    return frames


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------


def _load_model(model_name: str, device: str):
    """Qwen3-TTS モデルをロードして返す。

    qwen_tts パッケージの Qwen3TTSModel を使用する。
    TRANSFORMERS_CACHE と HF_HOME のパス差異（hub/ サブディレクトリ問題）を
    避けるため、snapshot_download でローカルパスを解決してから渡す。

    Returns:
        model。失敗時は None。
    """
    import os

    import torch
    from huggingface_hub import constants, snapshot_download
    from qwen_tts import Qwen3TTSModel

    hf_home = os.environ.get("HF_HOME", constants.HF_HOME)
    cache_dir = os.path.join(hf_home, "hub")

    logger.info(f"TTS モデルをロード中: {model_name}")
    try:
        local_path = snapshot_download(repo_id=model_name, cache_dir=cache_dir)
        logger.info(f"ローカルパス: {local_path}")
        model = Qwen3TTSModel.from_pretrained(
            local_path,
            device_map="auto" if device == "cuda" else "cpu",
            dtype=torch.bfloat16,
        )
        logger.info("TTS モデルのロードが完了しました。")
        return model
    except Exception as e:
        logger.error(f"TTS モデル '{model_name}' のロードに失敗しました: {e}")
        return None


# ---------------------------------------------------------------------------
# 合成
# ---------------------------------------------------------------------------


def synthesize_text(
    text: str,
    model,
    voice: str = "Vivian",
    language: str = "Japanese",
    instruct: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
) -> Optional[tuple[np.ndarray, int]]:
    """テキストを float32 numpy 波形配列に合成する。

    Qwen3-TTS-CustomVoice の推論フロー:
      model.generate_custom_voice(text, language, speaker, instruct) を呼び出し、
      (wavs, sr) タプルの wavs[0] を返す。

    Returns:
        (waveform, native_sample_rate) タプル（成功時）、または None（失敗時）。
        None を受け取った呼び出し元は _write_silent_wav() を使うこと。
    """
    try:
        gen_kwargs: dict = {}
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        wavs, native_sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=voice,
            instruct=instruct or "",
            **gen_kwargs,
        )
        waveform: np.ndarray = np.asarray(wavs[0], dtype=np.float32)
        return waveform, int(native_sr)

    except Exception as e:
        logger.error(f"合成失敗 (text='{text[:60]}...'): {e}")
        return None


# ---------------------------------------------------------------------------
# バッチ処理メインループ
# ---------------------------------------------------------------------------


def _synthesize_via_server(
    text: str,
    server_url: str,
    voice: str,
    language: str,
    instruct: Optional[str],
    sample_rate: int,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: Optional[float],
    task_type: Optional[str] = None,
) -> Optional[tuple[np.ndarray, int]]:
    """TTS サーバ経由でテキストを合成する（vllm-omni 互換）。

    Args:
        server_url: TTS サーバのベース URL（例: http://localhost:8010）。
        task_type: vllm-omni のタスクタイプ（CustomVoice / VoiceDesign / Base）。

    Returns:
        (waveform, sample_rate) タプル（成功時）、または None（失敗時）。
    """
    import httpx

    url = f"{server_url.rstrip('/')}/v1/audio/speech"
    payload: dict = {
        "input": text,
        "voice": voice,
        "language": language,
    }
    if task_type is not None:
        payload["task_type"] = task_type
    # vllm-omni は "instructions" フィールドを使用する（FastAPI 版の "instruct" から変更）
    if instruct is not None:
        payload["instructions"] = instruct
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty

    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()

        import io

        import soundfile as sf

        buf = io.BytesIO(r.content)
        waveform, sr = sf.read(buf, dtype="float32")
        return np.asarray(waveform, dtype=np.float32), int(sr)
    except Exception as e:
        logger.error(f"サーバ合成失敗 ({url}): {e}")
        return None


def synthesize_frame(
    frame: dict,
    outdir: Path,
    model=None,
    server_url: Optional[str] = None,
    voice: str = "Vivian",
    language: str = "Japanese",
    instruct: Optional[str] = None,
    sample_rate: int = 12000,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    task_type: Optional[str] = None,
) -> Optional[Path]:
    """フレーム1件分のTTSを合成してWAVを保存する。

    - server_url 指定時 → _synthesize_via_server() 経由
    - model 指定時    → synthesize_text() 経由（ローカル）
    - どちらも None   → 無音WAVをフォールバック出力
    """
    ts = float(frame.get("video_timestamp") or 0.0)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"frame_{ts:.1f}s.wav"
    text = frame_to_tts_text(frame)
    logger.info(f"TTS合成中 {out_path.name}: {text!r}")

    if server_url is not None:
        result = _synthesize_via_server(
            text, server_url, voice, language, instruct,
            sample_rate, temperature, top_p, top_k, repetition_penalty,
            task_type=task_type,
        )
    elif model is not None:
        result = synthesize_text(
            text, model, voice=voice, language=language, instruct=instruct,
            temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        if result is not None:
            waveform, native_sr = result
            if sample_rate != native_sr:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=native_sr, target_sr=sample_rate)
            result = (waveform, sample_rate)
    else:
        logger.warning("TTS: model も server_url も未設定 → 無音WAVを出力")
        _write_silent_wav(out_path, sample_rate=sample_rate)
        return out_path

    if result is not None:
        waveform, sr = result
        sf.write(str(out_path), waveform, sr)
        duration = len(waveform) / sr
        logger.info(f"TTS保存完了: {out_path.name} ({duration:.2f}s)")
    else:
        _write_silent_wav(out_path, sample_rate=sample_rate)
    return out_path


def run_batch(
    input_path: Path,
    outdir: Path,
    model_name: str,
    sample_rate: int = 12000,
    voice: str = "Vivian",
    language: str = "Japanese",
    instruct: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    dry_run: bool = False,
    server_url: Optional[str] = None,
    task_type: Optional[str] = None,
) -> None:
    """フレームリストを読み込み、WAV ファイルを一括生成する。

    Args:
        input_path: perception_results.json のパス。
        outdir:     出力先ディレクトリ（frame_{ts:.1f}s.wav を書き出す）。
        model_name: HuggingFace モデル ID。
        sample_rate: 出力サンプルレート（Qwen3-TTS は 12000 Hz）。
        voice:      話者名。
        language:   合成言語。
        instruct:   音声スタイル指示テキスト（1.7B CustomVoice のみ有効）。
        temperature, top_p, top_k, repetition_penalty: 生成パラメータ。
        dry_run:    True の場合モデルをロードせずテキストのみ表示する。
        server_url: TTS サーバの URL（例: http://localhost:8010）。
                    指定時はサーバ経由で合成し、モデルをローカルにロードしない。
    """
    outdir.mkdir(parents=True, exist_ok=True)
    frames = _load_results(input_path)

    if not frames:
        logger.error("処理対象フレームがありません。終了します。")
        return

    if dry_run:
        logger.info("[dry-run] 合成予定のテキスト一覧:")
        for frame in frames:
            ts: float = frame.get("video_timestamp") or 0.0
            text = frame_to_tts_text(frame)
            out_path = outdir / f"frame_{ts:.1f}s.wav"
            logger.info(f"  {out_path.name}: {text!r}")
        return

    # サーバモード: ローカルモデルを使わず HTTP API を呼ぶ
    if server_url is not None:
        logger.info(f"サーバモードで合成: {server_url}")
        for frame in frames:
            ts = float(frame.get("video_timestamp") or 0.0)
            out_path = outdir / f"frame_{ts:.1f}s.wav"
            text = frame_to_tts_text(frame)
            logger.info(f"合成中 {out_path.name}: {text!r}")
            result = _synthesize_via_server(
                text=text,
                server_url=server_url,
                voice=voice,
                language=language,
                instruct=instruct,
                sample_rate=sample_rate,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                task_type=task_type,
            )
            if result is not None:
                waveform, sr = result
                sf.write(str(out_path), waveform, sr)
                duration = len(waveform) / sr
                logger.info(f"  保存完了: {out_path} ({duration:.2f}s)")
            else:
                _write_silent_wav(out_path, sample_rate=sample_rate)
        logger.info(f"完了。{len(frames)} ファイルを {outdir} に書き出しました。")
        return

    # ローカルモード: 従来通りモデルをロードして合成
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用デバイス: {device}")

    model = _load_model(model_name, device)

    for frame in frames:
        ts = float(frame.get("video_timestamp") or 0.0)
        out_path = outdir / f"frame_{ts:.1f}s.wav"
        text = frame_to_tts_text(frame)

        logger.info(f"合成中 {out_path.name}: {text!r}")

        if model is None:
            _write_silent_wav(out_path, sample_rate=sample_rate)
            continue

        result = synthesize_text(
            text, model,
            voice=voice,
            language=language,
            instruct=instruct,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        if result is not None:
            waveform, native_sr = result
            if sample_rate != native_sr:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=native_sr, target_sr=sample_rate)
                logger.info(f"  リサンプル: {native_sr} Hz → {sample_rate} Hz")
            sf.write(str(out_path), waveform, sample_rate)
            duration = len(waveform) / sample_rate
            logger.info(f"  保存完了: {out_path} ({duration:.2f}s)")
        else:
            _write_silent_wav(out_path, sample_rate=sample_rate)

    logger.info(f"完了。{len(frames)} ファイルを {outdir} に書き出しました。")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-TTS バッチ音声合成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/perception_results.json"),
        metavar="PATH",
        help="perception_results.json のパス (default: data/perception_results.json)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/voice"),
        metavar="DIR",
        help="WAV 出力先ディレクトリ (default: data/voice)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        metavar="PATH",
        help="設定ファイルのパス (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="モデルをロードせずテキストのみ表示する",
    )
    parser.add_argument(
        "--server-mode",
        action="store_true",
        help="configs/default.yaml の tts.server_url に接続してサーバ経由で合成する",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = _parse_args()

    # 設定ファイルから tts セクションを読み込む
    tts_cfg: dict = {}
    if args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        tts_cfg = cfg.get("tts", {})
    else:
        logger.warning(f"設定ファイルが見つかりません: {args.config}（デフォルト値を使用）")

    model_name: str = tts_cfg.get("model", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    sample_rate: int = int(tts_cfg.get("sample_rate", 12000))
    voice: str = tts_cfg.get("voice", "Vivian")
    language: str = tts_cfg.get("language", "Japanese")
    instruct: Optional[str] = tts_cfg.get("instructions") or tts_cfg.get("instruct") or None
    task_type: Optional[str] = tts_cfg.get("task_type") or None
    temperature: Optional[float] = tts_cfg.get("temperature")
    top_p: Optional[float] = tts_cfg.get("top_p")
    top_k: Optional[int] = tts_cfg.get("top_k")
    repetition_penalty: Optional[float] = tts_cfg.get("repetition_penalty")
    # --server-mode のときのみ設定ファイルの server_url を使用する
    server_url: Optional[str] = (tts_cfg.get("server_url") or None) if args.server_mode else None

    if server_url:
        logger.info(f"TTS server_url: {server_url}")
    if instruct:
        logger.info(f"instructions: {instruct!r}")

    run_batch(
        input_path=args.input,
        outdir=args.outdir,
        model_name=model_name,
        sample_rate=sample_rate,
        voice=voice,
        language=language,
        instruct=instruct,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        dry_run=args.dry_run,
        server_url=server_url,
        task_type=task_type,
    )


if __name__ == "__main__":
    main()
