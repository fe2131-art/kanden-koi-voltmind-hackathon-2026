"""Kokoro TTS ナレーター: assessment.safety_status をフレーム単位で音声化"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class TTSNarrator:
    """Kokoro TTS を使ってフレームの safety_status を WAV ファイルへ変換するクラス。

    設定例 (configs/default.yaml):
        tts:
          enabled: true
          voice: "jf_alpha"   # 日本語女性ナレーター
          speed: 1.0          # 読み上げ速度（0.5=遅, 1.0=標準, 1.5=速）
          output_dir: "data/voice"
          lang_code: "j"      # j=日本語, a=英語(US), b=英語(UK) など
          device: null        # null=自動(CUDA優先), "cpu", "cuda"
    """

    SAMPLE_RATE = 24000  # Kokoro の固定出力サンプルレート

    def __init__(self, config: dict) -> None:
        tts_cfg = config.get("tts", {})
        self.enabled: bool = tts_cfg.get("enabled", False)
        self.voice: str = tts_cfg.get("voice", "jf_alpha")
        self.speed: float = float(tts_cfg.get("speed", 1.0))
        self.output_dir: Path = Path(tts_cfg.get("output_dir", "data/voice"))
        self.lang_code: str = tts_cfg.get("lang_code", "j")
        self.device: Optional[str] = tts_cfg.get("device", None)
        self._pipeline: Optional[Any] = (
            None  # 遅延初期化（最初の generate() 呼び出し時にロード）
        )

    def _ensure_pipeline(self) -> None:
        """KPipeline を遅延初期化する。"""
        if self._pipeline is not None:
            return
        try:
            from kokoro import KPipeline  # type: ignore[import-untyped]

            # repo_id を明示して不要な print() ノイズを抑制
            self._pipeline = KPipeline(
                lang_code=self.lang_code,
                repo_id="hexgrad/Kokoro-82M",
                device=self.device,
            )
            logger.info(
                f"TTSNarrator: KPipeline 初期化完了"
                f" (lang={self.lang_code}, voice={self.voice}, speed={self.speed},"
                f" device={self.device or 'auto'})"
            )
        except Exception as e:
            logger.error(f"TTSNarrator: KPipeline 初期化失敗: {e}")
            raise

    def generate(self, frame_id: str, text: str) -> Optional[Path]:
        """safety_status テキストを WAV ファイルへ変換して保存する。

        Args:
            frame_id: フレーム識別子（出力ファイル名のベース名になる）
            text: 読み上げるテキスト（assessment.safety_status の値）

        Returns:
            保存した WAV ファイルのパス。生成スキップ時は None。
        """
        if not self.enabled:
            return None
        if not text or not text.strip():
            logger.debug(
                f"TTSNarrator: フレーム {frame_id} のテキストが空のためスキップ"
            )
            return None

        try:
            self._ensure_pipeline()
        except Exception as e:
            # 初期化失敗時は TTS を無効化してエージェントを継続させる
            logger.error(f"TTSNarrator: 初期化失敗のため TTS を無効化: {e}")
            self.enabled = False
            return None
        assert self._pipeline is not None  # _ensure_pipeline() で必ず設定される
        self.output_dir.mkdir(parents=True, exist_ok=True)

        audio_chunks: list[np.ndarray] = []
        try:
            for result in self._pipeline(text, voice=self.voice, speed=self.speed):
                if result.audio is not None:
                    # CUDA テンソルは先に CPU へ移してから numpy 変換
                    audio_chunks.append(result.audio.cpu().numpy())
        except Exception as e:
            logger.error(f"TTSNarrator: フレーム {frame_id} の音声合成失敗: {e}")
            return None

        if not audio_chunks:
            logger.warning(
                f"TTSNarrator: フレーム {frame_id} で音声チャンクが生成されませんでした"
            )
            return None

        audio = np.concatenate(audio_chunks)
        out_path = self.output_dir / f"{frame_id}.wav"
        # PCM_16 を明示して互換性の高い 16-bit PCM WAV を出力（32-bit float WAV は非対応デバイスあり）
        sf.write(out_path, audio, self.SAMPLE_RATE, subtype="PCM_16")
        duration = len(audio) / self.SAMPLE_RATE
        logger.info(f"TTSNarrator: 保存 → {out_path} ({duration:.1f}秒)")
        return out_path
