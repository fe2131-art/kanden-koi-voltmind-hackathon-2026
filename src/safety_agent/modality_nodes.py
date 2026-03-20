"""
Modality processing nodes for LangGraph fan-out architecture.

各モダリティ（視覚・音声など）の処理を独立したノードに分離。
vision_node / audio_node で並列実行され、fuse_modalities で統合される。
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import re
import tempfile
import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
from depth_anything_3.api import DepthAnything3
from openai import OpenAI
from PIL import Image

# PIL 9.1.0 未満では Image.Resampling が存在しないため互換エイリアスを定義
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]

from .schema import (
    AudioCue,
    NormalizedBBox,
    Sam3AnalysisResult,
    Sam3Region,
    VisionAnalysisResult,
)

logger = logging.getLogger(__name__)


# ─── vLLM file:// transport ────────────────────────────────────

# vLLM の --allowed-local-media-path に収まるデフォルト temp ディレクトリ
# （プロジェクトルート直下の tmp/）
_VLLM_TMP_DIR: Path = (Path(__file__).parents[2] / "tmp").resolve()


@contextlib.contextmanager
def _vllm_image_file(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    suffix: str = ".jpg",
    tmp_dir: Optional[Path] = None,
) -> Iterator[str]:
    """vLLM 向けに file:// URI を yield するコンテキストマネージャー。

    image_path 指定時: 絶対パス解決のみ行い temp file を作らず yield。
    image_bytes 指定時: NamedTemporaryFile に書き出し、finally で削除。

    Args:
        tmp_dir: temp file の書き出し先ディレクトリ。None の場合は _VLLM_TMP_DIR
                 （プロジェクトルート直下の tmp/）を使用。
                 vLLM サーバーの --allowed-local-media-path に合わせて設定すること。
    """
    if image_path is not None:
        yield f"file://{Path(image_path).resolve()}"
    elif image_bytes is not None:
        # with ブロックで close → write 失敗時もハンドルリークなし
        resolved_dir = tmp_dir or _VLLM_TMP_DIR
        resolved_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, dir=resolved_dir
        ) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            yield f"file://{tmp_path}"
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        raise ValueError(
            "_vllm_image_file: image_path か image_bytes のどちらかを指定してください。"
        )


# ─── 結果型 ────────────────────────────────────────────────────


@dataclass
class ModalityResult:
    """vision_node / audio_node が AgentState に書き込む統一結果型。"""

    modality_name: str  # "vision" | "audio" | "depth" etc.
    audio_cues: list[AudioCue] = field(default_factory=list)
    description: Optional[str] = None  # VLM テキスト出力
    extra: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ─── VisionAnalyzer（VLM） ─────────────────────────────────────


class VisionAnalyzer:
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "EMPTY",
        timeout: float = 3600.0,
        default_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        provider: str = "openai",
        vllm_tmp_dir: Optional[Path] = None,
    ):
        self.model = model
        self.provider = provider
        self.default_prompt = default_prompt
        self.max_tokens = max_tokens
        # vLLM の --allowed-local-media-path に合わせた temp file 書き出し先
        # None の場合は _vllm_image_file 内で Path.home() にフォールバック
        self.vllm_tmp_dir = vllm_tmp_dir

        # vLLM 用に base_url を正規化
        if provider == "vllm":
            base_url = base_url.rstrip("/")
            if base_url.endswith("/chat/completions"):
                raise ValueError(
                    f"base_url にはフル endpoint ではなく /v1 までを渡してください: {base_url}"
                )
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    @staticmethod
    def _encode_image(image_path: str) -> tuple[str, str]:
        """画像をBase64エンコードし、(image_url, media_type) を返す。"""
        ext = Path(image_path).suffix.lower()
        media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{media_type};base64,{image_data}", media_type

    @staticmethod
    def _parse_vision_json(text: str) -> Optional[dict]:
        """LLM テキスト出力から JSON を抽出・パース（4段階＋最終フォールバック）。"""
        # 1) 直接パース
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2) markdown コードブロック抽出（基本形）
        m = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # 3) { ... } 抽出（ネスト対応、最初のブロック）
        stripped = text.strip()
        if stripped.startswith("{"):
            depth = 0
            for i, ch in enumerate(stripped):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(stripped[: i + 1])
                        except Exception:
                            break

        # 4) より柔軟な Markdown コードブロック抽出
        # バックティックの後に改行がない場合や、前後に空白・テキストがある場合に対応
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # 5) テキスト内の { ... } ブロックを全て抽出（任意ネスト深度対応）
        # 説明文が前後にある場合でも深くネストされた JSON を正しく抽出できる。
        # 正規表現は任意深さのネストを表現できないため、括弧カウントスキャンで代替。
        candidates: list[str] = []
        _depth = 0
        _start = -1
        for _i, _ch in enumerate(text):
            if _ch == "{":
                if _depth == 0:
                    _start = _i
                _depth += 1
            elif _ch == "}":
                _depth -= 1
                if _depth == 0 and _start != -1:
                    candidates.append(text[_start : _i + 1])
                    _start = -1
        for candidate in reversed(candidates):
            try:
                return json.loads(candidate)
            except Exception:
                continue

        return None

    def analyze(
        self,
        image_path: Optional[str] = None,
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        image_bytes: Optional[bytes] = None,
        media_type: str = "image/jpeg",
    ) -> Optional[VisionAnalysisResult]:
        """現フレーム画像を VLM で分析し構造化結果を返す。

        image_bytes が指定された場合はバイト列を直接使用し、ファイル読み込みをスキップ（改善B）。
        時系列比較が必要な場合は temporal_node（side-by-side 結合画像）を使用する。
        """
        if image_bytes is None:
            # パス経由の従来フロー
            if not image_path or not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                return None

        if prompt is None:
            prompt = self.default_prompt or "この画像を詳しく説明してください。"

        if max_tokens is None:
            max_tokens = self.max_tokens or 2048

        # URL 解決: vLLM → file:// URI、OpenAI → base64 data URI
        if self.provider == "vllm":
            suffix = Path(image_path).suffix if image_path else ".jpg"
            _path = image_path if (image_path and Path(image_path).exists()) else None
            with _vllm_image_file(
                image_path=_path,
                image_bytes=image_bytes if _path is None else None,
                suffix=suffix or ".jpg",
                tmp_dir=self.vllm_tmp_dir,
            ) as url:
                content = [{"type": "text", "text": prompt}, self._image_block(url)]
                raw = self._call_vlm(content, max_tokens)
        else:
            current_url = (
                self._encode_image_bytes(image_bytes, media_type)
                if image_bytes is not None
                else self._encode_image(image_path)[0]  # type: ignore[arg-type]
            )
            if not current_url:
                return None
            content = [{"type": "text", "text": prompt}, self._image_block(current_url)]
            raw = self._call_vlm(content, max_tokens)

        if raw is None:
            return None

        # 共通パース処理
        parsed = self._parse_vision_json(raw)
        if parsed is None:
            logger.warning(
                f"[vision_analyze] VLM response could not be parsed as JSON. First 300 chars:\n{raw[:300]}"
            )
            return VisionAnalysisResult(
                scene_description=raw[:500] if raw else "No response",
                overall_risk="unknown",
                confidence_score=0.0,
            )

        return VisionAnalysisResult.model_validate(parsed)

    @staticmethod
    def _encode_image_bytes(
        image_bytes: Optional[bytes], media_type: str
    ) -> Optional[str]:
        """画像バイト列を Base64 エンコードし、data URL を返す。None の場合は None を返す。"""
        if image_bytes is None:
            return None
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{media_type};base64,{image_data}"

    def _image_block(self, url: str) -> dict[str, Any]:
        """image_url コンテンツブロックを生成。OpenAI は detail: high を付与。"""
        block: dict[str, Any] = {"type": "image_url", "image_url": {"url": url}}
        if self.provider != "vllm":
            block["image_url"]["detail"] = "high"
        return block

    def _call_vlm(
        self, content: list[dict[str, Any]], max_tokens: int
    ) -> Optional[str]:
        """vLLM / OpenAI 共通の chat.completions 呼び出し。生テキストを返す。失敗時は None。"""
        try:
            if self.provider == "vllm":
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],  # type: ignore[arg-type]
                    max_tokens=max_tokens,
                )
            else:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],  # type: ignore[arg-type]
                    max_completion_tokens=max_tokens,
                )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Vision API error: {e}", exc_info=True)
            return None

    def analyze_bytes_raw(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        image_path: Optional[str] = None,
    ) -> Optional[dict]:
        """画像バイト列をVLMで分析し、生JSON辞書を返す（型変換なし）。

        深度解析結果画像（side-by-side PNG）の分析用。
        VisionAnalysisResult ではなく dict を返すのは、DepthAnalysisResult への変換を
        ノードレベルで行うため。

        Args:
            image_path: vLLM 向けに既存ファイルパスが分かる場合に指定。
                        指定時は temp file を作らず file:// で直接参照する。
        """
        if prompt is None:
            prompt = self.default_prompt or "この画像を詳しく説明してください。"

        if max_tokens is None:
            max_tokens = self.max_tokens or 2048

        # URL 解決: vLLM → file:// URI、OpenAI → base64 data URI
        suffix = ".png" if "png" in media_type else ".jpg"

        if self.provider == "vllm":
            _path = image_path if (image_path and Path(image_path).exists()) else None
            with _vllm_image_file(
                image_path=_path,
                image_bytes=image_bytes if _path is None else None,
                suffix=suffix,
                tmp_dir=self.vllm_tmp_dir,
            ) as url:
                content = [{"type": "text", "text": prompt}, self._image_block(url)]
                raw = self._call_vlm(content, max_tokens)
        else:
            image_url = self._encode_image_bytes(image_bytes, media_type)
            if image_url is None:
                logger.warning("analyze_bytes_raw: image_bytes is None, skipping")
                return None
            content = [{"type": "text", "text": prompt}, self._image_block(image_url)]
            raw = self._call_vlm(content, max_tokens)

        if raw is None:
            return None

        # 共通パース処理
        parsed = self._parse_vision_json(raw)
        if parsed is None:
            logger.warning(
                f"[depth_analysis] VLM response could not be parsed as JSON. "
                f"First 500 chars:\n{raw[:500]}"
            )
            return {
                "scene_description": raw[:500]
                if raw
                else "VLM response could not be parsed",
                "depth_layers": [],
                "overall_risk": "unknown",
                "confidence_score": 0.0,
            }

        return parsed


# ─── AudioAnalyzer ──────────────────────────────────────────────


class AudioAnalyzer:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "EMPTY",
        timeout: float = 3600.0,
        sample_rate: int = 16000,
        window_seconds: float = 3.0,
        default_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        provider: str = "vllm",
        vllm_tmp_dir: Optional[Path] = None,
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.default_prompt = default_prompt
        self.max_tokens = max_tokens
        self.provider = provider
        self.vllm_tmp_dir = vllm_tmp_dir
        self.client: Optional[OpenAI] = None

        if self.model:
            # LLM 用に base_url を正規化
            if provider == "vllm":
                base_url = base_url.rstrip("/")
                if base_url.endswith("/chat/completions"):
                    raise ValueError(
                        f"base_url にはフル endpoint ではなく /v1 までを渡してください: {base_url}"
                    )
                if not base_url.endswith("/v1"):
                    base_url = f"{base_url}/v1"

            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )

    @staticmethod
    def _stringify_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
            return "\n".join(parts)
        return ""

    @staticmethod
    def _parse_audio_json(text: str) -> Optional[list[dict[str, Any]]]:
        """JSON をパースして events リストを返す。パース失敗時は None（空イベントは []）。"""
        parsed = VisionAnalyzer._parse_vision_json(text)
        if parsed is None:
            return None
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            events = parsed.get("events")
            if isinstance(events, list):
                return [item for item in events if isinstance(item, dict)]
        return None

    def _trim_audio_window(
        self,
        audio_path: str,
        sample_rate: int,
        video_timestamp: Optional[float],
        window_seconds: Optional[float],
    ) -> tuple[Any, Any]:
        """音声をロードしてタイムスタンプでトリムし (trimmed_audio, sr) を返す。

        trimmed_audio がゼロ長の場合あり（呼び出し側で .size == 0 を確認すること）。
        """
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        total_samples = len(audio)

        if video_timestamp is None:
            logger.warning(
                "Audio timestamp missing; using full audio without trimming."
            )
            return audio, sr

        end_sample = int(max(0.0, video_timestamp) * sr)
        end_sample = min(end_sample, total_samples)
        lookback_seconds = (
            self.window_seconds if window_seconds is None else window_seconds
        )
        lookback_samples = int(max(0.0, lookback_seconds) * sr)
        start_sample = max(0, end_sample - lookback_samples)
        return audio[start_sample:end_sample], sr

    def _encode_audio_window(
        self,
        audio_path: str,
        sample_rate: int,
        video_timestamp: Optional[float],
        window_seconds: Optional[float],
    ) -> str:
        """OpenAI 向け: トリム済み音声を base64 WAV 文字列で返す。"""
        trimmed, sr = self._trim_audio_window(
            audio_path, sample_rate, video_timestamp, window_seconds
        )
        if trimmed.size == 0:
            return ""
        buffer = BytesIO()
        sf.write(buffer, trimmed, sr, format="WAV")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @contextlib.contextmanager
    def _trim_to_temp_wav(
        self,
        audio_path: str,
        sample_rate: int,
        video_timestamp: Optional[float],
        window_seconds: Optional[float],
    ) -> Iterator[str]:
        """vLLM 向け: トリム済み音声を temp WAV に書き出し、file:// URI を yield。

        空音声の場合は "" を yield する。
        """
        trimmed, sr = self._trim_audio_window(
            audio_path, sample_rate, video_timestamp, window_seconds
        )
        if trimmed.size == 0:
            yield ""
            return
        # with ブロックで即 close → sf.write 失敗時もハンドルリークなし
        resolved_dir = self.vllm_tmp_dir or _VLLM_TMP_DIR
        resolved_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=resolved_dir
        ) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, trimmed, sr, format="WAV")
            yield f"file://{tmp_path}"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _normalize_audio_events(
        self,
        events: list[dict[str, Any]],
    ) -> list[AudioCue]:
        cues: list[AudioCue] = []
        for item in events:
            cue = item.get("cue")
            if not isinstance(cue, str) or not cue.strip():
                continue

            severity = item.get("severity", "unknown")
            if severity not in {"low", "medium", "high", "critical", "unknown"}:
                severity = "unknown"

            evidence = item.get("evidence")
            if evidence is None:
                evidence = ""
            elif not isinstance(evidence, str):
                evidence = str(evidence)

            cues.append(
                AudioCue(
                    cue=cue.strip(),
                    severity=severity,
                    evidence=evidence.strip() if evidence else "",
                )
            )
        return cues

    def analyze(
        self,
        audio_input: Optional[str],
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        video_timestamp: Optional[float] = None,
        previous_vision_summary: Optional[str] = None,
        window_seconds: Optional[float] = None,
    ) -> list[AudioCue]:
        if not audio_input:
            return []

        try:
            exists = Path(audio_input).exists()
        except OSError as e:
            logger.warning(f"Audio path check failed: {audio_input}: {e}")
            return []
        if not exists:
            logger.warning(f"Audio file not found: {audio_input}")
            return []

        if not self.client or not self.model:
            logger.warning(
                "AudioAnalyzer client is not configured; returning empty cues."
            )
            return []

        if prompt is None:
            base_prompt = self.default_prompt or (
                "Analyze the audio clip and return only hazard-related or attention-worthy "
                "audio events as JSON. Output only "
                '{"events": [{"cue": "short_event_name", "severity": "low|medium|high|critical|unknown", '
                '"evidence": "short evidence"}]}. '
                'If there is no relevant event, return {"events": []}.'
            )
            if previous_vision_summary:
                prompt = (
                    f"{base_prompt}\n"
                    f"Previous frame visual summary: {previous_vision_summary}\n"
                    "Use the visual summary only as background context. "
                    "Return only what is supported by the audio clip."
                )
            else:
                prompt = base_prompt

        if max_tokens is None:
            max_tokens = self.max_tokens or 2048

        raw: str = ""
        try:
            if self.provider == "vllm":
                # vLLM: temp WAV ファイル経由で file:// URI を使用
                with self._trim_to_temp_wav(
                    audio_input,
                    self.sample_rate,
                    video_timestamp=video_timestamp,
                    window_seconds=window_seconds,
                ) as audio_uri:
                    if not audio_uri:
                        return []
                    content = [
                        {"type": "text", "text": prompt},
                        {
                            "type": "audio_url",
                            "audio_url": {"url": audio_uri},
                        },
                    ]
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": content}],  # type: ignore[arg-type]
                        max_tokens=max_tokens,
                    )
                    raw = self._stringify_message_content(
                        response.choices[0].message.content
                    )
            else:
                # OpenAI: base64 input_audio（現状維持）
                audio_base64 = self._encode_audio_window(
                    audio_input,
                    self.sample_rate,
                    video_timestamp=video_timestamp,
                    window_seconds=window_seconds,
                )
                if not audio_base64:
                    return []
                content = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_base64, "format": "wav"},
                    },
                ]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],  # type: ignore[arg-type]
                    max_completion_tokens=max_tokens,
                )
                raw = self._stringify_message_content(
                    response.choices[0].message.content
                )

            events = self._parse_audio_json(raw)
            if events is None:
                logger.warning(
                    "[audio_analyze] Audio model response could not be parsed as JSON. "
                    f"Returning empty cues. Raw response (first 500 chars): {raw[:500]!r}"
                )
                return []
            return self._normalize_audio_events(events)
        except Exception as e:
            logger.error(f"Audio API error: {e}", exc_info=True)
            return []


# ─── DepthEstimator（Depth Anything 3） ──────────────────────────


class DepthEstimator:
    """Depth Anything 3 による深度推定。モデルロード失敗時は self._model = None。"""

    def __init__(
        self,
        model_family: str = "mono",
        model_size: str = "large",
        model_id: Optional[str] = None,
        process_res: int = 504,
    ) -> None:
        self.process_res = process_res
        self._model = None
        self._device = None
        self._lock = threading.Lock()

        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            resolved_model_id = self._resolve_model_id(
                model_family, model_size, model_id
            )
            self._model = DepthAnything3.from_pretrained(resolved_model_id).to(
                self._device
            )
            logger.info(
                f"DepthEstimator initialized: {resolved_model_id} on {self._device}"
            )
        except Exception as e:
            logger.error(
                f"DepthEstimator: model load failed, depth estimation disabled: {e}",
                exc_info=True,
            )
            self._model = None
            self._device = None

    def _resolve_model_id(
        self,
        model_family: str,
        model_size: str,
        model_id: Optional[str],
    ) -> str:
        """モデル ID を家族・サイズから解決。model_id が指定されたら優先。"""
        if model_id:
            return model_id

        if model_family == "mono":
            if model_size != "large":
                raise ValueError("mono は large のみ指定できます。")
            return "depth-anything/DA3MONO-LARGE"

        if model_family == "metric":
            if model_size != "large":
                raise ValueError("metric は large のみ指定できます。")
            return "depth-anything/DA3METRIC-LARGE"

        if model_family == "any":
            size_to_id = {
                "small": "depth-anything/DA3-SMALL",
                "base": "depth-anything/DA3-BASE",
                "large": "depth-anything/DA3-LARGE",
                "giant": "depth-anything/DA3-GIANT",
            }
            if model_size not in size_to_id:
                raise ValueError(f"unknown size for 'any' family: {model_size}")
            return size_to_id[model_size]

        raise ValueError(f"unknown model_family: {model_family}")

    def _depth_to_turbo_rgb(
        self,
        depth: Any,  # np.ndarray
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> Any:  # np.ndarray
        """深度を Turbo カラーマップで可視化。近いほど暖色、遠いほど寒色。"""

        depth = depth.astype(np.float32)
        valid = np.isfinite(depth)

        if not np.any(valid):
            raise ValueError("Depth map has no finite values.")

        lo = float(np.percentile(depth[valid], percentile_low))
        hi = float(np.percentile(depth[valid], percentile_high))

        if hi <= lo:
            norm = np.zeros_like(depth, dtype=np.float32)
        else:
            clipped = np.clip(depth, lo, hi)
            norm = (clipped - lo) / (hi - lo)

        # 近いものを暖色にしたいので反転
        norm = 1.0 - norm

        gray_u8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
        color_bgr = cv2.applyColorMap(gray_u8, cv2.COLORMAP_TURBO)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        return color_rgb

    def _make_side_by_side_bytes(
        self,
        rgb: Any,  # np.ndarray
        depth_vis_rgb: Any,  # np.ndarray
    ) -> bytes:
        """RGB と深度可視化を side-by-side で合成し、PNG バイト列を返す。"""

        if rgb.shape[:2] != depth_vis_rgb.shape[:2]:
            depth_vis_rgb = cv2.resize(
                depth_vis_rgb,
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        merged = np.concatenate([rgb, depth_vis_rgb], axis=1)
        img = Image.fromarray(merged)

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def estimate(self, image_path: str) -> Optional[bytes]:
        """画像から深度推定し、side-by-side PNG バイト列を返す。

        モデルが None の場合は None を返す（深度推定は利用不可）。
        エラー発生時は例外を raise しない（ModalityResult.error に設定される）。
        """
        if self._model is None:
            return None

        if not Path(image_path).exists():
            logger.warning(f"Image not found for depth estimation: {image_path}")
            return None

        try:
            # 推論実行（GPU は スレッドセーフでないため lock で保護）
            with self._lock:
                prediction = self._model.inference(
                    [str(image_path)],
                    process_res=self.process_res,
                    process_res_method="upper_bound_resize",
                )

            if prediction is None:
                logger.warning("Depth model returned None prediction")
                return None
            depth = prediction.depth[0].astype(np.float32)
            rgb = prediction.processed_images[0].astype(np.uint8)

            depth_vis_rgb = self._depth_to_turbo_rgb(depth)
            side_by_side_bytes = self._make_side_by_side_bytes(rgb, depth_vis_rgb)

            logger.debug(f"Depth estimation completed for {image_path}")
            return side_by_side_bytes

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}", exc_info=True)
            return None


# ─── InfraredImageAnalyzer（赤外線画像分析） ─────────────────────────────────


class InfraredImageAnalyzer:
    """RGB フレームと赤外線フレームを side-by-side 結合し VLM で分析するクラス。

    外部推論モデルは不要。画像結合のみ担当。
    DepthEstimator と異なり、赤外線フレームはすでに抽出済みのため、
    画像読み込み・結合・VLM 送信のみを実行する。
    """

    @staticmethod
    def make_side_by_side_bytes(
        rgb_path: str,
        infrared_path: str,
        rgb_bytes: Optional[bytes] = None,
    ) -> Optional[bytes]:
        """RGB 画像と赤外線画像を横並びに結合し PNG バイト列を返す。

        Args:
            rgb_path: RGB フレームファイルパス
            infrared_path: 赤外線フレームファイルパス
            rgb_bytes: RGB 画像のバイト列（指定時はファイル読み込みをスキップ・改善B）

        Returns:
            Side-by-side PNG バイト列。どちらかのファイルが存在しない場合は None。

        Note:
            赤外線画像がグレースケールの場合、.convert("RGB") で 3 チャネルに変換。
            RGB サイズに赤外線をリサイズしてから結合する。
        """

        try:
            if rgb_bytes is not None:
                rgb = np.array(Image.open(BytesIO(rgb_bytes)).convert("RGB"))
            else:
                if not Path(rgb_path).exists():
                    logger.warning(f"RGB image not found: {rgb_path}")
                    return None
                rgb = np.array(Image.open(rgb_path).convert("RGB"))
        except Exception as e:
            logger.error(f"InfraredImageAnalyzer: failed to load RGB image: {e}")
            return None

        if not Path(infrared_path).exists():
            logger.warning(f"Infrared image not found: {infrared_path}")
            return None
        try:
            infrared = np.array(Image.open(infrared_path).convert("RGB"))
        except Exception as e:
            logger.error(f"InfraredImageAnalyzer: failed to load infrared image: {e}")
            return None

        # サイズを RGB に合わせてリサイズ
        if rgb.shape[:2] != infrared.shape[:2]:
            inf_img = Image.fromarray(infrared).resize(
                (rgb.shape[1], rgb.shape[0]), _LANCZOS
            )
            infrared = np.array(inf_img)

        merged = np.concatenate([rgb, infrared], axis=1)
        buf = BytesIO()
        Image.fromarray(merged).save(buf, format="PNG")
        return buf.getvalue()


class TemporalImageAnalyzer:
    """現フレームと前フレームを横並び結合し VLM で変化を分析するクラス。
    外部推論モデルは不要。画像結合のみ担当。
    """

    @staticmethod
    def make_temporal_bytes(
        current_path: str,
        prev_path: str,
        current_bytes: Optional[bytes] = None,
    ) -> Optional[bytes]:
        """現フレームと前フレームを横並び結合し PNG バイト列を返す。

        Args:
            current_path: 現フレームのパス（右側）
            prev_path: 前フレームのパス（左側）
            current_bytes: 現フレームのバイト列（指定時はファイル読み込みをスキップ・改善B）

        Returns:
            PNG バイト列。どちらかのファイルが存在しない場合は None。

        Note:
            左: 前フレーム、右: 現フレーム（時系列順）の順に並べる。
            両フレームのサイズが異なる場合は、現フレームのサイズに合わせてリサイズする。
        """
        try:
            if current_bytes is not None:
                current = np.array(Image.open(BytesIO(current_bytes)).convert("RGB"))
            else:
                if not Path(current_path).exists():
                    logger.warning(f"Current frame not found: {current_path}")
                    return None
                current = np.array(Image.open(current_path).convert("RGB"))
        except Exception as e:
            logger.error(f"TemporalImageAnalyzer: failed to load current frame: {e}")
            return None

        if not Path(prev_path).exists():
            logger.warning(f"Previous frame not found: {prev_path}")
            return None
        try:
            prev = np.array(Image.open(prev_path).convert("RGB"))
        except Exception as e:
            logger.error(f"TemporalImageAnalyzer: failed to load previous frame: {e}")
            return None

        # サイズを current に合わせてリサイズ
        if current.shape[:2] != prev.shape[:2]:
            prev_img = Image.fromarray(prev).resize(
                (current.shape[1], current.shape[0]), _LANCZOS
            )
            prev = np.array(prev_img)

        # 左: 前フレーム, 右: 現フレーム（時系列順に左→右）
        merged = np.concatenate([prev, current], axis=1)
        buf = BytesIO()
        Image.fromarray(merged).save(buf, format="PNG")
        return buf.getvalue()


# ─── SAM3 セグメンテーション ────────────────────────────────────────────────────


class Sam3Analyzer:
    """SAM3 image mode によるテキストプロンプトベースセグメンテーション。

    SAM3 のインポートまたはモデルロードに失敗した場合は available=False で動作継続（fail-open）。
    GPU メモリ競合を防ぐため threading.Lock() を保持する。
    """

    def __init__(self, model_cfg: dict, device: Optional[str] = None) -> None:
        self.available = False
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from sam3 import Sam3Processor, build_sam3_image_model  # noqa: PLC0415

            checkpoint = model_cfg.get("checkpoint", "facebook/sam3-hiera-large")
            self._model = build_sam3_image_model(checkpoint).to(self._device).eval()
            self._processor = Sam3Processor.from_pretrained(checkpoint)
            self.available = True
            logger.info(f"Sam3Analyzer initialized on {self._device} ({checkpoint})")
        except Exception as e:
            logger.warning(f"Sam3Analyzer: model load failed, SAM3 disabled: {e}")

    def analyze(
        self,
        image_path: str,
        frame_id: str,
        prompts: list[str],
        score_threshold: float = 0.35,
        max_regions_per_prompt: int = 3,
        save_masks: bool = True,
        output_dir: str = "data/sam3_masks",
    ) -> Sam3AnalysisResult:
        """テキストプロンプトごとにセグメンテーションを実行し Sam3AnalysisResult を返す。

        Args:
            image_path: 入力画像のパス
            frame_id: フレーム識別子（region_id と mask ファイル名に使用）
            prompts: テキストプロンプトリスト（例: ["person", "cable on floor"]）
            score_threshold: スコアフィルタ閾値
            max_regions_per_prompt: prompt ごとの最大 region 数
            save_masks: True の場合 mask PNG を output_dir に保存
            output_dir: mask 保存先ディレクトリ

        Returns:
            Sam3AnalysisResult（失敗時は空の結果）
        """
        if not self.available or not self._model or not self._processor:
            return Sam3AnalysisResult()

        regions: list[Sam3Region] = []
        global_idx = 0

        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            output_path = Path(output_dir)

            with self._lock:
                for prompt in prompts:
                    try:
                        state = self._processor.init_state(image=image)
                        output = self._processor.set_text_prompt(
                            state=state, prompt=prompt
                        )
                        masks = output["masks"]  # [N, H, W]
                        boxes = output["boxes"]  # [N, 4] xyxy 絶対座標
                        scores = output["scores"]  # [N]
                    except Exception as e:
                        logger.warning(f"Sam3Analyzer: prompt '{prompt}' failed: {e}")
                        continue

                    # Tensor → numpy に変換
                    if hasattr(scores, "cpu"):
                        scores_np = scores.cpu().numpy()
                    else:
                        scores_np = np.array(scores)
                    if hasattr(boxes, "cpu"):
                        boxes_np = boxes.cpu().numpy()
                    else:
                        boxes_np = np.array(boxes)
                    if hasattr(masks, "cpu"):
                        masks_np = masks.cpu().numpy()
                    else:
                        masks_np = np.array(masks)

                    # スコアフィルタ → 降順ソート → 上位 N 件に絞る
                    valid_indices = np.where(scores_np >= score_threshold)[0]
                    if len(valid_indices) == 0:
                        continue
                    sorted_indices = valid_indices[
                        np.argsort(scores_np[valid_indices])[::-1]
                    ][:max_regions_per_prompt]

                    # prompt 文字列を安全なファイル名形式に変換
                    prompt_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", prompt)

                    for idx in sorted_indices:
                        region_id = f"sam3_{frame_id}_{global_idx:03d}"
                        score = float(scores_np[idx])

                        # bbox 正規化 (xyxy → 0-1)
                        bbox: Optional[NormalizedBBox] = None
                        if len(boxes_np) > idx:
                            box = boxes_np[idx]
                            if len(box) >= 4:
                                bbox = NormalizedBBox(
                                    x_min=max(0.0, min(1.0, float(box[0]) / width)),
                                    y_min=max(0.0, min(1.0, float(box[1]) / height)),
                                    x_max=max(0.0, min(1.0, float(box[2]) / width)),
                                    y_max=max(0.0, min(1.0, float(box[3]) / height)),
                                )

                        # mask PNG 保存
                        mask_path: Optional[str] = None
                        if save_masks and len(masks_np) > idx:
                            try:
                                mask_filename = (
                                    f"{frame_id}_region_{global_idx:03d}.png"
                                )
                                mask_file = output_path / mask_filename
                                mask_arr = masks_np[idx]
                                # bool/float → uint8
                                if mask_arr.dtype != np.uint8:
                                    mask_arr = (mask_arr > 0.5).astype(np.uint8) * 255
                                Image.fromarray(mask_arr).save(str(mask_file))
                                mask_path = str(mask_file)
                            except Exception as e:
                                logger.warning(
                                    f"Sam3Analyzer: mask save failed for "
                                    f"{frame_id}/{prompt_safe}/{idx}: {e}"
                                )

                        regions.append(
                            Sam3Region(
                                region_id=region_id,
                                prompt=prompt,
                                label=prompt,
                                score=score,
                                normalized_bbox=bbox,
                                mask_path=mask_path,
                            )
                        )
                        global_idx += 1

        except Exception as e:
            logger.warning(f"Sam3Analyzer.analyze() failed for {frame_id}: {e}")
            return Sam3AnalysisResult()

        confidence_score = (
            float(np.mean([r.score for r in regions])) if regions else 0.0
        )
        return Sam3AnalysisResult(regions=regions, confidence_score=confidence_score)
