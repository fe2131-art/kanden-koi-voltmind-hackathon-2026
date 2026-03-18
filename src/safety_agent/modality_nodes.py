"""
Modality processing nodes for LangGraph fan-out architecture.

各モダリティ（視覚・音声など）の処理を独立したノードに分離。
vision_node / audio_node で並列実行され、fuse_modalities で統合される。
"""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import librosa
import soundfile as sf
from openai import OpenAI

from .schema import (
    AudioCue,
    VisionAnalysisResult,
    VisionOverallAssessment,
)

logger = logging.getLogger(__name__)


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
    ):
        self.model = model
        self.provider = provider
        self.default_prompt = default_prompt
        self.max_tokens = max_tokens

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

        # 5) 複数の { ... } ブロック抽出（最後のものを採用）
        # 説明文が前後にある場合、最後の JSON ブロックを採用
        matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text))
        if matches:
            # 最後のマッチをパース試行
            for match in reversed(matches):
                try:
                    candidate = match.group(0)
                    return json.loads(candidate)
                except Exception:
                    continue

        return None

    def analyze(
        self,
        image_path: str,
        prev_image_path: Optional[str] = None,
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[VisionAnalysisResult]:
        """2枚の画像（現フレーム + 前フレーム）を VLM で分析し構造化結果を返す。

        prev_image_path が None の場合は同じ画像を2枚送る（最初のフレーム対応）。
        """
        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return None

        if prompt is None:
            prompt = self.default_prompt or "この画像を詳しく説明してください。"

        if max_tokens is None:
            max_tokens = self.max_tokens or 2048

        # 現フレームをエンコード
        current_url, _ = self._encode_image(image_path)

        # 前フレーム（なければ同じ画像をフォールバック）
        if prev_image_path and Path(prev_image_path).exists():
            prev_url, _ = self._encode_image(prev_image_path)
        else:
            prev_url = current_url

        # コンテンツブロック: テキスト + 1枚目（先の時刻）+ 2枚目（現在）
        if self.provider == "vllm":
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": prev_url}},
                {"type": "image_url", "image_url": {"url": current_url}},
            ]
        else:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": prev_url, "detail": "high"}},
                {
                    "type": "image_url",
                    "image_url": {"url": current_url, "detail": "high"},
                },
            ]

        try:
            if self.provider == "vllm":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_completion_tokens=max_tokens,
                )

            raw = response.choices[0].message.content or ""
            parsed = self._parse_vision_json(raw)
            if parsed is None:
                logger.warning(
                    f"[vision_analyze] VLM response could not be parsed as JSON. First 300 chars:\n{raw[:300]}"
                )
                # フォールバック: scene_description のみで VisionAnalysisResult を構築
                return VisionAnalysisResult(
                    scene_description=raw[:500] if raw else "No response",
                    overall_assessment=VisionOverallAssessment(
                        severity="unknown", reason="JSON parse failed"
                    ),
                )

            return VisionAnalysisResult.model_validate(parsed)

        except Exception as e:
            logger.error(f"Vision API error: {e}", exc_info=True)
            return None

    @staticmethod
    def _encode_image_bytes(image_bytes: bytes, media_type: str) -> str:
        """画像バイト列を Base64 エンコードし、data URL を返す。"""
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{media_type};base64,{image_data}"

    def analyze_bytes_raw(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[dict]:
        """画像バイト列をVLMで分析し、生JSON辞書を返す（型変換なし）。

        深度解析結果画像（side-by-side PNG）の分析用。
        VisionAnalysisResult ではなく dict を返すのは、DepthAnalysisResult への変換を
        ノードレベルで行うため。
        """
        if prompt is None:
            prompt = self.default_prompt or "この画像を詳しく説明してください。"

        if max_tokens is None:
            max_tokens = self.max_tokens or 2048

        image_url = self._encode_image_bytes(image_bytes, media_type)

        content: list[dict[str, Any]]
        if self.provider == "vllm":
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"},
                },
            ]

        try:
            if self.provider == "vllm":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_completion_tokens=max_tokens,
                )

            raw = response.choices[0].message.content or ""

            parsed = self._parse_vision_json(raw)
            if parsed is None:
                # エラーログにも raw の最初の 500 文字を含める
                logger.warning(
                    f"[depth_analysis] VLM response could not be parsed as JSON. "
                    f"First 500 chars:\n{raw[:500]}"
                )
                # フォールバック: 生テキストを scene_description として返し、depth_layers はデフォルト値
                return {
                    "scene_description": raw[:500]
                    if raw
                    else "VLM response could not be parsed",
                    "depth_layers": [],
                }

            return parsed

        except Exception as e:
            logger.error(f"Vision API error (depth analysis): {e}", exc_info=True)
            return None


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
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.default_prompt = default_prompt
        self.max_tokens = max_tokens
        self.provider = provider
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
    def _parse_audio_json(text: str) -> list[dict[str, Any]]:
        parsed = VisionAnalyzer._parse_vision_json(text)
        if parsed is None:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            events = parsed.get("events")
            if isinstance(events, list):
                return [item for item in events if isinstance(item, dict)]
        return []

    def _encode_audio_window(
        self,
        audio_path: str,
        sample_rate: int,
        video_timestamp: Optional[float],
        window_seconds: Optional[float],
    ) -> str:
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        total_samples = len(audio)

        if video_timestamp is None:
            logger.warning(
                "Audio timestamp missing; using full audio without trimming."
            )
            trimmed = audio
        else:
            end_sample = int(max(0.0, video_timestamp) * sr)
            end_sample = min(end_sample, total_samples)
            lookback_seconds = (
                self.window_seconds if window_seconds is None else window_seconds
            )
            lookback_samples = int(max(0.0, lookback_seconds) * sr)
            start_sample = max(0, end_sample - lookback_samples)
            trimmed = audio[start_sample:end_sample]

        if trimmed.size == 0:
            return ""

        buffer = BytesIO()
        sf.write(buffer, trimmed, sr, format="WAV")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

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

        if not Path(audio_input).exists():
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

        try:
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
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
            }
            if self.provider == "vllm":
                create_kwargs["max_tokens"] = max_tokens
            else:
                create_kwargs["max_completion_tokens"] = max_tokens

            response = self.client.chat.completions.create(**create_kwargs)
            raw = self._stringify_message_content(response.choices[0].message.content)
            events = self._parse_audio_json(raw)
            if not events:
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
            import torch
            from depth_anything_3.api import DepthAnything3

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
        except ImportError as e:
            logger.warning(
                f"depth_anything_3 not available: {e}. DepthEstimator will not work."
            )
        except Exception as e:
            logger.warning(f"Failed to initialize DepthEstimator: {e}")

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
        import cv2
        import numpy as np

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
        from io import BytesIO

        import cv2
        import numpy as np
        from PIL import Image

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
            import numpy as np

            # 推論実行（GPU は スレッドセーフでないため lock で保護）
            with self._lock:
                prediction = self._model.inference(
                    [str(image_path)],
                    process_res=self.process_res,
                    process_res_method="upper_bound_resize",
                )

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
    def make_side_by_side_bytes(rgb_path: str, infrared_path: str) -> Optional[bytes]:
        """RGB 画像と赤外線画像を横並びに結合し PNG バイト列を返す。

        Args:
            rgb_path: RGB フレームファイルパス
            infrared_path: 赤外線フレームファイルパス

        Returns:
            Side-by-side PNG バイト列。どちらかのファイルが存在しない場合は None。

        Note:
            赤外線画像がグレースケールの場合、.convert("RGB") で 3 チャネルに変換。
            RGB サイズに赤外線をリサイズしてから結合する。
        """
        from io import BytesIO

        import numpy as np
        from PIL import Image

        if not Path(rgb_path).exists():
            logger.warning(f"RGB image not found: {rgb_path}")
            return None
        if not Path(infrared_path).exists():
            logger.warning(f"Infrared image not found: {infrared_path}")
            return None

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        infrared = np.array(Image.open(infrared_path).convert("RGB"))

        # サイズを RGB に合わせてリサイズ
        if rgb.shape[:2] != infrared.shape[:2]:
            inf_img = Image.fromarray(infrared).resize(
                (rgb.shape[1], rgb.shape[0]), Image.Resampling.LANCZOS
            )
            infrared = np.array(inf_img)

        merged = np.concatenate([rgb, infrared], axis=1)
        buf = BytesIO()
        Image.fromarray(merged).save(buf, format="PNG")
        return buf.getvalue()
