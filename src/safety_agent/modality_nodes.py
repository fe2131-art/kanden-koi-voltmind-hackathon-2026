"""
Modality processing nodes for LangGraph fan-out architecture.

各モダリティ（視覚・音声など）の処理を独立したノードに分離。
vision_node / audio_node で並列実行され、fuse_modalities で統合される。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from openai import OpenAI

from .schema import (
    AudioCue,
    BoundingBox,
    DetectedObject,
    VisionAnalysisResult,
)

logger = logging.getLogger(__name__)


# ─── 結果型 ────────────────────────────────────────────────────


@dataclass
class ModalityResult:
    """vision_node / audio_node が AgentState に書き込む統一結果型。"""

    modality_name: str  # "vision" | "audio" | "lidar" etc.
    objects: list[DetectedObject] = field(default_factory=list)
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
        """LLM テキスト出力から JSON を抽出・パース。"""
        # 1) 直接パース
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2) markdown コードブロック抽出
        m = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # 3) { ... } 抽出（ネスト対応）
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
                {"type": "image_url", "image_url": {"url": current_url, "detail": "high"}},
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
                logger.warning("VLM response could not be parsed as JSON; raw text returned")
                # フォールバック: summary のみで VisionAnalysisResult を構築
                return VisionAnalysisResult(summary=raw[:500] if raw else "No response")

            return VisionAnalysisResult.model_validate(parsed)

        except Exception as e:
            logger.error(f"Vision API error: {e}")
            return None


# ─── YOLODetector ──────────────────────────────────────────────


class YOLODetector:
    """YOLO モデルをラップ。スレッドセーフのため Lock を保持。"""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(model_path)
        except ImportError:
            self._model = None
        self._lock = threading.Lock()  # ultralytics はスレッドセーフでないため

    def detect(self, image_path: str) -> list[DetectedObject]:
        """YOLO 検出を実行。マルチスレッド対応。"""
        if not image_path or not os.path.exists(image_path):
            return self._simple_image_analysis(image_path)

        if not self._model:
            return self._simple_image_analysis(image_path)

        try:
            with self._lock:
                res = self._model(image_path, verbose=False)[0]
            out: list[DetectedObject] = []
            for b in res.boxes:
                cls = int(b.cls.item())
                conf = float(b.conf.item())
                label = self._model.names.get(cls, str(cls))
                xyxy = b.xyxy[0].tolist()
                out.append(
                    DetectedObject(
                        label=label,
                        confidence=conf,
                        bbox=BoundingBox(
                            x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]
                        ),
                    )
                )
            if out:
                logger.debug(f"YOLO detected {len(out)} objects")
            return out
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}, using simple analysis")
            return self._simple_image_analysis(image_path)

    def _simple_image_analysis(self, image_path: str) -> list[DetectedObject]:
        """YOLO が利用できない場合のフォールバック。"""
        if not image_path or not os.path.exists(image_path):
            return []

        try:
            from PIL import Image

            img = Image.open(image_path)
            width, height = img.size

            objects = []
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)

            if file_size_mb > 0.1:
                objects.append(
                    DetectedObject(
                        label="foreground_object",
                        confidence=0.5,
                        bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.9, y2=0.8),
                    )
                )

            aspect_ratio = width / height if height > 0 else 1.0
            if aspect_ratio > 1.5:
                objects.append(
                    DetectedObject(
                        label="background_landscape",
                        confidence=0.6,
                        bbox=BoundingBox(x1=0.0, y1=0.5, x2=1.0, y2=1.0),
                    )
                )

            return objects
        except Exception as e:
            logger.warning(f"Simple image analysis error: {e}")
            return []


# ─── AudioAnalyzer ──────────────────────────────────────────────


class AudioAnalyzer:
    """音声テキストから AudioCue リストを抽出するヒューリスティック。"""

    def analyze(self, audio_text: Optional[str]) -> list[AudioCue]:
        """音声テキストを解析して AudioCue リストを返す。"""
        if not audio_text:
            return []

        t = audio_text.lower()
        cues: list[AudioCue] = []

        # Example heuristics; replace with audio model output
        if "right" in t and ("car" in t or "vehicle" in t or "approach" in t):
            cues.append(
                AudioCue(
                    cue="vehicle_approaching",
                    confidence=0.7,
                    direction="right",
                    evidence=audio_text,
                )
            )
        if "left" in t and ("car" in t or "vehicle" in t or "approach" in t):
            cues.append(
                AudioCue(
                    cue="vehicle_approaching",
                    confidence=0.7,
                    direction="left",
                    evidence=audio_text,
                )
            )
        return cues
