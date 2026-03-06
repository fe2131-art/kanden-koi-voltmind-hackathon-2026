"""
Modality processing nodes for LangGraph fan-out architecture.

各モダリティ（視覚・音声など）の処理を独立したノードに分離。
vision_node / audio_node で並列実行され、fuse_modalities で統合される。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from .schema import (
    AudioCue,
    BoundingBox,
    DetectedObject,
)


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
    """OpenAI互換のVision API を使用して画像を分析するクラス。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 60.0,
        default_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.default_prompt = default_prompt
        self.max_tokens = max_tokens
        # GPT-5 系モデルは max_tokens ではなく max_completion_tokens を使用
        self._use_max_completion_tokens = "gpt-5" in model.lower()

    def analyze(
        self, image_path: str, prompt: Optional[str] = None, max_tokens: Optional[int] = None
    ) -> str:
        """画像を分析してテキスト結果を返す。"""
        import base64

        if not Path(image_path).exists():
            return f"Image not found: {image_path}"

        # デフォルトプロンプト（外部化済み）
        if prompt is None:
            if self.default_prompt is None:
                raise ValueError(
                    "VisionAnalyzer.analyze() にプロンプトが未指定で、"
                    "default_prompt も未設定です。"
                    "configs/prompt.yaml の vision_analysis.default_prompt を確認してください。"
                )
            prompt = self.default_prompt

        # デフォルト max_tokens（外部化済み）
        if max_tokens is None:
            if self.max_tokens is None:
                raise ValueError(
                    "VisionAnalyzer.analyze() の max_tokens が未指定で、"
                    "コンストラクタの max_tokens も未設定です。"
                    "configs/default.yaml の tokens.vision_max_completion_tokens を確認してください。"
                )
            max_tokens = self.max_tokens

        # 画像を読み込んでエンコード
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # 画像タイプを判定
        ext = Path(image_path).suffix.lower()
        image_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # モデル固有のパラメータでペイロードを構築
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_type};base64,{image_data}"},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                },
            ],
        }
        # GPT-5系モデルはカスタム temperature をサポートしないため、デフォルト値を使用
        if not self._use_max_completion_tokens:
            payload["temperature"] = 0.2

        # GPT-5系モデルは max_completion_tokens、その他は max_tokens を使用
        if self._use_max_completion_tokens:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens

        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                try:
                    error_detail = r.json().get("error", {})
                    error_msg = error_detail.get("message", "Unknown error")
                except Exception:
                    # HTML エラーレスポンスの場合、簡潔なエラーメッセージに変換
                    if "<html>" in r.text.lower():
                        error_msg = f"HTTP {r.status_code} (likely upstream error)"
                    else:
                        error_msg = r.text[:200]  # テキストエラーは最初の200文字
                return f"Vision API error {r.status_code}: {error_msg}"
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            if not content or not content.strip():
                import json as json_module

                return f"Vision analysis returned empty response from {self.model}. Response: {json_module.dumps(data, ensure_ascii=False)}"
            return content


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
                print(f"✅ YOLO detected {len(out)} objects")
            return out
        except Exception as e:
            print(f"⚠️  YOLO detection failed: {e}, using simple analysis")
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
            print(f"⚠️  Simple image analysis error: {e}")
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
