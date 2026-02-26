import os
from pathlib import Path
from typing import List, Optional

import httpx

from .schema import (
    AudioCue,
    BoundingBox,
    DetectedObject,
    Hazard,
    Observation,
    PerceptionIR,
    UnobservedRegion,
)

# =========================
# Vision Language Model (VLM) クライアント
# =========================


class VisionAnalyzer:
    """OpenAI互換のVision API を使用して画像を分析するクラス。"""

    def __init__(
        self, base_url: str, model: str, api_key: str = "EMPTY", timeout_s: float = 60.0
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        # GPT-5 系モデルは max_tokens ではなく max_completion_tokens を使用
        self._use_max_completion_tokens = "gpt-5" in model.lower()

    def analyze(
        self, image_path: str, prompt: Optional[str] = None, max_tokens: int = 4000
    ) -> str:
        """画像を分析してテキスト結果を返す。"""
        import base64

        if not Path(image_path).exists():
            return f"Image not found: {image_path}"

        # デフォルトプロンプト（日本語・安全性フォーカス）
        if prompt is None:
            prompt = (
                "この画像の安全性を分析してください。以下に焦点を当ててください：\n"
                "1. 人物や動く物体の有無\n"
                "2. 潜在的な危険や危機的な状況\n"
                "3. ブラインドスポットや注意が必要な領域\n"
                "4. 総合的な安全性評価（安全/注意/危険）\n"
                "簡潔に（100-200語程度）答えてください。"
            )

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
                            "image_url": {
                                "url": f"data:{image_type};base64,{image_data}"
                            },
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
                    error_msg = error_detail.get("message", r.text)
                except Exception:
                    error_msg = r.text
                return f"Vision API error {r.status_code}: {error_msg}"
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            if not content or not content.strip():
                # デバッグ情報：APIレスポンス全体をログに出力
                import json as json_module

                return f"Vision analysis returned empty response from {self.model}. Response: {json_module.dumps(data, ensure_ascii=False)}"
            return content


# =========================
# Perception module (stub + hooks)
# =========================


class Perceiver:
    """
    Replace internals with:
      - YOLO (ultralytics) detections
      - VLM caption/scene graph -> hazards/unobserved
      - audio classifier -> cues
    Keep output stable as PerceptionIR.
    """

    def __init__(self, enable_yolo: bool = False, vlm: Optional[VisionAnalyzer] = None):
        self.enable_yolo = enable_yolo
        self.vlm = vlm
        self._yolo = None
        if enable_yolo:
            try:
                from ultralytics import YOLO  # optional

                self._yolo = YOLO("yolov8n.pt")
            except Exception:
                self._yolo = None

    def _simple_image_analysis(self, image_path: str) -> List[DetectedObject]:
        """Simple image analysis without YOLO (Vision heuristics)."""
        if not image_path or not os.path.exists(image_path):
            return []

        try:
            import os as os_module

            from PIL import Image

            img = Image.open(image_path)
            width, height = img.size

            # Simple vision heuristics
            objects = []

            # Analyze image properties
            file_size_mb = os_module.path.getsize(image_path) / (1024 * 1024)

            # Heuristic: Larger files might contain more objects
            if file_size_mb > 0.1:
                # Assume there might be foreground objects
                objects.append(
                    DetectedObject(
                        label="foreground_object",
                        confidence=0.5,
                        bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.9, y2=0.8),
                    )
                )

            # Heuristic: Image with certain dimensions might have specific objects
            aspect_ratio = width / height if height > 0 else 1.0
            if aspect_ratio > 1.5:  # Wide image
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

    def _yolo_detect(self, image_path: str) -> List[DetectedObject]:
        """YOLO detection with fallback to simple analysis."""
        if not image_path:
            return []

        # Try YOLO if enabled
        if self._yolo:
            try:
                if not os.path.exists(image_path):
                    print(f"⚠️  Image file not found: {image_path}")
                    return self._simple_image_analysis(image_path)

                res = self._yolo(image_path, verbose=False)[0]
                out: List[DetectedObject] = []
                for b in res.boxes:
                    cls = int(b.cls.item())
                    conf = float(b.conf.item())
                    label = self._yolo.names.get(cls, str(cls))
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
        else:
            # Use simple analysis as fallback
            return self._simple_image_analysis(image_path)

    def _audio_to_cues(self, audio_text: Optional[str]) -> List[AudioCue]:
        if not audio_text:
            return []
        t = audio_text.lower()
        cues: List[AudioCue] = []
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

    def _infer_unobserved(
        self, obs: Observation, objects: List[DetectedObject], audio: List[AudioCue]
    ) -> List[UnobservedRegion]:
        # Placeholder: in reality use depth/segmentation/FOV geometry, or dataset-provided blindspots
        # Here we just create a few generic regions
        regions = [
            UnobservedRegion(
                region_id="blind_left",
                description="左側の死角（未確認領域）",
                risk=0.4,
                suggested_pan_deg=obs.camera_pose.pan_deg - 30,
            ),
            UnobservedRegion(
                region_id="blind_right",
                description="右側の死角（未確認領域）",
                risk=0.4,
                suggested_pan_deg=obs.camera_pose.pan_deg + 30,
            ),
            UnobservedRegion(
                region_id="blind_back",
                description="背後の死角（未確認領域）",
                risk=0.3,
                suggested_pan_deg=obs.camera_pose.pan_deg + 180,
            ),
        ]
        # Increase risk if audio suggests something from a direction
        for c in audio:
            if c.direction == "right":
                for r in regions:
                    if r.region_id == "blind_right":
                        r.risk = min(1.0, r.risk + 0.3)
            if c.direction == "left":
                for r in regions:
                    if r.region_id == "blind_left":
                        r.risk = min(1.0, r.risk + 0.3)
        return regions

    def _infer_hazards(
        self, objects: List[DetectedObject], audio: List[AudioCue]
    ) -> List[Hazard]:
        hazards: List[Hazard] = []
        labels = {o.label for o in objects}

        # Check for detected objects
        if "person" in labels:
            hazards.append(
                Hazard(
                    hazard_type="human_present",
                    confidence=0.6,
                    related_objects=["person"],
                    evidence="person detected",
                )
            )

        # Generic hazard for foreground objects
        if "foreground_object" in labels:
            hazards.append(
                Hazard(
                    hazard_type="unidentified_foreground_object",
                    confidence=0.5,
                    related_objects=["foreground_object"],
                    evidence="foreground object detected in image",
                )
            )

        # Audio-based hazards
        if any(c.cue == "vehicle_approaching" for c in audio):
            hazards.append(
                Hazard(
                    hazard_type="vehicle_possible_blindspot",
                    confidence=0.6,
                    related_objects=[],
                    evidence="audio cue suggests vehicle",
                )
            )

        return hazards

    def run(self, obs: Observation) -> PerceptionIR:
        objects = self._yolo_detect(obs.image_path) if obs.image_path else []
        audio = self._audio_to_cues(obs.audio_text)
        unobs = self._infer_unobserved(obs, objects, audio)
        hazards = self._infer_hazards(objects, audio)

        # VLM による画像分析
        vision_description: Optional[str] = None
        if self.vlm and obs.image_path:
            try:
                vision_description = self.vlm.analyze(obs.image_path)
            except Exception as e:
                vision_description = f"VLM 分析エラー: {e}"

        return PerceptionIR(
            obs_id=obs.obs_id,
            camera_pose=obs.camera_pose,
            objects=objects,
            hazards=hazards,
            unobserved=unobs,
            audio=audio,
            vision_description=vision_description,
        )
