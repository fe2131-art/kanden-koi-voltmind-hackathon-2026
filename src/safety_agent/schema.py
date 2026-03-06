from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# =========================
# Schemas (Pydantic)
# =========================


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectedObject(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: Optional[BoundingBox] = None


class AudioCue(BaseModel):
    cue: str  # e.g. "vehicle_approaching_right"
    confidence: float = Field(ge=0, le=1)
    direction: Optional[Literal["left", "right", "front", "back"]] = None
    evidence: Optional[str] = None


class UnobservedRegion(BaseModel):
    region_id: str
    description: str
    risk: float = Field(
        ge=0, le=1
    )  # heuristic risk that something hazardous may be there
    suggested_pan_deg: Optional[float] = None
    suggested_tilt_deg: Optional[float] = None


class Hazard(BaseModel):
    hazard_type: str  # e.g. "forklift", "human_near_machine", "vehicle"
    confidence: float = Field(ge=0, le=1)
    related_objects: List[str] = Field(default_factory=list)
    evidence: Optional[str] = None
    region_hint: Optional[str] = None  # region_id etc.


class CameraPose(BaseModel):
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    zoom: float = 1.0


class PerceptionIR(BaseModel):
    obs_id: str
    camera_pose: CameraPose
    objects: List[DetectedObject] = Field(default_factory=list)
    hazards: List[Hazard] = Field(default_factory=list)
    unobserved: List[UnobservedRegion] = Field(default_factory=list)
    audio: List[AudioCue] = Field(default_factory=list)
    vision_description: Optional[str] = None  # VLM による画像分析結果
    modality_errors: List[str] = Field(default_factory=list)  # モダリティ処理エラー（vision/audio など）


class WorldModel(BaseModel):
    # Minimal world model: fused hazards + outstanding unobserved regions
    fused_hazards: List[Hazard] = Field(default_factory=list)
    outstanding_unobserved: List[UnobservedRegion] = Field(default_factory=list)
    last_assessment: Optional["SafetyAssessment"] = None


class SafetyAssessment(BaseModel):
    """LLM による総合安全判断"""
    # 現在の危険状態
    risk_level: Literal["high", "medium", "low"]  # 総合リスク度
    safety_status: str  # 状態説明（例: "フォークリフトが人に接近中"）
    detected_hazards: List[str] = Field(default_factory=list)  # 検出された危険のリスト

    # 行動指示
    action_type: Literal["focus_region", "increase_safety", "continue_observation"]
    target_region: Optional[str] = None  # 注視すべき領域ID（focus_region 時）
    reason: str  # 行動の根拠
    priority: float = Field(ge=0, le=1)  # 行動の優先度（0=低, 1=高）


WorldModel.model_rebuild()  # needed because of forward reference to SafetyAssessment


# =========================
# Observation + Provider
# =========================


@dataclass
class Observation:
    obs_id: str
    image_path: Optional[str] = None
    audio_text: Optional[str] = None
    camera_pose: CameraPose = field(default_factory=CameraPose)
    video_timestamp: Optional[float] = None  # 動画内の秒数


class ObservationProvider:
    """Swap this for: dataset iterator / sensor interface / ROS bridge etc."""

    def __init__(self, observations: List[Observation]):
        self._obs = observations
        self._i = 0

    def next(self) -> Optional[Observation]:
        if self._i >= len(self._obs):
            return None
        o = self._obs[self._i]
        self._i += 1
        return o
