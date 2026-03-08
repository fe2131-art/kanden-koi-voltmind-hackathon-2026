from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# =========================
# Schemas (Pydantic)
# =========================


# ─── VLM 構造化出力モデル ─────────────────────────────────────


class VisionElement(BaseModel):
    label: str
    position: str
    note: Optional[str] = None


class VisionRisk(BaseModel):
    description: str
    position: str
    level: Literal["low", "medium", "high", "unknown"]


class VisionBlindSpot(BaseModel):
    description: str
    position: str


class VisionInspectionFinding(BaseModel):
    description: str
    position: str
    severity: Literal["normal", "minor", "moderate", "critical", "unknown"]


class VisionTemporalChange(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    target: str
    change: Literal["moved", "appeared", "disappeared", "shifted", "state_changed", "unknown"]
    from_state: str = Field(alias="from")  # "from" は Python 予約語のため alias
    to_state: str = Field(alias="to")
    note: Optional[str] = None


class VisionNavigability(BaseModel):
    status: Literal["clear", "partial", "blocked", "unknown"]
    description: str


class VisionOverallAssessment(BaseModel):
    level: Literal["safe", "caution", "dangerous", "unknown"]
    reason: str


class VisionAnalysisResult(BaseModel):
    summary: str
    elements: List[VisionElement] = Field(default_factory=list)
    risks: List[VisionRisk] = Field(default_factory=list)
    blind_spots: List[VisionBlindSpot] = Field(default_factory=list)
    inspection_findings: List[VisionInspectionFinding] = Field(default_factory=list)
    temporal_changes: List[VisionTemporalChange] = Field(default_factory=list)
    navigability: Optional[VisionNavigability] = None
    overall_assessment: Optional[VisionOverallAssessment] = None


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
    """未確認領域（ブラインドスポット）"""
    model_config = ConfigDict(
        exclude_none=False,  # None 値も含める
        # JSON 出力時、内部用フィールドを除外
    )

    region_id: str
    description: str
    risk: float = Field(
        ge=0, le=1
    )  # heuristic risk that something hazardous may be there
    suggested_pan_deg: Optional[float] = Field(
        default=None, exclude=True
    )  # システム内部用（JSON出力から除外）


class Hazard(BaseModel):
    hazard_type: str  # e.g. "forklift", "human_near_machine", "vehicle"
    confidence: float = Field(ge=0, le=1)
    related_objects: List[str] = Field(default_factory=list)
    evidence: Optional[str] = None


class CameraPose(BaseModel):
    pan_deg: Optional[float] = None
    tilt_deg: Optional[float] = None
    zoom: Optional[float] = None


class PerceptionIR(BaseModel):
    obs_id: str
    camera_pose: Optional[CameraPose] = None
    objects: List[DetectedObject] = Field(default_factory=list)
    audio: List[AudioCue] = Field(default_factory=list)
    vision_analysis: Optional[VisionAnalysisResult] = None  # VLM による構造化画像分析結果
    modality_errors: List[str] = Field(default_factory=list)  # モダリティ処理エラー（vision/audio など）


class WorldModel(BaseModel):
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
    prev_image_path: Optional[str] = None  # 前フレームの画像パス（2枚比較用）
    audio_text: Optional[str] = None
    camera_pose: Optional[CameraPose] = None
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
