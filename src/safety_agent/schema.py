from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Pydantic スキーマ定義
#
# 使用箇所マップ:
#
#  ┌─ VLM 出力 ──────────────────────────────────────────────────────────────┐
#  │  VisionElement          VisionAnalysisResult.elements                  │
#  │  VisionRisk             VisionAnalysisResult.risks                     │
#  │  VisionBlindSpot        VisionAnalysisResult.blind_spots               │
#  │  VisionInspectionFinding VisionAnalysisResult.inspection_findings      │
#  │  VisionTemporalChange   VisionAnalysisResult.temporal_changes          │
#  │  VisionNavigability     VisionAnalysisResult.navigability              │
#  │  VisionOverallAssessment VisionAnalysisResult.overall_assessment       │
#  │  VisionAnalysisResult   PerceptionIR.vision_analysis                   │
#  │    modality_nodes.py: VisionAnalyzer.analyze() の返却値               │
#  │    agent.py: vlm_node() → PerceptionIR                                 │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  ┌─ YOLO 出力 ─────────────────────────────────────────────────────────────┐
#  │  BoundingBox            DetectedObject.bbox                             │
#  │  DetectedObject         PerceptionIR.objects                            │
#  │    modality_nodes.py: YOLODetector.detect() の返却値                   │
#  │    agent.py: yolo_node() → PerceptionIR, determine_next_action_llm()   │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  ┌─ 音声出力 ──────────────────────────────────────────────────────────────┐
#  │  AudioCue               PerceptionIR.audio                             │
#  │    modality_nodes.py: AudioAnalyzer.analyze() の返却値                 │
#  │    agent.py: audio_node() → PerceptionIR                               │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  ┌─ 知覚統合 ──────────────────────────────────────────────────────────────┐
#  │  CameraPose             PerceptionIR.camera_pose / Observation.camera_pose │
#  │  PerceptionIR           AgentState.ir （フレームごとの知覚統合結果）    │
#  │    agent.py: join_modalities() → fuse_modalities() で組み立て          │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  ┌─ 安全判断 ──────────────────────────────────────────────────────────────┐
#  │  AssessmentEvidence     SafetyAssessment.evidence                      │
#  │  SafetyAssessment       AgentState.assessment / WorldModel.last_assessment │
#  │    agent.py: determine_next_action_llm() / _heuristic_assessment()     │
#  │    agent.py: _get_json_schema_for_vllm() で vLLM 構造化出力スキーマ定義 │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  ┌─ 状態管理 ──────────────────────────────────────────────────────────────┐
#  │  last_assessment        AgentState.last_assessment （前フレーム引き継ぎ） │
#  │    agent.py: determine_next_action_llm() で更新                        │
#  │    run.py: 初期化時に None を設定                                      │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  ┌─ 観測入力 ──────────────────────────────────────────────────────────────┐
#  │  Observation            AgentState.observation （入力データ単位）       │
#  │    run.py: prepare_observations() でリスト生成                         │
#  │  ObservationProvider    agent.py: ingest_observation() が next() を呼ぶ │
#  │    run.py: 初期化時に作成、context に渡す                              │
#  └─────────────────────────────────────────────────────────────────────────┘
# =============================================================================


# ─── VLM 構造化出力モデル ─────────────────────────────────────────────────────
# VisionAnalyzer.analyze() の JSON 出力を Pydantic で型付け。
# PerceptionIR.vision_analysis に格納される。


class VisionElement(BaseModel):
    """画像中に見えている要素（人・物・設備など）。"""
    label: str
    position: str
    note: Optional[str] = None


class VisionRisk(BaseModel):
    """画像から検出された危険の内容。"""
    description: str
    position: str
    level: Literal["low", "medium", "high", "unknown"]


class VisionBlindSpot(BaseModel):
    """画像中の死角・見えにくい場所。"""
    description: str
    position: str


class VisionInspectionFinding(BaseModel):
    """設備・施設の異常または点検上の所見。"""
    description: str
    position: str
    severity: Literal["normal", "minor", "moderate", "critical", "unknown"]


class VisionTemporalChange(BaseModel):
    """2枚の画像間での変化（移動・出現・消失など）。"""
    model_config = ConfigDict(populate_by_name=True)

    target: str
    change: Literal["moved", "appeared", "disappeared", "shifted", "state_changed", "unknown"]
    from_state: str = Field(alias="from")  # "from" は Python 予約語のため alias
    to_state: str = Field(alias="to")
    note: Optional[str] = None


class VisionNavigability(BaseModel):
    """通路・移動経路の通行しやすさ。"""
    status: Literal["clear", "partial", "blocked", "unknown"]
    description: str


class VisionOverallAssessment(BaseModel):
    """画像全体の安全レベル評価。"""
    level: Literal["safe", "caution", "dangerous", "unknown"]
    reason: str


class VisionAnalysisResult(BaseModel):
    """VisionAnalyzer.analyze() の構造化出力。PerceptionIR.vision_analysis に格納。"""
    summary: str
    elements: List[VisionElement] = Field(default_factory=list)
    risks: List[VisionRisk] = Field(default_factory=list)
    blind_spots: List[VisionBlindSpot] = Field(default_factory=list)
    inspection_findings: List[VisionInspectionFinding] = Field(default_factory=list)
    temporal_changes: List[VisionTemporalChange] = Field(default_factory=list)
    navigability: Optional[VisionNavigability] = None
    overall_assessment: Optional[VisionOverallAssessment] = None


# ─── YOLO 検出結果 ────────────────────────────────────────────────────────────
# YOLODetector.detect() の出力。PerceptionIR.objects に格納。


class BoundingBox(BaseModel):
    """YOLO 検出ボックスの座標（正規化）。"""
    x1: float
    y1: float
    x2: float
    y2: float


class DetectedObject(BaseModel):
    """YOLO で検出された1オブジェクト。PerceptionIR.objects の要素。"""
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: Optional[BoundingBox] = None


# ─── 音声キュー ───────────────────────────────────────────────────────────────
# AudioAnalyzer.analyze() の出力。PerceptionIR.audio に格納。


class AudioCue(BaseModel):
    """音声解析から抽出した危険キュー。PerceptionIR.audio の要素。"""
    cue: str  # e.g. "vehicle_approaching"
    confidence: float = Field(ge=0, le=1)
    direction: Optional[Literal["left", "right"]] = None
    evidence: Optional[str] = None


# ─── 深度解析結果 ──────────────────────────────────────────────────────────────
# DepthEstimator.estimate() + VisionAnalyzer.analyze_bytes_raw() の出力。
# PerceptionIR.depth_analysis に格納。


class DepthAnalysisResult(BaseModel):
    """Depth Anything 3 による深度解析結果。PerceptionIR.depth_analysis に格納。"""
    summary: str
    depth_zones: List[str] = Field(default_factory=list)  # 深度層別の領域説明
    nearest_hazards: List[str] = Field(default_factory=list)  # 最も近い危険物
    occlusions: List[str] = Field(default_factory=list)  # 隠れた領域・死角
    spatial_layout: Optional[str] = None  # 空間的配置の整体的説明


# ─── カメラ姿勢 ───────────────────────────────────────────────────────────────
# Observation と PerceptionIR に付属するメタ情報。現在は値が入らないことが多い。


class CameraPose(BaseModel):
    """カメラの物理姿勢（pan/tilt/zoom）。Optional フィールドで値なしも可。"""
    pan_deg: Optional[float] = None
    tilt_deg: Optional[float] = None
    zoom: Optional[float] = None


# ─── 知覚統合結果 ─────────────────────────────────────────────────────────────
# agent.py の join_modalities() → fuse_modalities() で組み立てられ、
# AgentState.ir に格納される。フレームごとに作成・破棄される。


class PerceptionIR(BaseModel):
    """1フレーム分の知覚統合結果（YOLO + VLM + 音声 + 深度）。AgentState.ir に格納。"""
    obs_id: str
    camera_pose: Optional[CameraPose] = None
    objects: List[DetectedObject] = Field(default_factory=list)       # YOLO 出力
    audio: List[AudioCue] = Field(default_factory=list)               # 音声出力
    vision_analysis: Optional[VisionAnalysisResult] = None            # VLM 出力
    depth_analysis: Optional[DepthAnalysisResult] = None              # 深度解析出力
    modality_errors: List[str] = Field(default_factory=list)          # 各モダリティのエラー


# ─── 安全判断 ─────────────────────────────────────────────────────────────────
# determine_next_action_llm() または _heuristic_assessment() が生成し、
# AgentState.assessment と WorldModel.last_assessment に格納される。


class AssessmentEvidence(BaseModel):
    """SafetyAssessment の判断根拠（モダリティ別）。SafetyAssessment.evidence に格納。"""
    vision: List[str] = Field(default_factory=list)    # VLM 分析から使った根拠
    yolo: List[str] = Field(default_factory=list)      # YOLO 検出から使った根拠
    audio: List[str] = Field(default_factory=list)     # 音声キューから使った根拠
    previous: List[str] = Field(default_factory=list)  # 前回判断から参照した根拠


class SafetyAssessment(BaseModel):
    """LLM または heuristic による総合安全判断。AgentState.assessment に格納。

    action_type の意味:
      - "emergency_stop"  : 即時停止・退避が必要
      - "inspect_region"  : 特定領域の重点確認が必要（target_region 必須）
      - "mitigate"        : 安全対策の実施・強化が必要
      - "monitor"         : 継続監視でよい

    temporal_status の意味:
      - "new"        : 今回初めて検出
      - "persistent" : 前回から継続
      - "worsening"  : 前回から悪化
      - "improving"  : 前回から改善
      - "resolved"   : 解消された
      - "unknown"    : 判断不可（初回またはヒューリスティック）
    """
    # 現在の危険状態
    risk_level: Literal["high", "medium", "low"]
    safety_status: str
    detected_hazards: List[str] = Field(default_factory=list)

    # 行動指示
    action_type: Literal["emergency_stop", "inspect_region", "mitigate", "monitor"]
    target_region: Optional[str] = None  # inspect_region 時のみ設定
    reason: str
    priority: float = Field(ge=0, le=1)

    # 時系列・根拠
    temporal_status: Literal[
        "new", "persistent", "worsening", "improving", "resolved", "unknown"
    ] = "unknown"
    evidence: Optional[AssessmentEvidence] = None




# =============================================================================
# 観測入力
# run.py の prepare_observations() でリスト生成 → ObservationProvider に渡される。
# agent.py の ingest_observation() が provider.next() を呼び AgentState.observation に設定。
# =============================================================================


@dataclass
class Observation:
    """1フレーム分の入力データ。run.py で生成され agent に渡される。"""
    obs_id: str
    image_path: Optional[str] = None
    prev_image_path: Optional[str] = None  # 前フレームの画像パス（2枚比較用）
    audio_path: Optional[str] = None       # 音声ファイルパス（推奨）
    audio_text: Optional[str] = None       # 音声文字起こしやテキスト入力（後方互換）
    camera_pose: Optional[CameraPose] = None
    video_timestamp: Optional[float] = None  # 動画内の秒数


class ObservationProvider:
    """観測データのイテレータ。dataset / sensor / ROS bridge などに差し替え可能。

    使用箇所: agent.py の ingest_observation() が next() を呼ぶ。
    初期化: run.py で ObservationProvider(obs_list) を作成し context に渡す。
    """

    def __init__(self, observations: List[Observation]):
        self._obs = observations
        self._i = 0

    def next(self) -> Optional[Observation]:
        if self._i >= len(self._obs):
            return None
        o = self._obs[self._i]
        self._i += 1
        return o
