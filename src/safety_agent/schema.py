from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

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
#  │    agent.py: determine_next_action_llm()                               │
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


class NormalizedBBox(BaseModel):
    """正規化された bounding box（0.0-1.0 範囲）。"""

    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)


class CriticalPoint(BaseModel):
    """画像中の危険箇所。位置を特定できる重要な危険。"""

    region_id: str
    description: str
    normalized_bbox: Optional[NormalizedBBox] = None
    severity: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"


class VisionBlindSpot(BaseModel):
    """画像中の死角・見えにくい場所。"""

    region_id: str
    description: str
    position: str
    severity: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"


class VisionAnalysisResult(BaseModel):
    """VisionAnalyzer.analyze() の構造化出力。PerceptionIR.vision_analysis に格納。"""

    scene_description: str
    critical_points: List[CriticalPoint] = Field(default_factory=list)
    blind_spots: List[VisionBlindSpot] = Field(default_factory=list)
    overall_risk: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


# ─── 音声キュー ───────────────────────────────────────────────────────────────
# AudioAnalyzer.analyze() の出力。PerceptionIR.audio に格納。


class AudioCue(BaseModel):
    """音声解析から抽出した危険キュー。PerceptionIR.audio の要素。"""

    cue: str  # e.g. "vehicle_approaching"
    severity: Literal["low", "medium", "high", "critical", "unknown"]
    evidence: str  # 根拠説明


class AudioAnalysisResult(BaseModel):
    """音声解析ノードの出力スキーマ。prompt.yaml audio_analysis の出力形式に対応。"""

    events: List[AudioCue] = Field(default_factory=list)
    overall_risk: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


# ─── 深度解析結果 ──────────────────────────────────────────────────────────────
# DepthEstimator.estimate() + VisionAnalyzer.analyze_bytes_raw() の出力。
# PerceptionIR.depth_analysis に格納。


class DepthZoneDescription(BaseModel):
    """深度層別の領域説明。"""

    zone: Literal["near", "mid", "far"]
    description: str


class DepthAnalysisResult(BaseModel):
    """Depth Anything 3 による深度解析結果。PerceptionIR.depth_analysis に格納。"""

    depth_layers: List[DepthZoneDescription] = Field(default_factory=list)
    overall_risk: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class InfraredHotSpot(BaseModel):
    """赤外線画像中の高温箇所。"""

    region_id: str
    description: str
    severity: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"


class InfraredAnalysisResult(BaseModel):
    """赤外線画像解析結果。PerceptionIR.infrared_analysis に格納。"""

    hot_spots: List[InfraredHotSpot] = Field(default_factory=list)
    overall_risk: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class TemporalChange(BaseModel):
    """時系列で検出された変化。"""

    region_id: str
    description: str
    severity: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"


class TemporalAnalysisResult(BaseModel):
    """時系列（前後フレーム比較）変化検出結果。PerceptionIR.temporal_analysis に格納。"""

    change_detected: bool  # 明らかな変化があったか（VLM が true/false を返す）
    changes: List[TemporalChange] = Field(
        default_factory=list,
        description="実際に確認できた変化の具体的説明リスト。変化がなければ空配列 []",
    )
    overall_risk: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


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
    """1フレーム分の知覚統合結果（VLM + 音声 + 深度 + 赤外線 + 時系列変化）。AgentState.ir に格納。"""

    obs_id: str
    camera_pose: Optional[CameraPose] = None
    audio: List[AudioCue] = Field(default_factory=list)  # 音声出力
    vision_analysis: Optional[VisionAnalysisResult] = None  # VLM 出力
    depth_analysis: Optional[DepthAnalysisResult] = None  # 深度解析出力
    infrared_analysis: Optional[InfraredAnalysisResult] = None  # 赤外線画像解析出力
    temporal_analysis: Optional[TemporalAnalysisResult] = None  # 時系列変化検出出力
    modality_errors: List[str] = Field(default_factory=list)  # 各モダリティのエラー


# ─── 信念状態（Belief State） ─────────────────────────────────────────────────
# update_belief_state_llm() が生成し、AgentState.belief_state に格納される。
# フレーム間の危険状態の継続・悪化・改善・解消を LLM が管理する。


class HazardTrack(BaseModel):
    """継続中の個別危険状態。BeliefState.hazard_tracks の要素。"""

    hazard_id: str
    hazard_type: Literal[
        "visible_hazard",
        "blind_spot",
        "overheat",
        "abnormal_sound",
        "scene_change",
        "obstacle",
        "equipment_anomaly",
        "unknown",
    ]
    region_id: Optional[str] = None
    status: Literal[
        "new", "persistent", "worsening", "improving", "resolved", "unknown"
    ]
    severity: Literal["low", "medium", "high", "critical", "unknown"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    supporting_modalities: List[
        Literal["vision", "audio", "depth", "infrared", "temporal"]
    ] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class BeliefState(BaseModel):
    """LLM が管理する時系列危険状態の内部表現。AgentState.belief_state に格納。

    hazard_tracks: 継続中の危険状態のリスト（フレームをまたいで更新される）
    overall_risk:  全体リスクレベル（hazard_tracks から総合判断）
    recommended_focus_regions: 次フレームで注視すべき領域
    """

    hazard_tracks: List[HazardTrack] = Field(default_factory=list)
    overall_risk: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"
    recommended_focus_regions: List[str] = Field(default_factory=list)


# ─── 安全判断 ─────────────────────────────────────────────────────────────────
# determine_next_action_llm() が生成し、
# AgentState.assessment と WorldModel.last_assessment に格納される。


class SafetyAssessment(BaseModel):
    """LLM による総合安全判断。AgentState.assessment に格納。

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
    target_region: Optional[str] = (
        None  # 常にキー出力。inspect_region 時のみ region_id、それ以外は null
    )
    reason: str
    priority: float = Field(ge=0, le=1)

    # 時系列・信頼度
    temporal_status: Literal[
        "new", "persistent", "worsening", "improving", "resolved", "unknown"
    ] = "unknown"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


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
    audio_path: Optional[str] = None  # 音声ファイルパス（推奨）
    audio_text: Optional[str] = None  # 音声文字起こしやテキスト入力（後方互換）
    infrared_image_path: Optional[str] = None  # 赤外線フレームパス
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


# =============================================================================
# vLLM Structured Outputs 用 JSON スキーマ生成
# =============================================================================


# スキーママップ：型名 → Pydantic モデルクラス
_SCHEMA_MAP: Dict[str, type] = {
    "vision_analysis": VisionAnalysisResult,
    "depth_analysis": DepthAnalysisResult,
    "infrared_analysis": InfraredAnalysisResult,
    "temporal_analysis": TemporalAnalysisResult,
    "audio_analysis": AudioAnalysisResult,
    "belief_state": BeliefState,
    "safety_assessment": SafetyAssessment,
}

SchemaType = Literal[
    "vision_analysis",
    "depth_analysis",
    "infrared_analysis",
    "temporal_analysis",
    "audio_analysis",
    "belief_state",
    "safety_assessment",
]


def get_json_schema(schema_type: SchemaType) -> Dict[str, Any]:
    """指定されたスキーマ型の JSON Schema を返す。

    Pydantic モデルの model_json_schema() で自動生成するため、
    モデル定義の変更時に自動追従する。

    Args:
        schema_type: スキーマ型名

    Returns:
        vLLM Structured Outputs で使用可能な JSON Schema 辞書

    Raises:
        ValueError: unknown schema_type
    """
    if schema_type not in _SCHEMA_MAP:
        raise ValueError(
            f"Unknown schema_type: {schema_type}. "
            f"Allowed values: {list(_SCHEMA_MAP.keys())}"
        )
    schema = _SCHEMA_MAP[schema_type].model_json_schema()

    # safety_assessment の target_region を required に強制追加
    # vLLM の guided decoding で target_region キーが必ず出力されるようにする
    if schema_type == "safety_assessment":
        if "required" in schema and "target_region" not in schema["required"]:
            schema["required"].append("target_region")

    return schema
