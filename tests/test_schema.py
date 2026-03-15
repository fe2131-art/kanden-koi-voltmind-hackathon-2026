"""Test schema validation."""

from src.safety_agent.schema import (
    AssessmentEvidence,
    AudioCue,
    CameraPose,
    CriticalPoint,
    DepthAnalysisResult,
    DepthZoneDescription,
    NormalizedBBox,
    Observation,
    ObservationProvider,
    PerceptionIR,
    SafetyAssessment,
    VisionAnalysisResult,
    VisionBlindSpot,
    VisionOverallAssessment,
)


def test_audio_cue():
    cue = AudioCue(
        cue="vehicle_approaching", severity="medium", evidence="車が接近している音"
    )
    assert cue.cue == "vehicle_approaching"
    assert cue.severity == "medium"


def test_camera_pose():
    pose = CameraPose(pan_deg=45.0, tilt_deg=10.0, zoom=1.5)
    assert pose.pan_deg == 45.0
    assert pose.tilt_deg == 10.0


def test_perception_ir():
    ir = PerceptionIR(
        obs_id="t0",
        camera_pose=CameraPose(),
        audio=[],
    )
    assert ir.obs_id == "t0"
    assert len(ir.audio) == 0


def test_safety_assessment():
    assessment = SafetyAssessment(
        risk_level="high",
        safety_status="フォークリフトが人に接近中",
        detected_hazards=["forklift_proximity"],
        action_type="inspect_region",
        target_region="zone_A",
        reason="高リスク未確認領域を優先観測",
        priority=0.85,
        temporal_status="worsening",
        evidence=AssessmentEvidence(
            vision=["フォークリフトが接近中"],
        ),
    )
    assert assessment.risk_level == "high"
    assert assessment.action_type == "inspect_region"
    assert assessment.priority == 0.85
    assert assessment.temporal_status == "worsening"
    assert assessment.evidence is not None
    assert len(assessment.evidence.vision) == 1
    assert assessment.evidence.audio == []


def test_safety_assessment_defaults():
    """temporal_status と evidence のデフォルト値を確認。"""
    assessment = SafetyAssessment(
        risk_level="low",
        safety_status="異常なし",
        action_type="monitor",
        reason="継続監視",
        priority=0.1,
    )
    assert assessment.temporal_status == "unknown"
    assert assessment.evidence is None
    assert assessment.detected_hazards == []


def test_observation():
    obs = Observation(
        obs_id="t0",
        image_path="/path/to/image.jpg",
        prev_image_path="/path/to/prev.jpg",
        audio_path="/path/to/audio.wav",
        audio_text="Some audio",
        camera_pose=CameraPose(pan_deg=0.0),
    )
    assert obs.obs_id == "t0"
    assert obs.image_path == "/path/to/image.jpg"
    assert obs.prev_image_path == "/path/to/prev.jpg"
    assert obs.audio_path == "/path/to/audio.wav"


def test_vision_analysis_result_basic():
    result = VisionAnalysisResult(
        scene_description="工場内の状況。フォークリフト1台と作業員2名が確認。",
        critical_points=[
            CriticalPoint(
                description="フォークリフトが接近中",
                severity="high",
                normalized_bbox=NormalizedBBox(
                    x_min=0.4, y_min=0.3, x_max=0.7, y_max=0.8
                ),
            )
        ],
        blind_spots=[
            VisionBlindSpot(description="棚の背後が見えない", position="右奥")
        ],
        overall_assessment=VisionOverallAssessment(
            severity="high", reason="フォークリフト接近による高リスク"
        ),
    )
    assert result.scene_description.startswith("工場")
    assert len(result.critical_points) == 1
    assert result.critical_points[0].severity == "high"
    assert len(result.blind_spots) == 1


def test_perception_ir_with_vision_analysis():
    """PerceptionIR が vision_analysis フィールドを持つことを確認。"""
    analysis = VisionAnalysisResult(
        scene_description="テスト分析結果",
        overall_assessment=VisionOverallAssessment(severity="low", reason="異常なし"),
    )
    ir = PerceptionIR(obs_id="t0", vision_analysis=analysis)
    assert ir.vision_analysis is not None
    assert ir.vision_analysis.scene_description == "テスト分析結果"


def test_observation_provider():
    obs_list = [
        Observation(obs_id="t0"),
        Observation(obs_id="t1"),
    ]
    provider = ObservationProvider(obs_list)

    o1 = provider.next()
    assert o1.obs_id == "t0"

    o2 = provider.next()
    assert o2.obs_id == "t1"

    o3 = provider.next()
    assert o3 is None


def test_depth_analysis_result():
    """DepthAnalysisResult の基本構築を検証。"""
    depth = DepthAnalysisResult(
        scene_description="近い領域に危険物あり。段階的に奥へ深くなる。",
        depth_layers=[
            DepthZoneDescription(
                zone="near", description="フォークリフト（0.3m）接近中"
            ),
            DepthZoneDescription(zone="mid", description="棚（0.5-2m）"),
            DepthZoneDescription(zone="far", description="壁（2m以上）"),
        ],
    )
    assert depth.scene_description.startswith("近い領域")
    assert len(depth.depth_layers) == 3
    assert depth.depth_layers[0].zone == "near"


def test_perception_ir_with_depth_analysis():
    """PerceptionIR が depth_analysis フィールドを持つことを確認。"""
    depth = DepthAnalysisResult(
        scene_description="テスト深度分析",
        depth_layers=[],
    )
    ir = PerceptionIR(obs_id="t0", depth_analysis=depth)
    assert ir.depth_analysis is not None
    assert ir.depth_analysis.scene_description == "テスト深度分析"


def test_perception_ir_full_modalities():
    """PerceptionIR に vision_analysis と depth_analysis の両方を含めることを確認。"""
    vision = VisionAnalysisResult(
        scene_description="ビジョン分析結果",
        overall_assessment=VisionOverallAssessment(severity="low", reason="OK"),
    )
    depth = DepthAnalysisResult(
        scene_description="深度分析結果",
        depth_layers=[],
    )
    ir = PerceptionIR(
        obs_id="t0",
        vision_analysis=vision,
        depth_analysis=depth,
    )
    assert ir.vision_analysis is not None
    assert ir.depth_analysis is not None
    assert ir.vision_analysis.scene_description == "ビジョン分析結果"
    assert ir.depth_analysis.scene_description == "深度分析結果"
