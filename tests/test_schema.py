"""Test schema validation."""

from src.safety_agent.schema import (
    AssessmentEvidence,
    AudioCue,
    BoundingBox,
    CameraPose,
    DetectedObject,
    Observation,
    ObservationProvider,
    PerceptionIR,
    SafetyAssessment,
    VisionAnalysisResult,
    VisionBlindSpot,
    VisionOverallAssessment,
    VisionRisk,
    VisionTemporalChange,
    WorldModel,
)


def test_bounding_box():
    bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9)
    assert bbox.x1 == 0.1
    assert bbox.y2 == 0.9


def test_detected_object():
    bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9)
    obj = DetectedObject(label="person", confidence=0.95, bbox=bbox)
    assert obj.label == "person"
    assert obj.confidence == 0.95


def test_audio_cue():
    cue = AudioCue(
        cue="vehicle_approaching", confidence=0.7, direction="right", evidence="sound"
    )
    assert cue.cue == "vehicle_approaching"
    assert cue.direction == "right"


def test_camera_pose():
    pose = CameraPose(pan_deg=45.0, tilt_deg=10.0, zoom=1.5)
    assert pose.pan_deg == 45.0
    assert pose.tilt_deg == 10.0


def test_perception_ir():
    ir = PerceptionIR(
        obs_id="t0",
        camera_pose=CameraPose(),
        objects=[],
        audio=[],
    )
    assert ir.obs_id == "t0"
    assert len(ir.objects) == 0


def test_world_model():
    world = WorldModel()
    assert world.last_assessment is None


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
            yolo=["forklift confidence=0.92"],
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
        audio_text="Some audio",
        camera_pose=CameraPose(pan_deg=0.0),
    )
    assert obs.obs_id == "t0"
    assert obs.image_path == "/path/to/image.jpg"
    assert obs.prev_image_path == "/path/to/prev.jpg"


def test_vision_analysis_result_basic():
    result = VisionAnalysisResult(
        summary="工場内の状況。フォークリフト1台と作業員2名が確認。",
        risks=[VisionRisk(description="フォークリフトが接近中", position="中央右", level="high")],
        blind_spots=[VisionBlindSpot(description="棚の背後が見えない", position="右奥")],
    )
    assert result.summary.startswith("工場")
    assert len(result.risks) == 1
    assert result.risks[0].level == "high"
    assert len(result.blind_spots) == 1
    assert result.navigability is None
    assert result.overall_assessment is None


def test_vision_temporal_change_alias():
    """VisionTemporalChange の from/to エイリアスが正しく動作することを確認。"""
    change = VisionTemporalChange.model_validate({
        "target": "forklift",
        "change": "moved",
        "from": "左端",
        "to": "中央",
        "note": "約2m移動",
    })
    assert change.from_state == "左端"
    assert change.to_state == "中央"

    # by_alias=True でシリアライズすると "from"/"to" キーが出力される
    dumped = change.model_dump(by_alias=True, exclude_none=True)
    assert "from" in dumped
    assert "to" in dumped
    assert dumped["from"] == "左端"


def test_vision_analysis_result_full():
    """VisionAnalysisResult が overall_assessment を含む場合の構築を検証。"""
    result = VisionAnalysisResult(
        summary="安全な状態",
        overall_assessment=VisionOverallAssessment(level="safe", reason="危険物なし"),
    )
    dumped = result.model_dump(exclude_none=True)
    assert dumped["overall_assessment"]["level"] == "safe"


def test_perception_ir_with_vision_analysis():
    """PerceptionIR が vision_analysis フィールドを持つことを確認。"""
    analysis = VisionAnalysisResult(summary="テスト分析結果")
    ir = PerceptionIR(obs_id="t0", vision_analysis=analysis)
    assert ir.vision_analysis is not None
    assert ir.vision_analysis.summary == "テスト分析結果"

    assert not hasattr(ir, "vision_description")


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
