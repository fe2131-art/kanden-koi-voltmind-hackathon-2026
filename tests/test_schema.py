"""Test schema validation."""

from src.safety_agent.schema import (
    AudioCue,
    BoundingBox,
    CameraPose,
    DetectedObject,
    Hazard,
    SafetyAssessment,
    Observation,
    ObservationProvider,
    PerceptionIR,
    UnobservedRegion,
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


def test_unobserved_region():
    region = UnobservedRegion(
        region_id="blind_left", description="Left blind spot", risk=0.4
    )
    assert region.region_id == "blind_left"
    assert region.risk == 0.4


def test_hazard():
    hazard = Hazard(
        hazard_type="human_present", confidence=0.6, related_objects=["person"]
    )
    assert hazard.hazard_type == "human_present"
    assert len(hazard.related_objects) == 1


def test_camera_pose():
    pose = CameraPose(pan_deg=45.0, tilt_deg=10.0, zoom=1.5)
    assert pose.pan_deg == 45.0
    assert pose.tilt_deg == 10.0


def test_perception_ir():
    ir = PerceptionIR(
        obs_id="t0",
        camera_pose=CameraPose(),
        objects=[],
        hazards=[],
        unobserved=[],
        audio=[],
    )
    assert ir.obs_id == "t0"
    assert len(ir.objects) == 0


def test_world_model():
    world = WorldModel()
    assert len(world.fused_hazards) == 0
    assert len(world.outstanding_unobserved) == 0
    assert world.last_assessment is None


def test_safety_assessment():
    assessment = SafetyAssessment(
        risk_level="high",
        safety_status="フォークリフトが人に接近中",
        detected_hazards=["forklift_proximity"],
        action_type="focus_region",
        target_region="zone_A",
        reason="高リスク未確認領域を優先観測",
        priority=0.85,
    )
    assert assessment.risk_level == "high"
    assert assessment.action_type == "focus_region"
    assert assessment.priority == 0.85


def test_observation():
    obs = Observation(
        obs_id="t0",
        image_path="/path/to/image.jpg",
        audio_text="Some audio",
        camera_pose=CameraPose(pan_deg=0.0),
    )
    assert obs.obs_id == "t0"
    assert obs.image_path == "/path/to/image.jpg"


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
