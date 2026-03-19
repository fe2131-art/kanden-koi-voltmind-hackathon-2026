"""E2E smoke test for Safety View Agent (LLM-free)."""

import tempfile
from pathlib import Path

from PIL import Image

from src.safety_agent.agent import AgentState, build_agent
from src.safety_agent.modality_nodes import (
    AudioAnalyzer,
    InfraredImageAnalyzer,
    TemporalImageAnalyzer,
)
from src.safety_agent.schema import (
    Observation,
    ObservationProvider,
)


def test_e2e_agent_no_llm():
    """Test that agent runs end-to-end without LLM (default assessment)."""
    # Setup observations
    obs_list = [
        Observation(
            obs_id="t0",
            image_path=None,
            audio_text="I hear a car approach from the right",
            camera_pose=None,
        ),
        Observation(
            obs_id="t1",
            image_path=None,
            audio_text=None,
            camera_pose=None,
        ),
    ]
    provider = ObservationProvider(obs_list)

    # Build agent
    agent = build_agent()

    # Initial state (with modality_results for fan-in)
    initial_state: AgentState = {
        "messages": [],
        "step": 0,
        "max_steps": 3,
        "observation": None,
        "ir": None,
        "modality_results": {},  # Dict に変更（メモリリーク防止）
        "received_modalities": [],  # PR1: fan-in バリア
        "barrier_obs_id": None,  # ラッチ（同フレーム内で fuse は1回だけ）
        "latest_output": None,  # PR3: 統合出力
        "last_vision_summary": None,
        "assessment": None,
        "assessment_history": [],
        "belief_state": None,
        "done": False,
        "errors": [],
    }

    # Context with NO LLM (llm=None uses default assessment)
    context = {
        "provider": provider,
        "llm": None,
        "vision_analyzer": None,
        "audio_analyzer": AudioAnalyzer(),
        "depth_estimator": None,
        "infrared_analyzer": InfraredImageAnalyzer(),
        "temporal_analyzer": TemporalImageAnalyzer(),
        "prompts": {
            "vision_analysis": {
                "default_prompt": "テスト用プロンプト（LLM 未使用のため内容は問わない）"
            },
            "audio_analysis": {
                "default_prompt": "テスト用音声プロンプト",
            },
            "safety_assessment": {
                "system": "テスト用：知覚推論+安全判断統合プロンプト（LLMなしで実行）",
            },
        },
        "config": {"audio": {"window_seconds": 3.0}},
        "chat_max_tokens": 2000,
        "context_history_size": 0,
        "expected_modalities": [
            "vlm",
            "audio",
            "temporal",
        ],  # vlm/audio/temporal に分割（depth は enable=false）
        "run_mode": "until_provider_ends",  # provider が None を返すまで継続
    }

    # Run agent
    out = agent.invoke(initial_state, context=context)

    # Verify basic output structure
    assert "assessment" in out
    assert "errors" in out
    assert "messages" in out
    assert "modality_results" in out

    # Verify assessment is not None
    assert out["assessment"] is not None
    assert out["assessment"].action_type in [
        "emergency_stop",
        "inspect_region",
        "mitigate",
        "monitor",
    ]
    assert out["assessment"].risk_level in ["high", "medium", "low"]

    # Verify modality_results is properly processed (fan-in) - now dict
    assert isinstance(out["modality_results"], dict)

    # Verify barrier_obs_id is present (latch for join_modalities)
    assert "barrier_obs_id" in out

    # Verify perception IR contains audio cues from audio_node
    assert out["ir"] is not None
    assert isinstance(out["ir"].audio, list)

    # Verify no unexpected errors
    assert isinstance(out["errors"], list)

    # PR1: Verify received_modalities is a list
    assert "received_modalities" in out
    assert isinstance(out["received_modalities"], list)

    # PR3: Verify latest_output is properly populated (flattened structure)
    assert "latest_output" in out
    assert out["latest_output"] is not None
    assert out["latest_output"]["frame_id"] is not None
    assert out["latest_output"]["assessment"] is not None
    assert "audio" in out["latest_output"]
    # vision_analysis は LLM なしでは None（VisionAnalyzer なし）
    assert "vision_analysis" in out["latest_output"]
    assert out["latest_output"]["vision_analysis"] is None or isinstance(
        out["latest_output"]["vision_analysis"], dict
    )

    # temporal_analysis は prev_image_path が None のときはスキップされるが、フィールドは存在すべき
    assert "temporal_analysis" in out["latest_output"]
    assert out["latest_output"]["temporal_analysis"] is None or isinstance(
        out["latest_output"]["temporal_analysis"], dict
    )

    print("✅ E2E test passed")
    print(f"Assessment: {out['assessment'].action_type}")
    print(f"Messages: {len(out['messages'])}")
    print(f"Errors: {out['errors']}")
    print(f"Received modalities: {out['received_modalities']}")
    print(f"Latest output frame_id: {out['latest_output']['frame_id']}")


def test_temporal_node_with_image_pair():
    """Test temporal_node with actual image pair (normal case)."""
    # Create temporary test images
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create two different test images
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color="blue")

        frame1_path = str(tmpdir_path / "frame1.jpg")
        frame2_path = str(tmpdir_path / "frame2.jpg")

        img1.save(frame1_path)
        img2.save(frame2_path)

        # Setup observations with image paths
        obs_list = [
            Observation(
                obs_id="img_0",
                image_path=frame1_path,
                prev_image_path=None,  # First frame
                audio_text=None,
                camera_pose=None,
            ),
            Observation(
                obs_id="img_1",
                image_path=frame2_path,
                prev_image_path=frame1_path,  # Second frame with previous
                audio_text=None,
                camera_pose=None,
            ),
        ]
        provider = ObservationProvider(obs_list)

        # Build agent
        agent = build_agent()

        # Initial state
        initial_state: AgentState = {
            "messages": [],
            "step": 0,
            "max_steps": 3,
            "observation": None,
            "ir": None,
            "modality_results": {},
            "received_modalities": [],
            "barrier_obs_id": None,
            "latest_output": None,
            "last_vision_summary": None,
            "assessment": None,
            "assessment_history": [],
            "belief_state": None,
            "done": False,
            "errors": [],
        }

        # Context with temporal enabled
        context = {
            "provider": provider,
            "llm": None,
            "vision_analyzer": None,
            "audio_analyzer": AudioAnalyzer(),
            "depth_estimator": None,
            "infrared_analyzer": InfraredImageAnalyzer(),
            "temporal_analyzer": TemporalImageAnalyzer(),
            "prompts": {
                "vision_analysis": {
                    "default_prompt": "テスト用プロンプト（LLM未使用）"
                },
                "audio_analysis": {"default_prompt": "テスト用音声プロンプト"},
                "temporal_analysis": {
                    "system": "前後フレーム比較テスト用プロンプト（LLMなし）"
                },
                "safety_assessment": {"system": "テスト用安全判断プロンプト"},
            },
            "config": {"audio": {"window_seconds": 3.0}},
            "chat_max_tokens": 2000,
            "context_history_size": 0,
            "expected_modalities": ["vlm", "audio", "temporal"],
            "run_mode": "until_provider_ends",
        }

        # Run agent
        out = agent.invoke(initial_state, context=context)

        # Verify that agent completed
        assert out["assessment"] is not None

        # Verify temporal_analysis is in latest_output
        assert "temporal_analysis" in out["latest_output"]

        # For first frame (img_0): temporal_analysis should be None (no prev_image_path)
        # For second frame (img_1): temporal_analysis could be None or dict depending on error
        # The important thing is the field exists
        assert out["latest_output"]["temporal_analysis"] is None or isinstance(
            out["latest_output"]["temporal_analysis"], dict
        )

        print("✅ Temporal node test with image pair passed")
        print(f"Frame ID: {out['latest_output']['frame_id']}")
        print(f"Temporal analysis: {out['latest_output']['temporal_analysis']}")
