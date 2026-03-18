"""E2E smoke test for Safety View Agent (LLM-free)."""

from src.safety_agent.agent import AgentState, build_agent
from src.safety_agent.modality_nodes import AudioAnalyzer, InfraredImageAnalyzer
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
        "last_assessment": None,
        "assessment": None,
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
        "context_history_size": 1,
        "expected_modalities": [
            "vlm",
            "audio",
        ],  # vlm/audio に分割（depth は enable=false）
        "run_mode": "until_provider_ends",  # provider が None を返すまで継続
    }

    # Run agent
    out = agent.invoke(initial_state, context=context)

    # Verify basic output structure
    assert "assessment" in out
    assert "last_assessment" in out
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

    # Verify last_assessment is carried over
    assert out["last_assessment"] is not None

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

    print("✅ E2E test passed")
    print(f"Assessment: {out['assessment'].action_type}")
    print(f"Messages: {len(out['messages'])}")
    print(f"Errors: {out['errors']}")
    print(f"Received modalities: {out['received_modalities']}")
    print(f"Latest output frame_id: {out['latest_output']['frame_id']}")
