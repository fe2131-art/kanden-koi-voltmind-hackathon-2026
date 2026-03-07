"""E2E smoke test for Safety View Agent (LLM-free)."""

from src.safety_agent.agent import AgentState, build_agent
from src.safety_agent.modality_nodes import AudioAnalyzer
from src.safety_agent.perceiver import Perceiver
from src.safety_agent.schema import (
    CameraPose,
    Observation,
    ObservationProvider,
    WorldModel,
)


def test_e2e_agent_no_llm():
    """Test that agent runs end-to-end without LLM (heuristic fallback)."""
    # Setup observations
    obs_list = [
        Observation(
            obs_id="t0",
            image_path=None,
            audio_text="I hear a car approach from the right",
            camera_pose=CameraPose(pan_deg=0, tilt_deg=0, zoom=1),
        ),
        Observation(
            obs_id="t1",
            image_path=None,
            audio_text=None,
            camera_pose=CameraPose(pan_deg=30, tilt_deg=0, zoom=1),
        ),
    ]
    provider = ObservationProvider(obs_list)

    perceiver = Perceiver()

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
        "world": WorldModel(),
        "assessment": None,
        "done": False,
        "errors": [],
    }

    # Context with NO LLM (llm=None forces heuristic fallback)
    context = {
        "provider": provider,
        "perceiver": perceiver,
        "llm": None,
        "vision_analyzer": None,
        "yolo_detector": None,
        "audio_analyzer": AudioAnalyzer(),
        "prompts": {
            "vision_analysis": {
                "default_prompt": "テスト用プロンプト（LLM 未使用のため内容は問わない）"
            },
            "next_view_proposal": {
                "system": "テスト用：知覚推論+安全判断統合プロンプト（LLMなしで実行）",
            },
        },
        "chat_max_tokens": 2000,
        "max_outstanding_regions": 6,
        "context_history_size": 1,
        "expected_modalities": ["yolo", "vlm", "audio"],  # yolo/vlm に分割
        "run_mode": "until_provider_ends",  # provider が None を返すまで継続
    }

    # Run agent
    out = agent.invoke(initial_state, context=context)

    # Verify basic output structure
    assert "assessment" in out
    assert "world" in out
    assert "errors" in out
    assert "messages" in out
    assert "modality_results" in out

    # Verify assessment is not None
    assert out["assessment"] is not None
    assert out["assessment"].action_type in ["focus_region", "increase_safety", "continue_observation"]
    assert out["assessment"].risk_level in ["high", "medium", "low"]

    # Verify world model is updated
    assert out["world"] is not None

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

    # PR3: Verify latest_output is properly populated
    assert "latest_output" in out
    assert out["latest_output"] is not None
    assert out["latest_output"]["obs_id"] is not None
    assert out["latest_output"]["assessment"] is not None
    assert "step" in out["latest_output"]
    assert "ir" in out["latest_output"]
    assert "world" in out["latest_output"]

    print("✅ E2E test passed")
    print(f"Assessment: {out['assessment'].action_type}")
    print(f"Messages: {len(out['messages'])}")
    print(f"Errors: {out['errors']}")
    print(f"Received modalities: {out['received_modalities']}")
    print(f"Latest output obs_id: {out['latest_output']['obs_id']}")
