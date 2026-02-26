"""E2E smoke test for Safety View Agent (LLM-free)."""

from src.safety_agent.agent import AgentState, build_agent
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

    perceiver = Perceiver(enable_yolo=False)

    # Build agent
    agent = build_agent()

    # Initial state
    initial_state: AgentState = {
        "messages": [],
        "step": 0,
        "max_steps": 3,
        "observation": None,
        "ir": None,
        "world": WorldModel(),
        "plan": None,
        "selected": None,
        "done": False,
        "errors": [],
    }

    # Context with NO LLM (llm=None forces heuristic fallback)
    context = {
        "provider": provider,
        "perceiver": perceiver,
        "llm": None,
        "risk_stop_threshold": 0.2,
        "hazard_focus_threshold": 0.6,
    }

    # Run agent
    out = agent.invoke(initial_state, context=context)

    # Verify basic output structure
    assert "selected" in out
    assert "world" in out
    assert "errors" in out
    assert "messages" in out

    # Verify selected view is not None
    assert out["selected"] is not None
    assert out["selected"].view_id is not None

    # Verify world model is updated
    assert out["world"] is not None

    print("✅ E2E test passed")
    print(f"Selected view: {out['selected'].view_id}")
    print(f"Messages: {len(out['messages'])}")
    print(f"Errors: {out['errors']}")
