"""Test configuration loading and LLM initialization."""

import os


def test_config_loading():
    """Test YAML configuration loading."""
    from run import load_config

    config = load_config("configs/default.yaml")

    # Basic structure checks
    assert "llm" in config
    assert "agent" in config
    assert "thresholds" in config

    # LLM provider check
    llm_cfg = config.get("llm", {})
    assert "provider" in llm_cfg
    assert llm_cfg["provider"] in ["openai", "vllm"]


def test_llm_initialization_openai_missing_key():
    """Test OpenAI initialization without API key (should return None)."""

    from run import get_llm

    # Create temporary config with openai provider
    config = {
        "llm": {
            "provider": "openai",
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o",
            },
        }
    }

    # Ensure API key is not set
    os.environ.pop("OPENAI_API_KEY", None)

    # Since env var is not set, should return None
    llm = get_llm(config)
    assert llm is None


def test_config_vllm_defaults():
    """Test that vLLM configuration has proper defaults."""
    from run import load_config

    config = load_config("configs/default.yaml")
    vllm_cfg = config.get("llm", {}).get("vllm", {})

    # Check defaults
    assert "base_url" in vllm_cfg
    assert "model" in vllm_cfg
    assert "api_key" in vllm_cfg


def test_config_openai_defaults():
    """Test that OpenAI configuration has proper defaults."""
    from run import load_config

    config = load_config("configs/default.yaml")
    openai_cfg = config.get("llm", {}).get("openai", {})

    # Check defaults
    assert "base_url" in openai_cfg
    assert "model" in openai_cfg
    assert openai_cfg["base_url"] == "https://api.openai.com/v1"


def test_thresholds_config():
    """Test threshold configuration."""
    from run import load_config

    config = load_config("configs/default.yaml")
    thresholds_cfg = config.get("thresholds", {})

    # Check thresholds
    assert "risk_stop_threshold" in thresholds_cfg
    assert "hazard_focus_threshold" in thresholds_cfg
    assert 0 <= thresholds_cfg["risk_stop_threshold"] <= 1
    assert 0 <= thresholds_cfg["hazard_focus_threshold"] <= 1


def test_agent_config():
    """Test agent configuration."""
    from run import load_config

    config = load_config("configs/default.yaml")
    agent_cfg = config.get("agent", {})

    # Check agent config
    assert "max_steps" in agent_cfg
    assert "enable_yolo" in agent_cfg
    assert isinstance(agent_cfg["max_steps"], int)
    assert isinstance(agent_cfg["enable_yolo"], bool)


def test_data_mode_config():
    """Test data mode configuration (manual/inspesafe)."""
    from run import load_config

    config = load_config("configs/default.yaml")
    data_cfg = config.get("data", {})

    # Check data config exists
    assert "mode" in data_cfg
    assert data_cfg["mode"] in ["manual", "inspesafe"]

    # Check inspesafe configuration
    inspesafe_cfg = data_cfg.get("inspesafe", {})
    assert "dataset_path" in inspesafe_cfg
    assert "session" in inspesafe_cfg


def test_prepare_observations_mode_selection():
    """Test that prepare_observations selects the correct implementation based on mode."""
    from unittest.mock import patch

    from run import prepare_observations

    # Test manual mode (default) - should raise FileNotFoundError if no frames
    config = {
        "data": {"mode": "manual"},
        "video": {"fps": 1.0, "max_frames": 30, "clear_frames": False},
        "audio": {"output_filename": "audio.wav", "sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"},
    }

    # Mock find_video to return None and load_frames to return empty list
    import pytest
    with patch("run.find_video", return_value=None):
        with patch("run.load_frames", return_value=[]):
            with pytest.raises(FileNotFoundError, match="フレームが見つかりません"):
                prepare_observations(
                    config,
                    {".mp4", ".avi"},
                    "frame_{timestamp}s.jpg",
                    config["audio"]
                )

    # Test inspesafe mode selection (should raise FileNotFoundError due to nonexistent path)
    config_inspesafe = {
        "data": {
            "mode": "inspesafe",
            "inspesafe": {
                "dataset_path": "/nonexistent/InspecSafe-V1",
                "session": "test/session"
            }
        },
        "video": {"fps": 1.0, "max_frames": 30, "clear_frames": False},
        "audio": {"output_filename": "audio.wav", "sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"},
    }

    # Should attempt to use inspesafe path and raise FileNotFoundError
    import pytest
    with pytest.raises(FileNotFoundError, match="セッションが見つかりません"):
        prepare_observations(
            config_inspesafe,
            {".mp4", ".avi"},
            "frame_{timestamp}s.jpg",
            config_inspesafe["audio"]
        )
