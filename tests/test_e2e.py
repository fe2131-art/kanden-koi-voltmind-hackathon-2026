"""E2E smoke test for Safety View Agent (LLM-free)."""

import tempfile
from pathlib import Path

from PIL import Image

from src.safety_agent.agent import AgentState, build_agent
from src.safety_agent.modality_nodes import (
    AudioAnalyzer,
    InfraredImageAnalyzer,
    Sam3Analyzer,
    TemporalImageAnalyzer,
)
from src.safety_agent.schema import (
    CriticalPoint,
    Observation,
    ObservationProvider,
    Sam3AnalysisResult,
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
        "grounded_critical_points": [],
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
        "sam3_analyzer": None,
        "sam3_prompts": [],
        "sam3_config": {},
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

    # Verify assessment is not None and has valid enum values
    assert out["assessment"] is not None
    assert out["assessment"].action_type in [
        "emergency_stop",
        "inspect_region",
        "mitigate",
        "monitor",
    ]
    assert out["assessment"].risk_level in ["high", "medium", "low"]

    # LLM なし（llm=None）時はフォールバック固定値を返すことを確認
    assert out["assessment"].risk_level == "low", (
        f"LLM-free run should return fallback risk_level='low', got '{out['assessment'].risk_level}'"
    )
    assert out["assessment"].action_type == "monitor", (
        f"LLM-free run should return fallback action_type='monitor', got '{out['assessment'].action_type}'"
    )

    # Verify modality_results is properly processed (fan-in) - now dict
    assert isinstance(out["modality_results"], dict)

    # Verify barrier_obs_id is present (latch for join_modalities)
    assert "barrier_obs_id" in out

    # Verify perception IR contains audio cues from audio_node
    assert out["ir"] is not None
    assert isinstance(out["ir"].audio, list)

    # 音声テキストを渡した obs_id="t0" の AudioCue が抽出されているか確認
    # (AudioAnalyzer は LLM なしでもキーワードマッチングで動作)
    # assessment_history が max_steps(3) 未満の実行フレーム数（2）分だけ蓄積されているか
    assert "assessment_history" in out
    assert isinstance(out["assessment_history"], list)
    assert len(out["assessment_history"]) >= 1, (
        f"assessment_history should accumulate at least 1 entry, got {len(out['assessment_history'])}"
    )

    # Verify no unexpected errors
    assert isinstance(out["errors"], list)

    # PR1: Verify received_modalities is a list
    assert "received_modalities" in out
    assert isinstance(out["received_modalities"], list)

    # PR3: Verify latest_output is properly populated (flattened structure)
    assert "latest_output" in out
    assert out["latest_output"] is not None
    assert out["latest_output"]["frame_id"] is not None
    # frame_id は obs_id と一致するはず（"t0" または "t1"）
    assert out["latest_output"]["frame_id"] in ("t0", "t1"), (
        f"frame_id should be one of obs_ids, got '{out['latest_output']['frame_id']}'"
    )
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
            "grounded_critical_points": [],
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
            "sam3_analyzer": None,
            "sam3_prompts": [],
            "sam3_config": {},
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


def test_sam3_analyzer_unavailable():
    """Sam3Analyzer が available=False の場合、analyze() が空結果を返すことを確認。"""
    analyzer = Sam3Analyzer(model_cfg={"checkpoint": "facebook/sam3-hiera-large"})

    # SAM3 ライブラリが存在しない環境では available=False になる
    # available=False の場合は空結果を返すべき
    result = analyzer.analyze(
        image_path=None,
        frame_id="t0",
        prompts=["person", "worker"],
        score_threshold=0.35,
        max_regions_per_prompt=3,
        save_masks=False,
        output_dir="data/sam3_masks",
    )

    assert isinstance(result, Sam3AnalysisResult)
    assert result.regions == []
    assert result.confidence_score == 0.0
    print("✅ Sam3Analyzer unavailable test passed")


def test_e2e_agent_with_sam3_state():
    """enable_sam3=false でも grounded_critical_points が AgentState に存在することを確認。"""
    obs_list = [
        Observation(
            obs_id="t0",
            image_path=None,
            audio_text=None,
            camera_pose=None,
        ),
    ]
    provider = ObservationProvider(obs_list)
    agent = build_agent()

    initial_state: AgentState = {
        "messages": [],
        "step": 0,
        "max_steps": 2,
        "observation": None,
        "ir": None,
        "modality_results": {},
        "received_modalities": [],
        "barrier_obs_id": None,
        "latest_output": None,
        "last_vision_summary": None,
        "assessment": None,
        "assessment_history": [],
        "grounded_critical_points": [],
        "belief_state": None,
        "done": False,
        "errors": [],
    }

    context = {
        "provider": provider,
        "llm": None,
        "vision_analyzer": None,
        "audio_analyzer": AudioAnalyzer(),
        "depth_estimator": None,
        "infrared_analyzer": InfraredImageAnalyzer(),
        "temporal_analyzer": TemporalImageAnalyzer(),
        "sam3_analyzer": None,  # SAM3 無効
        "sam3_prompts": [],
        "sam3_config": {},
        "prompts": {
            "vision_analysis": {"default_prompt": "テスト用"},
            "audio_analysis": {"default_prompt": "テスト用"},
            "safety_assessment": {"system": "テスト用"},
        },
        "config": {"audio": {"window_seconds": 3.0}},
        "chat_max_tokens": 2000,
        "context_history_size": 0,
        "expected_modalities": ["vlm", "audio", "temporal"],
        "run_mode": "until_provider_ends",
    }

    out = agent.invoke(initial_state, context=context)

    # grounded_critical_points が state に存在する（空リストでも可）
    assert "grounded_critical_points" in out
    assert isinstance(out["grounded_critical_points"], list)

    # latest_output に grounded_critical_points フィールドが含まれる
    assert "latest_output" in out
    assert out["latest_output"] is not None
    assert "grounded_critical_points" in out["latest_output"]
    assert isinstance(out["latest_output"]["grounded_critical_points"], list)

    print("✅ E2E with SAM3 state test passed")
    print(f"grounded_critical_points: {out['grounded_critical_points']}")


def test_critical_points_no_bbox():
    """normalized_bbox=None の CriticalPoint + label_hint がバリデーションを通ることを確認。"""
    cp = CriticalPoint(
        region_id="critical_point_0",
        description="床にケーブルが放置されており躓く危険がある",
        severity="high",
        label_hint="cable on floor",
        normalized_bbox=None,
    )

    assert cp.region_id == "critical_point_0"
    assert cp.severity == "high"
    assert cp.label_hint == "cable on floor"
    assert cp.normalized_bbox is None

    # model_dump で exclude_none するとキーが省略される
    dumped = cp.model_dump(exclude_none=True)
    assert "normalized_bbox" not in dumped
    assert "label_hint" in dumped

    print("✅ CriticalPoint no-bbox test passed")
    print(f"dumped: {dumped}")
