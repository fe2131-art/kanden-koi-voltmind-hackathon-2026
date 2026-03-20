"""TTSNarrator のユニットテスト。

Kokoro KPipeline は重いモデルロードが必要なため、すべてのテストでモック化する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from safety_agent.tts_narrator import TTSNarrator

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _make_config(enabled: bool = True, **overrides: Any) -> dict:
    """テスト用最小設定を生成する。"""
    tts: dict[str, Any] = {
        "enabled": enabled,
        "voice": "jf_alpha",
        "speed": 1.0,
        "lang_code": "j",
        "output_dir": "data/voice",  # tmp_path で上書きする場合はoverrideで渡す
        "device": None,
    }
    tts.update(overrides)
    return {"tts": tts}


def _make_fake_result(audio_array: np.ndarray) -> MagicMock:
    """Kokoro の KPipeline が yield する result オブジェクトのモックを生成する。"""
    result = MagicMock()
    # audio は CPU tensor を模倣: .cpu().numpy() が呼べるようにする
    tensor_mock = MagicMock()
    tensor_mock.cpu.return_value.numpy.return_value = audio_array
    result.audio = tensor_mock
    return result


# ---------------------------------------------------------------------------
# enabled=False のテスト（KPipeline が一切ロードされないことを確認）
# ---------------------------------------------------------------------------


def test_generate_disabled_returns_none(tmp_path: Path):
    """tts.enabled=false のときは何もせず None を返す。"""
    config = _make_config(enabled=False, output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    result = narrator.generate("frame_0", "テストテキスト")

    assert result is None
    assert narrator._pipeline is None  # KPipeline が初期化されていない


def test_generate_disabled_creates_no_files(tmp_path: Path):
    """tts.enabled=false のとき WAV ファイルが生成されない。"""
    config = _make_config(enabled=False, output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    narrator.generate("frame_1", "警告: 危険エリアです")

    assert list(tmp_path.glob("*.wav")) == []


# ---------------------------------------------------------------------------
# 空テキストのテスト
# ---------------------------------------------------------------------------


def test_generate_empty_text_returns_none(tmp_path: Path):
    """空文字列のときはスキップして None を返す。"""
    config = _make_config(output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    assert narrator.generate("frame_0", "") is None
    assert narrator.generate("frame_0", "   ") is None


# ---------------------------------------------------------------------------
# 正常系テスト（KPipeline をモック化）
# ---------------------------------------------------------------------------


def test_generate_creates_wav_file(tmp_path: Path):
    """正常な音声合成で WAV ファイルが生成される。"""
    config = _make_config(output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    # 0.1秒分のダミー音声（24000Hz × 0.1s = 2400 サンプル）
    dummy_audio = np.zeros(2400, dtype=np.float32)
    fake_result = _make_fake_result(dummy_audio)

    mock_pipeline = MagicMock(return_value=[fake_result])

    with patch("safety_agent.tts_narrator.TTSNarrator._ensure_pipeline"):
        narrator._pipeline = mock_pipeline
        out_path = narrator.generate("img_0", "安全状態です。継続観測中。")

    assert out_path is not None
    assert out_path == tmp_path / "img_0.wav"
    assert out_path.exists()


def test_generate_filename_matches_frame_id(tmp_path: Path):
    """出力ファイル名が frame_id と一致する。"""
    config = _make_config(output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    dummy_audio = np.zeros(1000, dtype=np.float32)
    fake_result = _make_fake_result(dummy_audio)
    mock_pipeline = MagicMock(return_value=[fake_result])

    with patch("safety_agent.tts_narrator.TTSNarrator._ensure_pipeline"):
        narrator._pipeline = mock_pipeline
        out_path = narrator.generate("frame_42", "テキスト")

    assert out_path is not None
    assert out_path.name == "frame_42.wav"


def test_generate_concatenates_multiple_chunks(tmp_path: Path):
    """複数チャンク（長文分割時）を結合して1ファイルに保存する。"""
    import soundfile as sf

    config = _make_config(output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    chunk1 = np.full(2400, 0.1, dtype=np.float32)
    chunk2 = np.full(2400, 0.2, dtype=np.float32)
    results = [_make_fake_result(chunk1), _make_fake_result(chunk2)]
    mock_pipeline = MagicMock(return_value=results)

    with patch("safety_agent.tts_narrator.TTSNarrator._ensure_pipeline"):
        narrator._pipeline = mock_pipeline
        out_path = narrator.generate("img_multi", "長いテキスト…")

    assert out_path is not None
    data, sr = sf.read(str(out_path))
    assert sr == 24000
    assert len(data) == 4800  # 2チャンク結合で4800サンプル


def test_generate_audio_is_moved_to_cpu(tmp_path: Path):
    """.cpu() が必ず呼ばれること（CUDA テンソル対応）。"""
    config = _make_config(output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    dummy_audio = np.zeros(100, dtype=np.float32)
    fake_result = _make_fake_result(dummy_audio)
    mock_pipeline = MagicMock(return_value=[fake_result])

    with patch("safety_agent.tts_narrator.TTSNarrator._ensure_pipeline"):
        narrator._pipeline = mock_pipeline
        narrator.generate("frame_cpu", "テスト")

    # .cpu() が呼ばれたことを検証
    fake_result.audio.cpu.assert_called_once()


def test_generate_result_audio_none_skipped(tmp_path: Path):
    """result.audio が None のチャンクはスキップされ、有効チャンクのみ保存される。"""
    config = _make_config(output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    none_result = MagicMock()
    none_result.audio = None
    valid_result = _make_fake_result(np.zeros(2400, dtype=np.float32))

    mock_pipeline = MagicMock(return_value=[none_result, valid_result])

    with patch("safety_agent.tts_narrator.TTSNarrator._ensure_pipeline"):
        narrator._pipeline = mock_pipeline
        out_path = narrator.generate("frame_mix", "テキスト")

    assert out_path is not None
    assert out_path.exists()


def test_generate_all_audio_none_returns_none(tmp_path: Path):
    """すべての result.audio が None のとき、ファイルを作らず None を返す。"""
    config = _make_config(output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    none_result = MagicMock()
    none_result.audio = None
    mock_pipeline = MagicMock(return_value=[none_result])

    with patch("safety_agent.tts_narrator.TTSNarrator._ensure_pipeline"):
        narrator._pipeline = mock_pipeline
        out_path = narrator.generate("frame_none", "テキスト")

    assert out_path is None
    assert list(tmp_path.glob("*.wav")) == []


# ---------------------------------------------------------------------------
# エラーハンドリングテスト
# ---------------------------------------------------------------------------


def test_generate_pipeline_exception_returns_none(tmp_path: Path):
    """KPipeline 呼び出し中に例外が発生してもエージェントを止めず None を返す。"""
    config = _make_config(output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    mock_pipeline = MagicMock(side_effect=RuntimeError("合成エラー"))

    with patch("safety_agent.tts_narrator.TTSNarrator._ensure_pipeline"):
        narrator._pipeline = mock_pipeline
        result = narrator.generate("frame_err", "エラーテスト")

    assert result is None


def test_generate_init_failure_disables_tts_and_returns_none(tmp_path: Path):
    """KPipeline 初期化失敗時は enabled=False にフォールバックして None を返す（エージェント継続）。"""
    config = _make_config(output_dir=str(tmp_path))
    narrator = TTSNarrator(config)
    assert narrator.enabled is True

    with patch.object(
        narrator, "_ensure_pipeline", side_effect=RuntimeError("CUDA not available")
    ):
        result = narrator.generate("frame_init_err", "テスト")

    assert result is None
    assert narrator.enabled is False  # 以降のフレームも TTS スキップ


# ---------------------------------------------------------------------------
# 設定値の反映テスト
# ---------------------------------------------------------------------------


def test_config_values_applied(tmp_path: Path):
    """speed / voice / lang_code が設定値から正しく読み込まれる。"""
    config = _make_config(
        voice="jf_gongitsune",
        speed=0.8,
        lang_code="j",
        output_dir=str(tmp_path),
    )
    narrator = TTSNarrator(config)

    assert narrator.voice == "jf_gongitsune"
    assert narrator.speed == pytest.approx(0.8)
    assert narrator.lang_code == "j"


def test_speed_passed_to_pipeline(tmp_path: Path):
    """speed パラメータが KPipeline の呼び出し時に渡される。"""
    config = _make_config(speed=0.75, output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    dummy_audio = np.zeros(100, dtype=np.float32)
    fake_result = _make_fake_result(dummy_audio)
    mock_pipeline = MagicMock(return_value=[fake_result])

    with patch("safety_agent.tts_narrator.TTSNarrator._ensure_pipeline"):
        narrator._pipeline = mock_pipeline
        narrator.generate("frame_speed", "速度テスト")

    mock_pipeline.assert_called_once_with("速度テスト", voice="jf_alpha", speed=0.75)


# ---------------------------------------------------------------------------
# run.py の _on_frame 統合テスト
# ---------------------------------------------------------------------------


def test_on_frame_filename_uses_video_timestamp(tmp_path: Path):
    """video_timestamp がある場合、WAV ファイル名が frame_{ts:.1f}s.wav になる。"""
    config = _make_config(output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    dummy_audio = np.zeros(2400, dtype=np.float32)
    fake_result = _make_fake_result(dummy_audio)
    mock_pipeline = MagicMock(return_value=[fake_result])

    # run.py _on_frame のファイル名ロジックを再現
    frame_output = {
        "frame_id": "img_0",
        "video_timestamp": 1.0,
        "assessment": {"safety_status": "継続観測中"},
    }
    assessment = frame_output.get("assessment") or {}
    safety_status = assessment.get("safety_status", "")
    ts = frame_output.get("video_timestamp")
    tts_name = f"frame_{ts:.1f}s" if ts is not None else frame_output.get("frame_id", "frame")

    with patch("safety_agent.tts_narrator.TTSNarrator._ensure_pipeline"):
        narrator._pipeline = mock_pipeline
        out_path = narrator.generate(tts_name, safety_status)

    assert out_path is not None
    assert out_path.name == "frame_1.0s.wav"


def test_on_frame_assessment_none_no_error(tmp_path: Path):
    """assessment が None のフレーム出力でも AttributeError が発生しない。"""
    config = _make_config(output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    frame_output = {"frame_id": "img_0", "video_timestamp": 0.0, "assessment": None}

    assessment = frame_output.get("assessment") or {}
    safety_status = assessment.get("safety_status", "")

    result = narrator.generate("frame_0.0s", safety_status)
    assert result is None


def test_on_frame_assessment_missing_no_error(tmp_path: Path):
    """assessment キー自体が存在しないフレーム出力でもエラーが発生しない。"""
    config = _make_config(output_dir=str(tmp_path))
    narrator = TTSNarrator(config)

    frame_output = {"frame_id": "img_1", "video_timestamp": 1.0}

    assessment = frame_output.get("assessment") or {}
    safety_status = assessment.get("safety_status", "")

    result = narrator.generate("frame_1.0s", safety_status)
    assert result is None
