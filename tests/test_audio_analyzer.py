from src.safety_agent.modality_nodes import AudioAnalyzer


def test_audio_json_parse_success():
    analyzer = AudioAnalyzer()

    events = analyzer._parse_audio_json(
        '{"events":[{"cue":"alarm","confidence":0.9,"direction":"left","evidence":"beep"}]}'
    )

    cues = analyzer._normalize_audio_events(events)
    assert len(cues) == 1
    assert cues[0].cue == "alarm"
    assert cues[0].direction == "left"


def test_audio_json_parse_failure_returns_empty():
    analyzer = AudioAnalyzer()

    events = analyzer._parse_audio_json("not json")

    assert events == []
