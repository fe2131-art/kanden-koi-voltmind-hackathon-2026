from src.safety_agent.modality_nodes import AudioAnalyzer


def test_audio_json_parse_success():
    analyzer = AudioAnalyzer()

    events = analyzer._parse_audio_json(
        '{"events":[{"cue":"alarm","severity":"high","evidence":"警報音が鳴っている"}]}'
    )

    cues = analyzer._normalize_audio_events(events)
    assert len(cues) == 1
    assert cues[0].cue == "alarm"
    assert cues[0].severity == "high"
    assert cues[0].evidence == "警報音が鳴っている"


def test_audio_json_parse_failure_returns_none():
    analyzer = AudioAnalyzer()

    events = analyzer._parse_audio_json("not json")

    assert events is None
