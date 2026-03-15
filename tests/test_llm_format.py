"""Test LLM output format validation for SafetyAssessment."""

import json

from src.safety_agent.schema import SafetyAssessment


def test_safety_assessment_from_llm_output():
    """LLM が出力する JSON 形式がパースできることを確認。"""
    # prompt.yaml で指定される出力形式
    llm_output = {
        "risk_level": "high",
        "safety_status": "フォークリフトが人に接近中",
        "detected_hazards": ["forklift_proximity", "personnel_nearby"],
        "action_type": "inspect_region",
        "target_region": "critical_point_0",
        "temporal_status": "worsening",
        "reason": "危険物が人に接近している状況が悪化している",
        "priority": 0.85,
    }

    # SafetyAssessment で検証可能であることを確認
    assessment = SafetyAssessment.model_validate(llm_output)
    assert assessment.risk_level == "high"
    assert assessment.safety_status == "フォークリフトが人に接近中"
    assert len(assessment.detected_hazards) == 2
    assert assessment.action_type == "inspect_region"
    assert assessment.priority == 0.85


def test_safety_assessment_all_risk_levels():
    """すべての risk_level 値をテスト。"""
    for risk_level in ["low", "medium", "high"]:
        llm_output = {
            "risk_level": risk_level,
            "safety_status": f"Risk level: {risk_level}",
            "detected_hazards": [],
            "action_type": "monitor",
            "reason": "テスト",
            "priority": 0.5,
        }
        assessment = SafetyAssessment.model_validate(llm_output)
        assert assessment.risk_level == risk_level


def test_safety_assessment_invalid_risk_level():
    """無効な risk_level を拒否することを確認。"""
    llm_output = {
        "risk_level": "critical",  # "low", "medium", "high" のみが有効
        "safety_status": "テスト",
        "detected_hazards": [],
        "action_type": "monitor",
        "reason": "テスト",
        "priority": 0.5,
    }
    try:
        SafetyAssessment.model_validate(llm_output)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "risk_level" in str(e)


def test_safety_assessment_minimal():
    """最小限のフィールドで検証。"""
    llm_output = {
        "risk_level": "low",
        "safety_status": "安全",
        "action_type": "monitor",
        "reason": "継続監視",
        "priority": 0.0,
    }
    assessment = SafetyAssessment.model_validate(llm_output)
    assert assessment.detected_hazards == []
    assert assessment.target_region is None
    assert assessment.temporal_status == "unknown"


def test_llm_json_output_as_string():
    """LLM が返す JSON 文字列をパースしてからバリデート。"""
    llm_json_str = json.dumps(
        {
            "risk_level": "medium",
            "safety_status": "注意が必要",
            "detected_hazards": ["obstacle"],
            "action_type": "mitigate",
            "reason": "障害物がある",
            "priority": 0.6,
        }
    )

    # JSON をパースして SafetyAssessment で検証
    llm_dict = json.loads(llm_json_str)
    assessment = SafetyAssessment.model_validate(llm_dict)
    assert assessment.risk_level == "medium"
    assert assessment.action_type == "mitigate"
