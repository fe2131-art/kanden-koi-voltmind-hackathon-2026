"""vLLM Structured Outputs 用 JSON スキーマ定義。

複数の LLM 出力型に対応したスキーマを管理。
"""

from typing import Any, Dict, Literal


def get_belief_state_schema() -> Dict[str, Any]:
    """BeliefState 用 vLLM JSON スキーマを生成。

    時系列の危険状態追跡用スキーマ。
    """
    return {
        "type": "object",
        "properties": {
            "hazard_tracks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "hazard_id": {"type": "string"},
                        "hazard_type": {
                            "type": "string",
                            "enum": [
                                "visible_hazard",
                                "blind_spot",
                                "overheat",
                                "abnormal_sound",
                                "scene_change",
                                "obstacle",
                                "equipment_anomaly",
                                "unknown",
                            ],
                        },
                        "region_id": {"type": ["string", "null"]},
                        "status": {
                            "type": "string",
                            "enum": [
                                "new",
                                "persistent",
                                "worsening",
                                "improving",
                                "resolved",
                                "unknown",
                            ],
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical", "unknown"],
                        },
                        "confidence_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "supporting_modalities": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "vision",
                                    "audio",
                                    "depth",
                                    "infrared",
                                    "temporal",
                                ],
                            },
                            "default": [],
                        },
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                    },
                    "required": [
                        "hazard_id",
                        "hazard_type",
                        "status",
                        "severity",
                        "confidence_score",
                    ],
                },
                "default": [],
            },
            "overall_risk": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical", "unknown"],
                "default": "unknown",
            },
            "recommended_focus_regions": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
        },
        "required": ["hazard_tracks", "overall_risk", "recommended_focus_regions"],
    }


def get_safety_assessment_schema() -> Dict[str, Any]:
    """SafetyAssessment 用 vLLM JSON スキーマを生成。

    最終安全判断用スキーマ。
    """
    return {
        "type": "object",
        "properties": {
            "risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
            "safety_status": {"type": "string"},
            "detected_hazards": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
            "action_type": {
                "type": "string",
                "enum": ["emergency_stop", "inspect_region", "mitigate", "monitor"],
            },
            "target_region": {"type": ["string", "null"]},
            "reason": {"type": "string"},
            "priority": {"type": "number", "minimum": 0, "maximum": 1},
            "temporal_status": {
                "type": "string",
                "enum": [
                    "new",
                    "persistent",
                    "worsening",
                    "improving",
                    "resolved",
                    "unknown",
                ],
            },
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "risk_level",
            "safety_status",
            "detected_hazards",
            "action_type",
            "reason",
            "priority",
            "temporal_status",
            "confidence_score",
        ],
    }


def get_json_schema(
    schema_type: Literal["belief_state", "safety_assessment"],
) -> Dict[str, Any]:
    """指定されたスキーマ型を返すメイン関数。

    Args:
        schema_type: "belief_state" または "safety_assessment"

    Returns:
        vLLM Structured Outputs 対応の JSON スキーマ辞書
    """
    if schema_type == "belief_state":
        return get_belief_state_schema()
    elif schema_type == "safety_assessment":
        return get_safety_assessment_schema()
    else:
        raise ValueError(f"Unknown schema_type: {schema_type}")
