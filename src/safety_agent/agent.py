from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from typing_extensions import Annotated, TypedDict

from .perceiver import Perceiver
from .schema import (
    NextViewPlan,
    Observation,
    ObservationProvider,
    PerceptionIR,
    ViewCandidate,
    ViewCommand,
    WorldModel,
)

# =========================
# LLM クライアント（OpenAI互換）
# =========================


class OpenAICompatLLM:
    def __init__(
        self, base_url: str, model: str, api_key: str = "EMPTY", timeout_s: float = 60.0
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        # GPT-5 系モデルは max_tokens ではなく max_completion_tokens を使用
        self._use_max_completion_tokens = "gpt-5" in model.lower()

    def _build_payload(
        self,
        messages: list,
        max_tokens: int = 800,
        response_format: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """モデル固有のトークンパラメータでペイロードを構築する。"""
        payload = {
            "model": self.model,
            "messages": messages,
        }
        # GPT-5系モデルはカスタム temperature をサポートしないため、デフォルト値を使用
        if not self._use_max_completion_tokens:
            payload["temperature"] = 0.2

        if self._use_max_completion_tokens:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        if response_format:
            payload["response_format"] = response_format
        return payload

    def chat_json(
        self, system: str, user: str, max_tokens: int = 800
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # 最初に response_format を試す（gpt-4o はサポート）
        if not self._use_max_completion_tokens:
            payload = self._build_payload(messages, max_tokens, {"type": "json_object"})
        else:
            # gpt-5-nano は response_format をサポートしないためスキップ
            payload = self._build_payload(messages, max_tokens)

        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                # response_format なしで再試行（フォールバック）
                payload_no_format = self._build_payload(messages, max_tokens)
                r = client.post(url, headers=headers, json=payload_no_format)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            return _robust_json_loads(content)

    def chat_text(self, system: str, user: str, max_tokens: int = 500) -> str:
        """チャットリクエストを送信してプレーンテキストの応答を返す。"""
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload = self._build_payload(messages, max_tokens)
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            return content


def _robust_json_loads(text: str) -> Dict[str, Any]:
    # 1) 直接パース
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) 最初の JSON オブジェクトを抽出
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError(f"LLM出力からJSONを解析できませんでした: {text[:300]}...")


# =========================
# LangGraph 状態 + コンテキスト
# =========================


def _add_list(left, right):
    return (left or []) + (right or [])


class AgentState(TypedDict):
    # メッセージログ（オプションだがトレースに便利）
    messages: Annotated[List[Dict[str, str]], add_messages]

    step: int
    max_steps: int

    # 最新の観測と知覚結果
    observation: Optional[Observation]
    ir: Optional[PerceptionIR]

    # 世界モデル、計画、選択
    world: WorldModel
    plan: Optional[NextViewPlan]
    selected: Optional[ViewCommand]

    done: bool
    errors: Annotated[List[str], _add_list]


class ContextSchema(TypedDict):
    provider: ObservationProvider
    perceiver: Perceiver
    llm: Optional[OpenAICompatLLM]

    # 閾値設定
    risk_stop_threshold: float
    hazard_focus_threshold: float


# =========================
# グラフノード関数
# =========================


def ingest_observation(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    """
    step==0で state["observation"] が既に設定されている場合はそれを保持。
    それ以外の場合はプロバイダーから次の観測を取得。
    """
    step = state["step"]
    obs = state.get("observation")
    if step == 0 and obs is not None:
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"[ingest] using initial observation {obs.obs_id}",
                }
            ]
        }
    nxt = runtime.context["provider"].next()
    if nxt is None:
        return {
            "done": True,
            "messages": [
                {
                    "role": "assistant",
                    "content": "[ingest] no more observations -> done",
                }
            ],
        }
    return {
        "observation": nxt,
        "messages": [
            {
                "role": "assistant",
                "content": f"[ingest] loaded observation {nxt.obs_id}",
            }
        ],
    }


def perceive_and_extract_ir(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    obs = state.get("observation")
    if obs is None:
        return {"errors": ["No observation in state"], "done": True}
    ir = runtime.context["perceiver"].run(obs)
    return {
        "ir": ir,
        "messages": [
            {
                "role": "assistant",
                "content": f"[perceive] hazards={len(ir.hazards)} unobserved={len(ir.unobserved)} audio={len(ir.audio)}",
            }
        ],
    }


def update_world_model(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    ir = state.get("ir")
    world = state["world"]
    if ir is None:
        return {"errors": ["No IR to update world"], "done": True}

    # シンプルな融合：ハザードタイプごとに最大信度を保持
    fused: Dict[str, Any] = {h.hazard_type: h for h in world.fused_hazards}
    for h in ir.hazards:
        prev = fused.get(h.hazard_type)
        if prev is None or h.confidence > prev.confidence:
            fused[h.hazard_type] = h

    # Outstanding unobserved: take top risks; in a real system: track coverage over time
    outstanding = sorted(ir.unobserved, key=lambda r: r.risk, reverse=True)[:6]

    new_world = WorldModel(
        fused_hazards=sorted(fused.values(), key=lambda x: x.confidence, reverse=True),
        outstanding_unobserved=outstanding,
        last_selected_view=world.last_selected_view,
    )
    return {
        "world": new_world,
        "messages": [
            {
                "role": "assistant",
                "content": f"[world] fused_hazards={len(new_world.fused_hazards)} outstanding_unobserved={len(new_world.outstanding_unobserved)}",
            }
        ],
    }


def _heuristic_plan(state: AgentState) -> NextViewPlan:
    world = state["world"]
    # prioritize highest-risk unobserved, then hazards
    if world.outstanding_unobserved:
        r0 = world.outstanding_unobserved[0]
        pan = r0.suggested_pan_deg if r0.suggested_pan_deg is not None else 0.0
        tilt = r0.suggested_tilt_deg if r0.suggested_tilt_deg is not None else 0.0
        cand = ViewCandidate(
            view_id=f"heur_{r0.region_id}",
            pan_deg=float(pan),
            tilt_deg=float(tilt),
            zoom=1.0,
            target_region_id=r0.region_id,
            expected_info_gain=min(1.0, 0.5 + r0.risk),
            safety_priority=min(1.0, 0.6 + r0.risk),
            rationale=f"未確認領域({r0.region_id})のリスクが高いため優先観測",
        )
        return NextViewPlan(candidates=[cand], stop=False)
    # fallback: sweep
    cand = ViewCandidate(
        view_id="heur_sweep",
        pan_deg=state["ir"].camera_pose.pan_deg + 30 if state.get("ir") else 30,
        tilt_deg=0.0,
        zoom=1.0,
        target_region_id=None,
        expected_info_gain=0.5,
        safety_priority=0.5,
        rationale="未確認が特定できないためスイープ観測",
    )
    return NextViewPlan(candidates=[cand], stop=False)


def propose_next_view_llm(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    ir = state.get("ir")
    world = state["world"]
    if ir is None:
        return {"errors": ["No IR for planning"], "done": True}

    # Check if LLM is available; if not, use heuristic
    llm = runtime.context.get("llm")
    if llm is None:
        plan = _heuristic_plan(state)
        return {
            "plan": plan,
            "messages": [
                {
                    "role": "assistant",
                    "content": "[plan] LLM not configured -> heuristic fallback",
                }
            ],
        }

    # Compose compact planning context
    top_haz = [
        {"type": h.hazard_type, "conf": h.confidence, "evidence": h.evidence}
        for h in world.fused_hazards[:5]
    ]
    top_unobs = [
        {
            "id": r.region_id,
            "risk": r.risk,
            "desc": r.description,
            "pan": r.suggested_pan_deg,
            "tilt": r.suggested_tilt_deg,
        }
        for r in world.outstanding_unobserved[:6]
    ]
    audio = [{"cue": a.cue, "conf": a.confidence, "dir": a.direction} for a in ir.audio]

    system = (
        "あなたは安全支援エージェントです。目的は『未確認領域を減らし、危険の確信度を上げるために、次に観測すべき視点(画角)を提案する』ことです。\n"
        "必ずJSONのみで出力してください。余計な文章は禁止。\n"
        "JSONは NextViewPlan 形式：\n"
        "{\n"
        '  "candidates": [\n'
        "    {\n"
        '      "view_id": "string",\n'
        '      "pan_deg": number, "tilt_deg": number, "zoom": number,\n'
        '      "target_region_id": "string or null",\n'
        '      "expected_info_gain": 0..1,\n'
        '      "safety_priority": 0..1,\n'
        '      "rationale": "string"\n'
        "    }\n"
        "  ],\n"
        '  "stop": boolean,\n'
        '  "stop_reason": "string or null"\n'
        "}\n"
    )

    user = json.dumps(
        {
            "current_pose": ir.camera_pose.model_dump(),
            "top_hazards": top_haz,
            "top_unobserved_regions": top_unobs,
            "audio_cues": audio,
            "last_selected_view": world.last_selected_view.model_dump()
            if world.last_selected_view
            else None,
            "constraints": {
                "prefer_cover_high_risk_unobserved": True,
                "prefer_raise_confidence_for_existing_hazards": True,
                "return_3_candidates_if_possible": True,
            },
        },
        ensure_ascii=False,
        indent=2,
    )

    try:
        # Use context-configured token limit (for reasoning models like gpt-5-nano)
        chat_max_tokens = runtime.context.get("chat_max_tokens", 2000)
        raw = llm.chat_json(system=system, user=user, max_tokens=chat_max_tokens)
        plan = NextViewPlan.model_validate(raw)
    except Exception as e:
        import traceback

        error_detail = f"{type(e).__name__}: {str(e)}"
        plan = _heuristic_plan(state)
        print(f"⚠️  LLM Error Details:\n{traceback.format_exc()}")
        return {
            "plan": plan,
            "messages": [
                {
                    "role": "assistant",
                    "content": f"[plan] LLM failed -> heuristic. err={error_detail}",
                }
            ],
            "errors": [f"LLM plan fallback: {error_detail}"],
        }

    return {
        "plan": plan,
        "messages": [
            {
                "role": "assistant",
                "content": f"[plan] candidates={len(plan.candidates)} stop={plan.stop}",
            }
        ],
    }


def validate_and_guardrails(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    plan = state.get("plan")
    world = state["world"]
    if plan is None:
        return {"errors": ["No plan to validate"], "done": True}

    # Guardrail 1: must address highest-risk unobserved if above threshold
    risk_thr = runtime.context["hazard_focus_threshold"]
    if (
        world.outstanding_unobserved
        and world.outstanding_unobserved[0].risk >= risk_thr
    ):
        top = world.outstanding_unobserved[0]
        if not any(c.target_region_id == top.region_id for c in plan.candidates):
            # force insert at top
            forced = ViewCandidate(
                view_id=f"forced_{top.region_id}",
                pan_deg=float(top.suggested_pan_deg or 0.0),
                tilt_deg=float(top.suggested_tilt_deg or 0.0),
                zoom=1.0,
                target_region_id=top.region_id,
                expected_info_gain=min(1.0, 0.6 + top.risk),
                safety_priority=min(1.0, 0.7 + top.risk),
                rationale=f"高リスク未確認領域({top.region_id})が未カバーのため強制追加",
            )
            plan = NextViewPlan(
                candidates=[forced] + plan.candidates, stop=False, stop_reason=None
            )

    # Guardrail 2: if plan says stop but unobserved risk still high -> override
    stop_thr = runtime.context["risk_stop_threshold"]
    max_unobs_risk = max([r.risk for r in world.outstanding_unobserved], default=0.0)
    if plan.stop and max_unobs_risk > stop_thr:
        plan.stop = False
        plan.stop_reason = None

    return {
        "plan": plan,
        "messages": [{"role": "assistant", "content": "[validate] guardrails applied"}],
    }


def select_view(state: AgentState, runtime: Runtime[ContextSchema]) -> Dict[str, Any]:
    plan = state.get("plan")
    world = state["world"]
    if plan is None or not plan.candidates:
        # guaranteed fallback
        plan = _heuristic_plan(state)

    # score: prioritize safety first, then info gain
    def score(c: ViewCandidate) -> float:
        return 0.7 * c.safety_priority + 0.3 * c.expected_info_gain

    best = sorted(plan.candidates, key=score, reverse=True)[0]
    cmd = ViewCommand(
        view_id=best.view_id,
        pan_deg=best.pan_deg,
        tilt_deg=best.tilt_deg,
        zoom=best.zoom,
        why=best.rationale,
    )

    # update world with last_selected_view
    new_world = world.model_copy(update={"last_selected_view": cmd})

    # termination heuristic: stop if max unobserved risk below threshold and hazards low
    max_unobs_risk = max(
        [r.risk for r in new_world.outstanding_unobserved], default=0.0
    )
    max_haz_conf = max([h.confidence for h in new_world.fused_hazards], default=0.0)
    done = (max_unobs_risk <= runtime.context["risk_stop_threshold"]) and (
        max_haz_conf < 0.4
    )

    return {
        "selected": cmd,
        "world": new_world,
        "done": done,
        "messages": [
            {
                "role": "assistant",
                "content": f"[select] {cmd.view_id} pan={cmd.pan_deg} tilt={cmd.tilt_deg} done={done}",
            }
        ],
    }


def bump_step(state: AgentState) -> Dict[str, Any]:
    return {"step": state["step"] + 1}


def should_continue(state: AgentState) -> Literal["ingest_observation", END]:
    if state.get("done"):
        return END
    if state["step"] >= state["max_steps"]:
        return END
    return "ingest_observation"


# =========================
# Build graph
# =========================


def build_agent():
    builder = StateGraph(AgentState, context_schema=ContextSchema)

    builder.add_node("ingest_observation", ingest_observation)
    builder.add_node("perceive_and_extract_ir", perceive_and_extract_ir)
    builder.add_node("update_world_model", update_world_model)
    builder.add_node("propose_next_view_llm", propose_next_view_llm)
    builder.add_node("validate_and_guardrails", validate_and_guardrails)
    builder.add_node("select_view", select_view)
    builder.add_node("bump_step", bump_step)

    builder.add_edge(START, "ingest_observation")
    builder.add_edge("ingest_observation", "perceive_and_extract_ir")
    builder.add_edge("perceive_and_extract_ir", "update_world_model")
    builder.add_edge("update_world_model", "propose_next_view_llm")
    builder.add_edge("propose_next_view_llm", "validate_and_guardrails")
    builder.add_edge("validate_and_guardrails", "select_view")
    builder.add_edge("select_view", "bump_step")
    builder.add_conditional_edges("bump_step", should_continue)

    return builder.compile()
