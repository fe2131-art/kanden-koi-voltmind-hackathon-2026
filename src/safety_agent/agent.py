from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import Command, Send
from typing_extensions import Annotated, TypedDict

from .modality_nodes import (
    AudioAnalyzer,
    ModalityResult,
    VisionAnalyzer,
    YOLODetector,
)
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

    # fan-out/fan-in: modality_results が vision_node, audio_node から蓄積される
    modality_results: Annotated[List[ModalityResult], _add_list]

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
    vision_analyzer: Optional[VisionAnalyzer]
    yolo_detector: Optional[YOLODetector]
    audio_analyzer: AudioAnalyzer
    chat_max_tokens: int

    # 閾値設定
    risk_stop_threshold: float
    hazard_focus_threshold: float

    # ビュー選択戦略のパラメータ
    max_outstanding_regions: int
    safety_priority_weight: float
    info_gain_weight: float
    safety_priority_base: float


# =========================
# グラフノード関数
# =========================


def ingest_observation(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Command:
    """
    観測を取得して、vision_node と audio_node へ fan-out で送信。
    Command + Send API により真の並列実行を実現。
    """
    step = state["step"]
    obs = state.get("observation")

    # step==0 で初期観測がある場合はそのまま使う
    if step == 0 and obs is not None:
        pass
    else:
        nxt = runtime.context["provider"].next()
        if nxt is None:
            return Command(
                update={"done": True},
                goto=END,
            )
        obs = nxt

    # fan-out: vision と audio を並列ノードへ送信
    sends: list[Send] = [
        Send("vision_node", {"observation": obs}),
        Send("audio_node", {"observation": obs}),
    ]

    return Command(
        update={
            "observation": obs,
            "modality_results": [],  # fan-in バッファをリセット
            "messages": [{"role": "assistant", "content": f"[ingest] fan-out -> {obs.obs_id}"}],
        },
        goto=sends,
    )


def vision_node(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    """Vision ノード：VLM + YOLO を並列実行。"""
    obs = state.get("observation")
    if obs is None:
        return {
            "modality_results": [
                ModalityResult(
                    modality_name="vision",
                    error="No observation provided",
                )
            ]
        }

    ctx = runtime.context
    objects = []
    description = None
    error = None

    try:
        # YOLO 検出（CPU/GPU）
        if ctx.get("yolo_detector") and obs.image_path:
            objects = ctx["yolo_detector"].detect(obs.image_path)
        elif obs.image_path:
            # フォールバック：簡易画像分析
            objects = []

        # VLM 分析（HTTP I/O）
        if ctx.get("vision_analyzer") and obs.image_path:
            description = ctx["vision_analyzer"].analyze(obs.image_path)
    except Exception as e:
        error = f"vision_node error: {e}"

    result = ModalityResult(
        modality_name="vision",
        objects=objects,
        description=description,
        error=error,
    )
    return {
        "modality_results": [result],
        "messages": [
            {
                "role": "assistant",
                "content": f"[vision] objects={len(objects)} vlm={'ok' if description else 'none'}",
            }
        ],
    }


def audio_node(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    """Audio ノード：音声解析を実行。"""
    obs = state.get("observation")
    if obs is None:
        return {
            "modality_results": [
                ModalityResult(
                    modality_name="audio",
                    error="No observation provided",
                )
            ]
        }

    try:
        audio_cues = runtime.context["audio_analyzer"].analyze(obs.audio_text)
        error = None
    except Exception as e:
        audio_cues = []
        error = f"audio_node error: {e}"

    result = ModalityResult(
        modality_name="audio",
        audio_cues=audio_cues,
        error=error,
    )
    return {
        "modality_results": [result],
        "messages": [{"role": "assistant", "content": f"[audio] cues={len(audio_cues)}"}],
    }


def fuse_modalities(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    """
    全モダリティの ModalityResult を PerceptionIR に統合する。
    fan-in ポイント：vision_node と audio_node の完了後に実行。
    """
    obs = state.get("observation")
    if obs is None:
        return {"errors": ["No observation in state"], "done": True}

    # modality_results を辞書化して検索しやすく
    results = {r.modality_name: r for r in state.get("modality_results", [])}

    vision = results.get("vision")
    audio = results.get("audio")

    objects = vision.objects if vision else []
    audio_cues = audio.audio_cues if audio else []
    description = vision.description if vision else None
    modality_errors = [r.error for r in results.values() if r.error]

    # Perceiver（ハザード推定専用）を呼び出す
    ir = runtime.context["perceiver"].estimate(obs, objects, audio_cues, description)
    ir = ir.model_copy(update={"modality_errors": modality_errors})

    return {
        "ir": ir,
        "messages": [
            {
                "role": "assistant",
                "content": f"[fuse] hazards={len(ir.hazards)} unobserved={len(ir.unobserved)} errors={len(modality_errors)}",
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
    max_regions = runtime.context.get("max_outstanding_regions", 6)
    outstanding = sorted(ir.unobserved, key=lambda r: r.risk, reverse=True)[:max_regions]

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
    max_regions = runtime.context.get("max_outstanding_regions", 6)
    top_unobs = [
        {
            "id": r.region_id,
            "risk": r.risk,
            "desc": r.description,
            "pan": r.suggested_pan_deg,
            "tilt": r.suggested_tilt_deg,
        }
        for r in world.outstanding_unobserved[:max_regions]
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

        # LLM 応答の詳細ログ
        llm_log = f"[LLM-RESPONSE] Candidates: {len(plan.candidates)}, Stop: {plan.stop}"
        for i, cand in enumerate(plan.candidates[:3]):
            llm_log += f"\n  {i+1}. {cand.view_id}: safety={cand.safety_priority:.2f}, info={cand.expected_info_gain:.2f}"
            llm_log += f"\n     Reason: {cand.rationale[:60]}..."

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
                "content": f"[plan] LLM: {len(plan.candidates)} candidates, stop={plan.stop}",
            },
            {
                "role": "assistant",
                "content": llm_log,
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
                safety_priority=min(1.0, runtime.context.get("safety_priority_base", 0.7) + top.risk),
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
    safety_weight = runtime.context.get("safety_priority_weight", 0.7)
    info_weight = runtime.context.get("info_gain_weight", 0.3)

    def score(c: ViewCandidate) -> float:
        return safety_weight * c.safety_priority + info_weight * c.expected_info_gain

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

    # ノード登録（fan-out/fan-in 対応）
    builder.add_node("ingest_observation", ingest_observation)
    builder.add_node("vision_node", vision_node)
    builder.add_node("audio_node", audio_node)
    builder.add_node("fuse_modalities", fuse_modalities)
    builder.add_node("update_world_model", update_world_model)
    builder.add_node("propose_next_view_llm", propose_next_view_llm)
    builder.add_node("validate_and_guardrails", validate_and_guardrails)
    builder.add_node("select_view", select_view)
    builder.add_node("bump_step", bump_step)

    # エッジ設定
    builder.add_edge(START, "ingest_observation")
    # fan-out は ingest_observation が Command で goto=[Send(...), Send(...)] するため
    # ここには条件付きエッジは不要（Command が直接ルーティング）
    builder.add_edge("vision_node", "fuse_modalities")
    builder.add_edge("audio_node", "fuse_modalities")
    builder.add_edge("fuse_modalities", "update_world_model")
    builder.add_edge("update_world_model", "propose_next_view_llm")
    builder.add_edge("propose_next_view_llm", "validate_and_guardrails")
    builder.add_edge("validate_and_guardrails", "select_view")
    builder.add_edge("select_view", "bump_step")
    builder.add_conditional_edges("bump_step", should_continue)

    return builder.compile()
