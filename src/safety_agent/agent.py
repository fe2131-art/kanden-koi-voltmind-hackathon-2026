from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

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
    Hazard,
    SafetyAssessment,
    Observation,
    ObservationProvider,
    PerceptionIR,
    UnobservedRegion,
    WorldModel,
)

logger = logging.getLogger(__name__)


def _get_json_schema_for_vllm() -> Dict[str, Any]:
    """vLLM の Structured Outputs 用 JSON スキーマを生成。"""
    return {
        "type": "object",
        "properties": {
            "perceived_hazards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "hazard_type": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                        "related_objects": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                    },
                    "required": ["hazard_type", "confidence", "reason"],
                },
            },
            "estimated_unobserved": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "region_id": {"type": "string"},
                        "description": {"type": "string"},
                        "risk": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "required": ["region_id", "risk"],
                },
            },
            "safety_assessment": {
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
                        "enum": ["focus_region", "increase_safety", "continue_observation"],
                    },
                    "target_region": {"type": ["string", "null"]},
                    "reason": {"type": "string"},
                    "priority": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["risk_level", "safety_status", "action_type", "reason", "priority"],
            },
        },
        "required": ["perceived_hazards", "estimated_unobserved", "safety_assessment"],
    }


# =========================
# LLM クライアント（OpenAI互換）
# =========================


class OpenAICompatLLM:
    def __init__(
        self, base_url: str, model: str, api_key: str = "EMPTY", timeout_s: float = 60.0, is_vllm: bool = False
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.is_vllm = is_vllm or "localhost" in base_url  # ローカルサーバーなら vLLM と判定
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
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # vLLM: Structured Outputs で JSON スキーマを指定（安定化）
        if self.is_vllm:
            payload = self._build_payload(
                messages,
                max_tokens,
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "SafetyAssessmentOutput",
                        "schema": _get_json_schema_for_vllm(),
                        "strict": True,
                    },
                },
            )
        # OpenAI: json_object フォーマットを試す（gpt-5-nano はサポートしない可能性）
        elif not self._use_max_completion_tokens:
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

            # デバッグ: 空応答を確認
            if not content or content.strip() == "":
                print(f"\n⚠️  LLMから空応答\n{'='*60}")
                print(f"ステータスコード: {r.status_code}")
                print(f"APIレスポンス全体: {data}")
                print(f"送信されたメッセージ（user部分の最初500文字）:")
                print(f"  {user[:500]}...")
                print(f"\n{'='*60}\n")
                raise ValueError("LLMが空の応答を返しました")

            try:
                return _robust_json_loads(content)
            except ValueError as e:
                # JSON パースエラーの場合、LLM 出力の詳細を表示（デバッグ用）
                print(f"\n⚠️  LLM出力解析失敗\n{'='*60}")
                print(f"エラー: {str(e)}")
                print(f"\nLLM生出力（最初の3000文字）:\n{content[:3000]}")
                print(f"\n{'='*60}\n")
                logger.error(f"LLM JSON パースエラー: {str(e)}")
                logger.debug(f"LLM raw output:\n{content}")
                raise

    def chat_text(self, system: str, user: str, max_tokens: int = 500) -> str:
        """チャットリクエストを送信してプレーンテキストの応答を返す。"""
        url = f"{self.base_url}/v1/chat/completions"
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
    """LLM 出力から JSON を抽出・パース。複数の形式に対応。"""
    # 1) 直接パース
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) markdown コードブロック（```json ... ```）を抽出
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 3) ネストを考慮した JSON 抽出（最初の { から最後の } まで）
    text = text.strip()
    if text.startswith("{"):
        # 最初の開き括弧から始まる
        depth = 0
        end_idx = -1
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

        if end_idx > 0:
            try:
                return json.loads(text[:end_idx])
            except Exception:
                pass

    # 4) 簡易的な正規表現で { から } を抽出（最後の手段）
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    raise ValueError(f"LLM出力からJSONを解析できませんでした: {text[:500]}...")


# =========================
# LangGraph 状態 + コンテキスト
# =========================


def _merge_dict(left, right):
    """右勝ちの dict merge。right が {} のときは完全リセット。"""
    if right == {}:
        return {}
    return {**(left or {}), **(right or {})}


_RESET_SENTINEL = "__reset__"


def _unique_append_with_reset(left: List[str], right: List[str]) -> List[str]:
    """リセットセンチネルで空にできる重複なし追記 reducer。"""
    if _RESET_SENTINEL in (right or []):
        return []
    return sorted(set((left or []) + (right or [])))


_MAX_MESSAGES = 20


def _sliding_window_messages(left, right):
    """スライディングウィンドウで直近 _MAX_MESSAGES 件のみ保持する reducer。"""
    combined = add_messages(left or [], right or [])
    # add_messages 後のスライシング
    if isinstance(combined, list) and len(combined) > _MAX_MESSAGES:
        return combined[-_MAX_MESSAGES:]
    return combined


_MAX_ERRORS = 50


def _sliding_window_errors(left: List[str], right: List[str]) -> List[str]:
    """直近 _MAX_ERRORS 件のみ保持する reducer。"""
    combined = (left or []) + (right or [])
    if len(combined) > _MAX_ERRORS:
        return combined[-_MAX_ERRORS:]
    return combined


class AgentState(TypedDict):
    # メッセージログ（スライディングウィンドウ化）
    messages: Annotated[List, _sliding_window_messages]

    step: int
    max_steps: int

    # 最新の観測と知覚結果
    observation: Optional[Observation]
    ir: Optional[PerceptionIR]

    # fan-out/fan-in: modality_results を dict に変更（メモリリーク防止）
    modality_results: Annotated[Dict[str, ModalityResult], _merge_dict]

    # fan-in バリア：受け取ったモダリティ名（センチネルでリセット可能）
    received_modalities: Annotated[List[str], _unique_append_with_reset]

    # ラッチ：同フレーム内で fuse_modalities が2回以上実行されるのを防止
    barrier_obs_id: Optional[str]

    latest_output: Optional[Dict[str, Any]]

    # 世界モデル、安全判断
    world: WorldModel
    assessment: Optional[SafetyAssessment]

    done: bool
    errors: Annotated[List[str], _sliding_window_errors]


class ContextSchema(TypedDict):
    provider: ObservationProvider
    perceiver: Perceiver
    llm: Optional[OpenAICompatLLM]
    vision_analyzer: Optional[VisionAnalyzer]
    yolo_detector: Optional[YOLODetector]
    audio_analyzer: AudioAnalyzer
    prompts: dict  # プロンプト設定全体
    chat_max_tokens: int
    max_outstanding_regions: int
    context_history_size: int  # LLM に渡す前回結果の数（0=なし, 1=前回のみ）
    expected_modalities: List[str]
    # run_mode: "until_provider_ends" | "stop_when_safe"
    run_mode: str


# =========================
# グラフノード関数
# =========================


def ingest_observation(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Command:
    """
    観測を取得して、yolo_node、vlm_node、audio_node へ fan-out で送信。
    Command + Send API により真の LangGraph 並列実行を実現。
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

    # fan-out: yolo, vlm, audio を並列ノードへ送信
    sends: list[Send] = [
        Send("yolo_node", {"observation": obs}),
        Send("vlm_node", {"observation": obs}),
        Send("audio_node", {"observation": obs}),
    ]

    return Command(
        update={
            "observation": obs,
            "done": False,  # 毎フレーム開始時にクリア
            "modality_results": {},  # dict リセット
            "received_modalities": [_RESET_SENTINEL],  # バリアカウンタをリセット
            "barrier_obs_id": None,  # ラッチをリセット
            "messages": [{"role": "assistant", "content": f"[ingest] fan-out -> {obs.obs_id}"}],
        },
        goto=sends,
    )


def yolo_node(state: AgentState, runtime: Runtime[ContextSchema]) -> Command:
    """YOLO ノード：物体検出を実行。"""
    obs = state.get("observation")
    yolo = runtime.context.get("yolo_detector")
    objects = []
    error = None

    if obs and obs.image_path and yolo:
        try:
            objects = yolo.detect(obs.image_path)
        except Exception as e:
            error = f"yolo: {e}"

    result = ModalityResult(modality_name="yolo", objects=objects, error=error)
    return Command(
        update={
            "modality_results": {"yolo": result},
            "received_modalities": ["yolo"],
            "messages": [{"role": "assistant", "content": f"[yolo] objects={len(objects)}"}],
        },
        goto="join_modalities",
    )


def vlm_node(state: AgentState, runtime: Runtime[ContextSchema]) -> Command:
    """VLM ノード：画像テキスト分析を実行。"""
    obs = state.get("observation")
    analyzer = runtime.context.get("vision_analyzer")
    description = None
    error = None

    if obs and obs.image_path and analyzer:
        try:
            description = analyzer.analyze(obs.image_path)
            # VisionAnalyzer がエラー文字列を返す場合（"Vision API error ..." など）
            if description and description.startswith("Vision API error"):
                error = description
                # エラー時はデフォルトメッセージを設定（JSON 入力が文字列になるよう）
                description = "Vision API is temporarily unavailable. Using heuristic analysis."
        except Exception as e:
            error = f"vlm: {e}"
            description = "Vision API request failed. Using heuristic analysis."

    result = ModalityResult(modality_name="vlm", description=description, error=error)
    return Command(
        update={
            "modality_results": {"vlm": result},
            "received_modalities": ["vlm"],
            "messages": [{"role": "assistant", "content": f"[vlm] {'ok' if description and not description.startswith('Vision API') else 'error'}"}],
        },
        goto="join_modalities",
    )


def audio_node(state: AgentState, runtime: Runtime[ContextSchema]) -> Command:
    """Audio ノード：音声解析を実行。"""
    obs = state.get("observation")
    audio_cues = []
    error = None

    audio_analyzer = runtime.context.get("audio_analyzer")
    if audio_analyzer and obs:
        try:
            audio_cues = audio_analyzer.analyze(obs.audio_text)
        except Exception as e:
            error = f"audio: {e}"

    result = ModalityResult(modality_name="audio", audio_cues=audio_cues, error=error)
    return Command(
        update={
            "modality_results": {"audio": result},
            "received_modalities": ["audio"],
            "messages": [{"role": "assistant", "content": f"[audio] cues={len(audio_cues)}"}],
        },
        goto="join_modalities",
    )


def join_modalities(state: AgentState, runtime: Runtime[ContextSchema]) -> Command:
    """
    Fan-in バリア + ラッチ：全て期待するモダリティが受け取ったかを確認。
    ラッチ（barrier_obs_id）により、同フレーム内で fuse_modalities は1回だけ実行される。
    """
    expected = set(runtime.context.get("expected_modalities", ["yolo", "vlm", "audio"]))
    received = set(state.get("received_modalities", []))
    obs = state.get("observation")
    current_obs_id = obs.obs_id if obs else None
    barrier_obs_id = state.get("barrier_obs_id")

    if expected.issubset(received):
        # 全てのモダリティが揃った
        if barrier_obs_id == current_obs_id:
            # 既にこのフレーム（obs_id）で fuse 実行済み → スキップ
            return Command(
                update={"messages": [{"role": "assistant", "content": "[join] already fused, skip"}]},
                goto=END,
            )
        # 初回: ラッチをセットして fuse へ
        return Command(
            update={
                "barrier_obs_id": current_obs_id,
                "messages": [{"role": "assistant", "content": "[join] all modalities received"}],
            },
            goto="fuse_modalities",
        )
    else:
        # 未揃い → このサブタスクの枝を終了
        return Command(
            update={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "[join] waiting",
                    }
                ]
            },
            goto=END,
        )


def fuse_modalities(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    """
    全モダリティの ModalityResult を PerceptionIR に統合する。
    fan-in ポイント：yolo_node、vlm_node、audio_node の完了後に実行。
    """
    obs = state.get("observation")
    if obs is None:
        return {"errors": ["No observation in state"], "done": True}

    # modality_results は既に dict
    results: Dict[str, ModalityResult] = state.get("modality_results", {})

    yolo = results.get("yolo")
    vlm = results.get("vlm")
    audio = results.get("audio")

    objects = (yolo.objects if yolo and yolo.objects else [])
    audio_cues = (audio.audio_cues if audio and audio.audio_cues else [])
    description = (vlm.description if vlm else None)
    modality_errors = [r.error for r in results.values() if r.error]

    # PerceptionIR を作成（Perceiver 推論は determine_next_action_llm で統合実行）
    ir = PerceptionIR(
        obs_id=obs.obs_id,
        camera_pose=obs.camera_pose,
        objects=objects,
        hazards=[],  # Perceiver 推論は determine_next_action_llm で実行
        unobserved=[],  # Perceiver 推論は determine_next_action_llm で実行
        audio=audio_cues,
        vision_description=description,
        modality_errors=modality_errors,
    )

    return {
        "ir": ir,
        "messages": [
            {
                "role": "assistant",
                "content": f"[fuse] ok (perception inference in determine_next_action_llm)",
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
    max_regions = runtime.context["max_outstanding_regions"]
    outstanding = sorted(ir.unobserved, key=lambda r: r.risk, reverse=True)[:max_regions]

    new_world = WorldModel(
        fused_hazards=sorted(fused.values(), key=lambda x: x.confidence, reverse=True),
        outstanding_unobserved=outstanding,
        last_assessment=world.last_assessment,
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


def determine_next_action_llm(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    """VLM/YOLO/音声+世界モデルの統合結果から、次に起こすべき行動を LLM で決定。"""
    ir = state.get("ir")
    world = state["world"]
    if ir is None:
        return {"errors": ["No IR for action determination"], "done": True}

    llm = runtime.context.get("llm")
    if llm is None:
        # ヒューリスティックフォールバック
        assessment = _heuristic_assessment(world)
        return {
            "assessment": assessment,
            "world": world.model_copy(update={"last_assessment": assessment}),
            "messages": [{"role": "assistant", "content": "[assess] LLM not configured -> heuristic"}],
        }

    # コンテキスト構築
    context_history_size = runtime.context.get("context_history_size", 1)
    max_regions = runtime.context["max_outstanding_regions"]

    context_data = {
        # 現在フレームの知覚結果
        "vision_description": ir.vision_description,
        "detected_objects": [o.model_dump() for o in ir.objects[:10]],
        "audio_cues": [a.model_dump() for a in ir.audio],
        "fused_hazards": [h.model_dump() for h in world.fused_hazards[:5]],
        "outstanding_unobserved": [r.model_dump() for r in world.outstanding_unobserved[:max_regions]],
        # 前回の判断結果（context_history_size >= 1 の場合のみ）
        "previous_assessment": (
            world.last_assessment.model_dump() if context_history_size >= 1 and world.last_assessment else None
        ),
    }

    # prompt.yaml の next_view_proposal セクションから取得
    next_action_cfg = runtime.context["prompts"].get("next_view_proposal", {})
    system = next_action_cfg.get("system", "").strip()

    try:
        chat_max_tokens = runtime.context["chat_max_tokens"]
        raw = llm.chat_json(system=system, user=json.dumps(context_data, ensure_ascii=False), max_tokens=chat_max_tokens)

        # 新フォーマット: 知覚推論（perceived_hazards, estimated_unobserved）+ 安全判断（safety_assessment）
        perceived_hazards = raw.get("perceived_hazards", [])
        estimated_unobserved = raw.get("estimated_unobserved", [])
        safety_assessment_dict = raw.get("safety_assessment", raw)  # フォールバック: 旧フォーマット対応

        # ir を LLM 推論結果で更新（vision_description は保持）
        original_vision_description = ir.vision_description
        ir.hazards = [Hazard(**h) for h in perceived_hazards]

        # estimated_unobserved のデフォルト値補完（description など不足フィールド）
        completed_unobserved = []
        for u in estimated_unobserved:
            if "description" not in u:
                u["description"] = f"Unobserved region: {u.get('region_id', 'unknown')}"
            completed_unobserved.append(UnobservedRegion(**u))
        ir.unobserved = completed_unobserved

        # vision_description を元の値で復元（LLM 更新後も保持）
        ir.vision_description = original_vision_description

        # SafetyAssessment を検証・生成
        assessment = SafetyAssessment.model_validate(safety_assessment_dict)

    except Exception as e:
        import traceback
        print(f"⚠️  LLM Error Details:\n{traceback.format_exc()}")
        assessment = _heuristic_assessment(world)
        return {
            "assessment": assessment,
            "ir": ir,
            "world": world.model_copy(update={"last_assessment": assessment}),
            "errors": [f"LLM assess+perceiver fallback: {e}"],
            "messages": [{"role": "assistant", "content": f"[assess] LLM failed -> heuristic. err={e}"}],
        }

    new_world = world.model_copy(update={"last_assessment": assessment})
    return {
        "assessment": assessment,
        "ir": ir,
        "world": new_world,
        "messages": [{"role": "assistant", "content": f"[assess] {assessment.action_type} risk={assessment.risk_level} priority={assessment.priority:.2f} (perceiver+llm)"}],
    }


def _heuristic_assessment(world: WorldModel) -> SafetyAssessment:
    """LLM 不使用時のフォールバック。"""
    if world.outstanding_unobserved:
        top = world.outstanding_unobserved[0]
        return SafetyAssessment(
            risk_level="high" if top.risk >= 0.7 else "medium",
            safety_status=f"未確認領域({top.region_id})にリスクあり",
            detected_hazards=[top.description],
            action_type="focus_region",
            target_region=top.region_id,
            reason=f"高リスク未確認領域({top.region_id})を優先観測",
            priority=min(1.0, top.risk),
        )
    if world.fused_hazards and any(h.confidence < 0.7 for h in world.fused_hazards):
        low = [h for h in world.fused_hazards if h.confidence < 0.7]
        return SafetyAssessment(
            risk_level="medium",
            safety_status=f"低信度ハザード {len(low)} 件を確認中",
            detected_hazards=[h.hazard_type for h in low],
            action_type="increase_safety",
            reason=f"ハザード信度が低い: {len(low)} 件",
            priority=0.8,
        )
    return SafetyAssessment(
        risk_level="low",
        safety_status="環境は安全",
        detected_hazards=[],
        action_type="continue_observation",
        reason="環境は安全と判断",
        priority=0.0,
    )


def emit_output(state: AgentState) -> Dict[str, Any]:
    """
    フレームごとに latest_output を更新し、統合出力を生成。
    """
    obs = state.get("observation")
    ir = state.get("ir")
    world = state.get("world")
    assessment = state.get("assessment")

    output = {
        "obs_id": obs.obs_id if obs else None,
        "video_timestamp": obs.video_timestamp if obs else None,
        "ir": ir.model_dump() if ir else None,
        "world": {
            "fused_hazards": [h.model_dump() for h in (world.fused_hazards if world else [])],
            "outstanding_unobserved": [
                r.model_dump() for r in (world.outstanding_unobserved if world else [])
            ],
        },
        "assessment": assessment.model_dump() if assessment else None,
        "frame_errors": list(state.get("errors", [])),
        "step": state.get("step", 0),
    }

    return {
        "latest_output": output,
        "messages": [
            {
                "role": "assistant",
                "content": f"[emit] step={state.get('step', 0)} obs={obs.obs_id if obs else 'none'}",
            }
        ],
    }


def bump_step(state: AgentState) -> Dict[str, Any]:
    return {"step": state["step"] + 1}


def should_continue(state: AgentState) -> str:
    max_steps = state.get("max_steps", 0)

    if max_steps and state["step"] >= max_steps:
        return END
    if state.get("done"):
        return END
    return "ingest_observation"


# =========================
# Build graph
# =========================


def build_agent():
    builder = StateGraph(AgentState, context_schema=ContextSchema)

    # ノード登録（yolo/vlm に分割した fan-out/fan-in 対応）
    builder.add_node("ingest_observation", ingest_observation)
    builder.add_node("yolo_node", yolo_node)
    builder.add_node("vlm_node", vlm_node)
    builder.add_node("audio_node", audio_node)
    builder.add_node("join_modalities", join_modalities)  # ラッチ付き fan-in バリア
    builder.add_node("fuse_modalities", fuse_modalities)
    builder.add_node("update_world_model", update_world_model)
    builder.add_node("determine_next_action_llm", determine_next_action_llm)
    builder.add_node("emit_output", emit_output)
    builder.add_node("bump_step", bump_step)

    # エッジ設定
    builder.add_edge(START, "ingest_observation")
    # fan-out: yolo/vlm/audio ノードは Command で goto="join_modalities" するため静的エッジは不要
    builder.add_edge("fuse_modalities", "determine_next_action_llm")  # PerceptionIR 生成 → Perceiver 推論 + 安全判断
    builder.add_edge("determine_next_action_llm", "update_world_model")  # ir を LLM で更新 → world 融合
    builder.add_edge("update_world_model", "emit_output")
    builder.add_edge("emit_output", "bump_step")
    builder.add_conditional_edges("bump_step", should_continue)

    return builder.compile()
