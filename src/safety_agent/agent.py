import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import Command, Send
from typing_extensions import Annotated, TypedDict

from .modality_nodes import (
    AudioAnalyzer,
    DepthEstimator,
    ModalityResult,
    VisionAnalyzer,
    YOLODetector,
)
from .schema import (
    DepthAnalysisResult,
    Observation,
    ObservationProvider,
    PerceptionIR,
    SafetyAssessment,
)

logger = logging.getLogger(__name__)


def _get_json_schema_for_vllm() -> Dict[str, Any]:
    """vLLM の Structured Outputs 用 JSON スキーマを生成。"""
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
            "evidence": {
                "type": ["object", "null"],  # Optional: null の場合あり
                "properties": {
                    "vision": {"type": "array", "items": {"type": "string"}},
                    "yolo": {"type": "array", "items": {"type": "string"}},
                    "audio": {"type": "array", "items": {"type": "string"}},
                    "previous": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["vision", "yolo", "audio", "previous"],
            },
        },
        "required": [
            "risk_level",
            "safety_status",
            "detected_hazards",
            "action_type",
            "reason",
            "priority",
            "temporal_status",
        ],
    }


# =========================
# LLM クライアント（OpenAI互換）
# =========================


class OpenAICompatLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 60.0,
        is_vllm: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.is_vllm = (
            is_vllm or "localhost" in base_url
        )  # ローカルサーバーなら vLLM と判定
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
                logger.warning(
                    f"LLM returned empty response | Status: {r.status_code} | "
                    f"User message (first 500 chars): {user[:500]}..."
                )
                logger.debug(f"Full API response: {data}")
                raise ValueError("LLMが空の応答を返しました")

            try:
                return _robust_json_loads(content)
            except ValueError as e:
                # JSON パースエラーの場合、詳細をログに記録
                logger.error(f"LLM JSON パースエラー: {str(e)}")
                logger.debug(f"LLM raw output (first 3000 chars):\n{content[:3000]}")
                raise


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
    last_vision_summary: Optional[str]

    # 前フレーム、現フレームの安全判断
    last_assessment: Optional[SafetyAssessment]  # フレーム間の引き継ぎ
    assessment: Optional[SafetyAssessment]       # 現フレームの判断

    done: bool
    errors: Annotated[List[str], _sliding_window_errors]


class ContextSchema(TypedDict):
    provider: ObservationProvider
    llm: Optional[OpenAICompatLLM]
    vision_analyzer: Optional[VisionAnalyzer]
    yolo_detector: Optional[YOLODetector]
    audio_analyzer: AudioAnalyzer
    depth_estimator: Optional[DepthEstimator]
    prompts: dict  # プロンプト設定全体
    config: dict
    chat_max_tokens: int
    max_outstanding_regions: int
    context_history_size: int  # LLM に渡す前回結果の数（0=なし, 1=前回のみ）
    expected_modalities: List[str]
    # run_mode: "until_provider_ends" | "stop_when_safe"
    run_mode: str


# =========================
# グラフノード関数
# =========================


def ingest_observation(state: AgentState, runtime: Runtime[ContextSchema]) -> Command:
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

    # fan-out: yolo, vlm, audio（+ depth）を並列ノードへ送信
    sends: list[Send] = [
        Send("yolo_node", {"observation": obs}),
        Send("vlm_node", {"observation": obs}),
        Send("audio_node", {"observation": obs}),
    ]

    # enable_depth=true の場合のみ depth_node を追加
    if "depth" in runtime.context["expected_modalities"]:
        sends.append(Send("depth_node", {"observation": obs}))

    return Command(
        update={
            "observation": obs,
            "done": False,  # 毎フレーム開始時にクリア
            "modality_results": {},  # dict リセット
            "received_modalities": [_RESET_SENTINEL],  # バリアカウンタをリセット
            "barrier_obs_id": None,  # ラッチをリセット
            # 注: assessment と last_assessment はリセットしない
            # 前フレーム結果を determine_next_action_llm で参照するため
            "messages": [
                {"role": "assistant", "content": f"[ingest] fan-out -> {obs.obs_id}"}
            ],
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
            "messages": [
                {"role": "assistant", "content": f"[yolo] objects={len(objects)}"}
            ],
        },
        goto="join_modalities",
    )


def vlm_node(state: AgentState, runtime: Runtime[ContextSchema]) -> Command:
    """VLM ノード：2枚画像を比較分析し構造化 JSON を返す。"""
    obs = state.get("observation")
    analyzer = runtime.context.get("vision_analyzer")
    config = runtime.context.get("config", {})
    vision_analysis = None
    error = None

    if obs and obs.image_path and analyzer:
        try:
            # Vision API のトークン上限を設定から取得
            vision_max_tokens = config.get("tokens", {}).get("vision_max_completion_tokens", 4096)

            vision_analysis = analyzer.analyze(
                image_path=obs.image_path,
                prev_image_path=obs.prev_image_path,
                max_tokens=vision_max_tokens,
            )
        except Exception as e:
            error = f"vlm: {e}"

    result = ModalityResult(
        modality_name="vlm",
        extra={"vision_analysis": vision_analysis},
        error=error,
    )
    return Command(
        update={
            "modality_results": {"vlm": result},
            "received_modalities": ["vlm"],
            "messages": [
                {
                    "role": "assistant",
                    "content": f"[vlm] {'ok' if vision_analysis else 'none'}",
                }
            ],
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
            audio_cfg = runtime.context.get("config", {}).get("audio", {})
            audio_cues = audio_analyzer.analyze(
                audio_input=obs.audio_path or obs.audio_text,
                video_timestamp=obs.video_timestamp,
                previous_vision_summary=state.get("last_vision_summary"),
                window_seconds=audio_cfg.get("window_seconds", 3.0),
            )
        except Exception as e:
            error = f"audio: {e}"

    result = ModalityResult(modality_name="audio", audio_cues=audio_cues, error=error)
    return Command(
        update={
            "modality_results": {"audio": result},
            "received_modalities": ["audio"],
            "messages": [
                {"role": "assistant", "content": f"[audio] cues={len(audio_cues)}"}
            ],
        },
        goto="join_modalities",
    )


def depth_node(state: AgentState, runtime: Runtime[ContextSchema]) -> Command:
    """深度推定ノード：深度Anything3で推定し、VLMで分析、PNG出力。

    1. DepthEstimator.estimate() で side-by-side PNG バイト列を取得
    2. PNG をファイルに保存（data/depth/depth_Xs.png）
    3. VisionAnalyzer.analyze_bytes_raw() で VLM 分析
    4. DepthAnalysisResult に型変換してモダリティ結果に格納
    """
    obs = state.get("observation")
    error = None
    depth_analysis = None
    depth_image_path = None

    depth_estimator = runtime.context.get("depth_estimator")
    vision_analyzer = runtime.context.get("vision_analyzer")
    prompts = runtime.context.get("prompts", {})
    config = runtime.context.get("config", {})

    if depth_estimator and vision_analyzer and obs and obs.image_path:
        try:
            # ステップ1: 深度推定 + side-by-side PNG
            side_by_side_bytes = depth_estimator.estimate(obs.image_path)
            if side_by_side_bytes is None:
                error = "depth: depth_estimator returned None"
            else:
                # ステップ1.5: side-by-side 画像をファイルに保存
                try:
                    depth_output_dir = Path("data/depth")
                    depth_output_dir.mkdir(parents=True, exist_ok=True)

                    # ファイル名を frame と完全に同じにする（例：frame_0s.jpg → frame_0s.jpg）
                    frame_filename = Path(obs.image_path).name  # "frame_0s.jpg"
                    depth_filename = frame_filename
                    depth_image_path = str(depth_output_dir / depth_filename)

                    with open(depth_image_path, "wb") as f:
                        f.write(side_by_side_bytes)
                    logger.debug(f"Depth image saved to {depth_image_path}")
                except Exception as e:
                    logger.warning(f"Failed to save depth image: {e}")
                    depth_image_path = None

                # ステップ2: VLM で深度画像を分析
                depth_prompt = prompts.get("depth_analysis", {}).get("system")
                if not depth_prompt:
                    logger.debug("depth_analysis.system プロンプトが見つかりません。デフォルトプロンプトを使用します。")
                    depth_prompt = (
                        "この深度推定画像を分析して、空間的な危険性を評価してください。"
                    )

                # Vision API のトークン上限を設定から取得
                vision_max_tokens = config.get("tokens", {}).get("vision_max_completion_tokens", 4096)

                raw_result = vision_analyzer.analyze_bytes_raw(
                    side_by_side_bytes,
                    media_type="image/png",
                    prompt=depth_prompt,
                    max_tokens=vision_max_tokens,
                )

                if raw_result is None:
                    error = "depth: VLM analysis failed"
                else:
                    # ステップ3: JSON を DepthAnalysisResult に型変換
                    try:
                        depth_analysis = DepthAnalysisResult.model_validate(raw_result)
                        # フォールバック検出: summary が error message の場合
                        if depth_analysis.summary and "could not be parsed" in depth_analysis.summary:
                            logger.debug("Depth analysis returned fallback response (VLM レスポンスが不正)")
                            # fallback response の場合もエラーを記録するが depth_analysis は保持
                    except Exception as e:
                        logger.warning(f"Failed to validate depth analysis result: {e}")
                        error = f"depth: validation error: {e}"
        except Exception as e:
            logger.error(f"Depth node error: {e}")
            error = f"depth: {e}"
    elif not depth_estimator:
        # DepthEstimator が利用不可（enable_depth=false または import失敗）
        error = "depth: estimator not available"
    elif not vision_analyzer:
        error = "depth: vision_analyzer not available"

    result = ModalityResult(
        modality_name="depth",
        extra={"depth_analysis": depth_analysis, "depth_image_path": depth_image_path}
        if depth_analysis
        else {"depth_image_path": depth_image_path},
        error=error,
    )
    return Command(
        update={
            "modality_results": {"depth": result},
            "received_modalities": ["depth"],
            "messages": [
                {
                    "role": "assistant",
                    "content": f"[depth] {'ok' if depth_analysis else 'none'}",
                }
            ],
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
                update={
                    "messages": [
                        {"role": "assistant", "content": "[join] already fused, skip"}
                    ]
                },
                goto=END,
            )
        # 初回: ラッチをセットして fuse へ
        return Command(
            update={
                "barrier_obs_id": current_obs_id,
                "messages": [
                    {"role": "assistant", "content": "[join] all modalities received"}
                ],
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
    depth = results.get("depth")

    objects = yolo.objects if yolo and yolo.objects else []
    audio_cues = audio.audio_cues if audio and audio.audio_cues else []
    vision_analysis = vlm.extra.get("vision_analysis") if vlm else None
    depth_analysis = depth.extra.get("depth_analysis") if depth else None
    modality_errors = [r.error for r in results.values() if r.error]

    # PerceptionIR を作成
    ir = PerceptionIR(
        obs_id=obs.obs_id,
        camera_pose=obs.camera_pose,
        objects=objects,
        audio=audio_cues,
        vision_analysis=vision_analysis,
        depth_analysis=depth_analysis,
        modality_errors=modality_errors,
    )

    return {
        "ir": ir,
        "messages": [
            {
                "role": "assistant",
                "content": "[fuse] ok (perception inference in determine_next_action_llm)",
            }
        ],
    }


def determine_next_action_llm(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    """VLM/YOLO/音声から、次に起こすべき行動を LLM で決定。前フレーム結果を参照。

    フレーム間の状態引き継ぎ：
    - last_assessment: 前フレームの判断（LLM 入力で参照用）
    - assessment: 現フレームで新しく計算した判断

    返却時に両キーに同じ値を入れる理由：
    次フレームでこの値が last_assessment（前フレーム判断）として参照されるため。
    ingest_observation でリセットされないため、フレーム間で正しく引き継がれる。
    """
    ir = state.get("ir")
    if ir is None:
        return {"errors": ["No IR for action determination"], "done": True}

    llm = runtime.context.get("llm")
    if llm is None:
        # ヒューリスティックフォールバック
        assessment = _heuristic_assessment()
        # assessment と last_assessment の両方に同じ値を返す（次フレーム引き継ぎ用）
        return {
            "assessment": assessment,
            "last_assessment": assessment,
            "messages": [
                {
                    "role": "assistant",
                    "content": "[assess] LLM not configured -> heuristic",
                }
            ],
        }

    # コンテキスト構築
    context_history_size = runtime.context.get("context_history_size", 1)

    last_assessment = state.get("last_assessment")
    context_data = {
        "vision_analysis": (
            ir.vision_analysis.model_dump(exclude_none=True, by_alias=True)
            if ir.vision_analysis
            else None
        ),
        "detected_objects": [o.model_dump() for o in ir.objects[:10]],
        "audio_cues": [a.model_dump() for a in ir.audio],
        "previous_assessment": (
            last_assessment.model_dump()
            if context_history_size >= 1 and last_assessment
            else None
        ),
    }

    # prompt.yaml の safety_assessment セクションから取得
    next_action_cfg = runtime.context["prompts"].get("safety_assessment", {})
    system = next_action_cfg.get("system", "").strip()

    try:
        chat_max_tokens = runtime.context["chat_max_tokens"]
        raw = llm.chat_json(
            system=system,
            user=json.dumps(context_data, ensure_ascii=False),
            max_tokens=chat_max_tokens,
        )

        # SafetyAssessment を直接取得
        assessment = SafetyAssessment.model_validate(raw)
        # LLM で計算した judgment を次フレームの前判断として保存
        return {
            "assessment": assessment,
            "last_assessment": assessment,
            "messages": [
                {
                    "role": "assistant",
                    "content": f"[assess] {assessment.action_type} risk={assessment.risk_level} priority={assessment.priority:.2f} (llm)",
                }
            ],
        }

    except Exception as e:
        logger.error("LLM assessment failed, using heuristic fallback", exc_info=True)
        assessment = _heuristic_assessment()
        # フォールバック判断も同様に次フレーム用に保存
        return {
            "assessment": assessment,
            "last_assessment": assessment,
            "errors": [f"LLM assessment fallback: {e}"],
            "messages": [
                {
                    "role": "assistant",
                    "content": f"[assess] LLM failed -> heuristic. err={e}",
                }
            ],
        }


def _heuristic_assessment() -> SafetyAssessment:
    """LLM 不使用時のフォールバック。"""
    return SafetyAssessment(
        risk_level="low",
        safety_status="継続観測中（LLM なし）",
        detected_hazards=[],
        action_type="monitor",
        reason="LLM 未設定のためヒューリスティックで継続監視",
        priority=0.0,
        temporal_status="unknown",
        evidence=None,
    )


def emit_output(state: AgentState) -> Dict[str, Any]:
    """
    フレームごとに latest_output を更新し、統合出力を生成。
    新しいフラット構造で出力（ir をフラット展開）。
    """
    obs = state.get("observation")
    ir = state.get("ir")
    assessment = state.get("assessment")

    # ir からエラーと不要フィールドを除去
    ir_dump = ir.model_dump(exclude_none=True) if ir else {}
    modality_errors = ir_dump.pop("modality_errors", [])

    errors = list(state.get("errors", [])) + modality_errors

    output = {
        "frame_id": obs.obs_id if obs else None,
        "timestamp": None,  # save_analysis_results() で付与
        "video_timestamp": obs.video_timestamp if obs else None,
        "vision_analysis": (
            ir.vision_analysis.model_dump(exclude_none=True, by_alias=True)
            if ir and ir.vision_analysis
            else None
        ),
        "depth_analysis": (
            ir.depth_analysis.model_dump(exclude_none=True)
            if ir and ir.depth_analysis
            else None
        ),
        "objects": ir_dump.get("objects", []),
        "audio": ir_dump.get("audio", []),
        "assessment": assessment.model_dump(exclude_none=True) if assessment else None,
        "errors": errors,
    }

    last_vision_summary = (
        ir.vision_analysis.summary
        if ir and ir.vision_analysis and ir.vision_analysis.summary
        else state.get("last_vision_summary")
    )

    return {
        "latest_output": output,
        "last_vision_summary": last_vision_summary,
        "messages": [
            {
                "role": "assistant",
                "content": f"[emit] frame={obs.obs_id if obs else 'none'}",
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

    # ノード登録（yolo/vlm/audio/depth に分割した fan-out/fan-in 対応）
    builder.add_node("ingest_observation", ingest_observation)
    builder.add_node("yolo_node", yolo_node)
    builder.add_node("vlm_node", vlm_node)
    builder.add_node("audio_node", audio_node)
    builder.add_node("depth_node", depth_node)
    builder.add_node("join_modalities", join_modalities)  # ラッチ付き fan-in バリア
    builder.add_node("fuse_modalities", fuse_modalities)
    builder.add_node("determine_next_action_llm", determine_next_action_llm)
    builder.add_node("emit_output", emit_output)
    builder.add_node("bump_step", bump_step)

    # エッジ設定
    builder.add_edge(START, "ingest_observation")
    # fan-out: yolo/vlm/audio ノードは Command で goto="join_modalities" するため静的エッジは不要
    builder.add_edge(
        "fuse_modalities", "determine_next_action_llm"
    )  # PerceptionIR 生成 → 安全判断
    builder.add_edge("determine_next_action_llm", "emit_output")
    builder.add_edge("emit_output", "bump_step")
    builder.add_conditional_edges("bump_step", should_continue)

    return builder.compile()
