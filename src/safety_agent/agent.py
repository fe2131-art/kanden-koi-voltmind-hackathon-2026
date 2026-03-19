import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import Command, Send
from openai import OpenAI
from typing_extensions import Annotated, TypedDict

from .modality_nodes import (
    AudioAnalyzer,
    DepthEstimator,
    InfraredImageAnalyzer,
    ModalityResult,
    TemporalImageAnalyzer,
    VisionAnalyzer,
)
from .schema import (
    BeliefState,
    DepthAnalysisResult,
    InfraredAnalysisResult,
    Observation,
    ObservationProvider,
    PerceptionIR,
    SafetyAssessment,
    TemporalAnalysisResult,
    get_json_schema,
)

logger = logging.getLogger(__name__)


# =========================
# LLM クライアント（OpenAI互換）
# =========================


class OpenAICompatLLM:
    """OpenAI SDK を使った LLM クライアント。vLLM と OpenAI API の両方に対応。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 60.0,
        is_vllm: bool = False,
    ):
        # vLLM 用に base_url を正規化（/v1 の追加）
        normalized_url = base_url.rstrip("/")
        if is_vllm or "localhost" in base_url:
            if not normalized_url.endswith("/v1"):
                normalized_url = f"{normalized_url}/v1"

        self.client = OpenAI(
            api_key=api_key or "EMPTY",
            base_url=normalized_url,
            timeout=timeout_s,
        )
        self.model = model
        self.is_vllm = is_vllm or "localhost" in base_url
        # GPT-5 系モデルは max_tokens ではなく max_completion_tokens を使用
        self._use_max_completion_tokens = "gpt-5" in model.lower()

    def chat_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 800,
        schema_type: Literal["belief_state", "safety_assessment"] = "safety_assessment",
    ) -> Dict[str, Any]:
        """JSON 出力を期待するチャット API 呼び出し。

        Args:
            system: システムプロンプト
            user: ユーザーメッセージ
            max_tokens: 最大トークン数
            schema_type: "belief_state" または "safety_assessment"

        Returns:
            パースされた JSON 辞書
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # response_format の構築（vLLM と OpenAI で異なる）
        if self.is_vllm:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": f"{schema_type}_output",
                    "schema": get_json_schema(schema_type),
                    "strict": True,
                },
            }
        else:
            # OpenAI（gpt-5-nano はサポートしない可能性）
            response_format = {"type": "json_object"}

        try:
            # gpt-5 系は max_completion_tokens を使用
            if self._use_max_completion_tokens:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    response_format=response_format,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
        except Exception as e:
            # response_format なしで再試行
            logger.warning(
                f"LLM with response_format failed: {e}. Retrying without format."
            )
            if self._use_max_completion_tokens:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                )

        content = response.choices[0].message.content

        # デバッグ: 空応答を確認
        if not content or content.strip() == "":
            logger.warning(
                f"LLM returned empty response | User message (first 500 chars): {user[:500]}..."
            )
            raise ValueError("LLMが空の応答を返しました")

        try:
            return _robust_json_loads(content)
        except ValueError as e:
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


def _first_write_wins(left: Optional[str], right: Optional[str]) -> Optional[str]:
    """一度セットされた obs_id は上書きしない（ラッチ）。
    left が None のときのみ right を採用する。
    None リセット（ingest_observation が barrier_obs_id=None を送信）も正常に機能する:
      left=None, right=None → None を返す
    """
    if left is not None:
        return left
    return right


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


def _append_assessment(
    left: List["SafetyAssessment"], right: List["SafetyAssessment"]
) -> List["SafetyAssessment"]:
    """assessment を末尾に追記する reducer。"""
    return (left or []) + (right or [])


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

    # ラッチ：同フレーム内で fuse_modalities が2回以上実行されるのを防止（first-writer-wins reducer）
    barrier_obs_id: Annotated[Optional[str], _first_write_wins]

    latest_output: Optional[Dict[str, Any]]
    last_vision_summary: Optional[str]

    # 現フレームの安全判断
    assessment: Optional[SafetyAssessment]  # 現フレームの判断
    assessment_history: Annotated[List[SafetyAssessment], _append_assessment]  # 判断履歴

    # 信念状態：危険のトラッキング（フレーム間の引き継ぎ）
    belief_state: Optional[BeliefState]

    done: bool
    errors: Annotated[List[str], _sliding_window_errors]


class ContextSchema(TypedDict):
    provider: ObservationProvider
    llm: Optional[OpenAICompatLLM]
    vision_analyzer: Optional[VisionAnalyzer]
    audio_analyzer: AudioAnalyzer
    depth_estimator: Optional[DepthEstimator]
    infrared_analyzer: Optional[InfraredImageAnalyzer]
    temporal_analyzer: Optional[TemporalImageAnalyzer]
    prompts: dict  # プロンプト設定全体
    config: dict
    chat_max_tokens: int
    context_history_size: int  # LLM に渡す前回判断の数（0=なし, 1=直近1フレーム, N=直近Nフレーム）
    expected_modalities: List[str]
    # run_mode: "until_provider_ends" | "stop_when_safe"
    run_mode: str


# =========================
# グラフノード関数
# =========================


def ingest_observation(state: AgentState, runtime: Runtime[ContextSchema]) -> Command:
    """
    観測を取得して、vlm_node、audio_node へ fan-out で送信。
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

    # fan-out: vlm, audio（+ depth）を並列ノードへ送信
    sends: list[Send] = [
        Send("vlm_node", {"observation": obs}),
        Send("audio_node", {"observation": obs}),
    ]

    # enable_depth=true の場合のみ depth_node を追加
    if "depth" in runtime.context["expected_modalities"]:
        sends.append(Send("depth_node", {"observation": obs}))

    # enable_infrared=true の場合のみ infrared_node を追加
    if "infrared" in runtime.context["expected_modalities"]:
        sends.append(Send("infrared_node", {"observation": obs}))

    # enable_temporal=true の場合のみ temporal_node を追加
    if "temporal" in runtime.context["expected_modalities"]:
        sends.append(Send("temporal_node", {"observation": obs}))

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
            vision_max_tokens = config.get("tokens", {}).get(
                "vision_max_completion_tokens", 4096
            )

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
                    logger.debug(
                        "depth_analysis.system プロンプトが見つかりません。デフォルトプロンプトを使用します。"
                    )
                    depth_prompt = (
                        "この深度推定画像を分析して、空間的な危険性を評価してください。"
                    )

                # Vision API のトークン上限を設定から取得
                vision_max_tokens = config.get("tokens", {}).get(
                    "vision_max_completion_tokens", 4096
                )

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


def infrared_node(state: AgentState, runtime: Runtime[ContextSchema]) -> Command:
    """赤外線画像ノード：RGB と赤外線フレームを side-by-side 結合して VLM で分析。"""
    obs = state.get("observation")
    error = None
    infrared_analysis = None

    infrared_analyzer = runtime.context.get("infrared_analyzer")
    vision_analyzer = runtime.context.get("vision_analyzer")
    prompts = runtime.context.get("prompts", {})
    config = runtime.context.get("config", {})

    if (
        infrared_analyzer
        and vision_analyzer
        and obs
        and obs.image_path
        and obs.infrared_image_path
    ):
        try:
            side_by_side_bytes = infrared_analyzer.make_side_by_side_bytes(
                obs.image_path, obs.infrared_image_path
            )
            if side_by_side_bytes is None:
                error = "infrared: failed to create side-by-side image"
            else:
                infrared_prompt = prompts.get("infrared_analysis", {}).get(
                    "system"
                ) or (
                    "左が可視光カメラ画像、右が赤外線画像です。"
                    "異常箇所・高温箇所・火災リスクを JSON で出力してください。"
                    '{"hot_spots": [{"region_id": "infrared_hotspot_0", "description": "...", "severity": "unknown"}], "overall_risk": "unknown", "confidence_score": 0.0}'
                )
                vision_max_tokens = config.get("tokens", {}).get(
                    "vision_max_completion_tokens", 4096
                )
                raw_result = vision_analyzer.analyze_bytes_raw(
                    side_by_side_bytes,
                    media_type="image/png",
                    prompt=infrared_prompt,
                    max_tokens=vision_max_tokens,
                )
                if raw_result is None:
                    error = "infrared: VLM analysis failed"
                else:
                    try:
                        infrared_analysis = InfraredAnalysisResult.model_validate(
                            raw_result
                        )
                    except Exception as e:
                        error = f"infrared: validation error: {e}"
        except Exception as e:
            error = f"infrared: {e}"
    elif not infrared_analyzer:
        error = "infrared: analyzer not available"
    elif not vision_analyzer:
        error = "infrared: vision_analyzer not available"
    elif not obs or not obs.image_path or not obs.infrared_image_path:
        error = "infrared: RGB or infrared image path not available"

    result = ModalityResult(
        modality_name="infrared",
        extra={"infrared_analysis": infrared_analysis} if infrared_analysis else {},
        error=error,
    )
    return Command(
        update={
            "modality_results": {"infrared": result},
            "received_modalities": ["infrared"],
            "messages": [
                {
                    "role": "assistant",
                    "content": f"[infrared] {'ok' if infrared_analysis else 'none'}",
                }
            ],
        },
        goto="join_modalities",
    )


def temporal_node(state: AgentState, runtime: Runtime[ContextSchema]) -> Command:
    """時系列変化検出ノード：現フレームと前フレームを横並び結合して VLM で変化を分析。"""
    obs = state.get("observation")
    error = None
    temporal_analysis = None

    temporal_analyzer = runtime.context.get("temporal_analyzer")
    vision_analyzer = runtime.context.get("vision_analyzer")
    prompts = runtime.context.get("prompts", {})
    config = runtime.context.get("config", {})

    if (
        temporal_analyzer
        and vision_analyzer
        and obs
        and obs.image_path
        and obs.prev_image_path
    ):
        try:
            temporal_bytes = temporal_analyzer.make_temporal_bytes(
                obs.image_path, obs.prev_image_path
            )
            if temporal_bytes is None:
                error = "temporal: failed to create temporal image"
            else:
                temporal_prompt = prompts.get("temporal_analysis", {}).get(
                    "system"
                ) or (
                    "左が前フレーム、右が現フレームです。"
                    "2フレーム間の変化を分析し、JSON で出力してください。"
                    '{"change_detected": false, "changes": [{"region_id": "temporal_change_0", "description": "...", "severity": "unknown"}], "overall_risk": "unknown", "confidence_score": 0.0}'
                )
                vision_max_tokens = config.get("tokens", {}).get(
                    "vision_max_completion_tokens", 4096
                )
                raw_result = vision_analyzer.analyze_bytes_raw(
                    temporal_bytes,
                    media_type="image/png",
                    prompt=temporal_prompt,
                    max_tokens=vision_max_tokens,
                )
                if raw_result is None:
                    error = "temporal: VLM analysis failed"
                else:
                    try:
                        temporal_analysis = TemporalAnalysisResult.model_validate(
                            raw_result
                        )
                    except Exception as e:
                        error = f"temporal: validation error: {e}"
        except Exception as e:
            error = f"temporal: {e}"
    elif not temporal_analyzer:
        error = "temporal: analyzer not available"
    elif not vision_analyzer:
        error = "temporal: vision_analyzer not available"
    elif not obs or not obs.image_path or not obs.prev_image_path:
        error = "temporal: current or previous image path not available"

    result = ModalityResult(
        modality_name="temporal",
        extra={"temporal_analysis": temporal_analysis} if temporal_analysis else {},
        error=error,
    )
    return Command(
        update={
            "modality_results": {"temporal": result},
            "received_modalities": ["temporal"],
            "messages": [
                {
                    "role": "assistant",
                    "content": f"[temporal] {'ok' if temporal_analysis else 'none'}",
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
    expected = set(runtime.context.get("expected_modalities", ["vlm", "audio"]))
    received = set(state.get("received_modalities", []))
    obs = state.get("observation")
    current_obs_id = obs.obs_id if obs else None
    barrier_obs_id = state.get("barrier_obs_id")

    if expected.issubset(received):
        # 全てのモダリティが揃った
        if current_obs_id is not None and barrier_obs_id == current_obs_id:
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
    fan-in ポイント：vlm_node、audio_node の完了後に実行。
    """
    obs = state.get("observation")
    if obs is None:
        return {"errors": ["No observation in state"], "done": True}

    # modality_results は既に dict
    results: Dict[str, ModalityResult] = state.get("modality_results", {})

    vlm = results.get("vlm")
    audio = results.get("audio")
    depth = results.get("depth")
    infrared = results.get("infrared")
    temporal = results.get("temporal")

    audio_cues = audio.audio_cues if audio and audio.audio_cues else []
    vision_analysis = vlm.extra.get("vision_analysis") if vlm else None
    depth_analysis = depth.extra.get("depth_analysis") if depth else None
    infrared_analysis = infrared.extra.get("infrared_analysis") if infrared else None
    temporal_analysis = temporal.extra.get("temporal_analysis") if temporal else None
    modality_errors = [r.error for r in results.values() if r.error]

    # PerceptionIR を作成
    ir = PerceptionIR(
        obs_id=obs.obs_id,
        camera_pose=obs.camera_pose,
        audio=audio_cues,
        vision_analysis=vision_analysis,
        depth_analysis=depth_analysis,
        infrared_analysis=infrared_analysis,
        temporal_analysis=temporal_analysis,
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


def update_belief_state_llm(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    """信念状態更新ノード：全モダリティの知覚結果から危険状態を継続管理する。

    危険のトラッキング：
    - previous_belief_state: 前フレームの belief_state（フレーム間の継続性判断用）
    - belief_state: 現フレームで更新した belief_state

    返却時に belief_state キーに新しい値を入れる理由：
    次フレームでこの値が previous_belief_state として参照されるため。
    ingest_observation でリセットされないため、フレーム間で正しく引き継がれる。
    """
    ir = state.get("ir")
    if ir is None:
        # ir が未生成の場合、previous belief_state をそのまま維持
        return {
            "belief_state": state.get("belief_state"),
            "errors": ["No IR in update_belief_state_llm"],
            "messages": [
                {
                    "role": "assistant",
                    "content": "[belief] no IR -> skip",
                }
            ],
        }

    llm = runtime.context.get("llm")
    if llm is None:
        # LLM が未設定の場合、previous belief_state をそのまま維持
        return {
            "belief_state": state.get("belief_state"),
            "messages": [
                {
                    "role": "assistant",
                    "content": "[belief] LLM not configured -> skip",
                }
            ],
        }

    # コンテキスト構築
    previous_belief_state = state.get("belief_state")
    context_data = {
        "vision_analysis": (
            ir.vision_analysis.model_dump(exclude_none=True, by_alias=True)
            if ir.vision_analysis
            else None
        ),
        "depth_analysis": (
            ir.depth_analysis.model_dump(exclude_none=True)
            if ir.depth_analysis
            else None
        ),
        "infrared_analysis": (
            ir.infrared_analysis.model_dump(exclude_none=True)
            if ir.infrared_analysis
            else None
        ),
        "temporal_analysis": (
            ir.temporal_analysis.model_dump(exclude_none=True)
            if ir.temporal_analysis
            else None
        ),
        "audio_cues": [a.model_dump() for a in ir.audio],
        "previous_belief_state": (
            previous_belief_state.model_dump(exclude_none=True)
            if previous_belief_state
            else None
        ),
        "step": state["step"],
    }

    # prompt.yaml の belief_update セクションから取得
    belief_cfg = runtime.context["prompts"].get("belief_update", {})
    system = belief_cfg.get("system", "").strip()

    try:
        chat_max_tokens = runtime.context["chat_max_tokens"]
        # chat_json() 経由で vLLM/OpenAI 両対応、retry ロジック搭載
        raw = llm.chat_json(
            system=system,
            user=json.dumps(context_data, ensure_ascii=False),
            max_tokens=chat_max_tokens,
            schema_type="belief_state",
        )
        # BeliefState を直接検証
        belief_state = BeliefState.model_validate(raw)
        # 新しい belief_state を次フレームに引き継ぎ
        return {
            "belief_state": belief_state,
            "messages": [
                {
                    "role": "assistant",
                    "content": f"[belief] ok hazards={len(belief_state.hazard_tracks)} risk={belief_state.overall_risk}",
                }
            ],
        }

    except Exception as e:
        logger.error(
            "Belief state update failed, keeping previous belief_state", exc_info=True
        )
        error_summary = f"{type(e).__name__}: {str(e)[:150]}"
        return {
            "belief_state": previous_belief_state,
            "errors": [f"Belief state update failed: {error_summary}"],
            "messages": [
                {
                    "role": "assistant",
                    "content": "[belief] failed -> keep previous",
                }
            ],
        }


def determine_next_action_llm(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    """VLM/音声から、次に起こすべき行動を LLM で決定。belief_state と前フレーム結果を参照。

    フレーム間の状態引き継ぎ：
    - last_assessment: 前フレームの判断（LLM 入力で参照用）
    - assessment: 現フレームで新しく計算した判断

    返却時に両キーに同じ値を入れる理由：
    次フレームでこの値が last_assessment（前フレーム判断）として参照されるため。
    ingest_observation でリセットされないため、フレーム間で正しく引き継がれる。
    """
    ir = state.get("ir")
    if ir is None:
        logger.warning(
            "IR is None in determine_next_action_llm, returning default assessment"
        )
        assessment = SafetyAssessment(
            risk_level="low",
            safety_status="継続観測中",
            detected_hazards=[],
            action_type="monitor",
            reason="IR 未生成のため継続監視",
            priority=0.0,
            temporal_status="unknown",
            confidence_score=0.0,
        )
        return {
            "assessment": assessment,
            "assessment_history": [assessment],
            "errors": ["No IR, used default assessment"],
            "messages": [
                {
                    "role": "assistant",
                    "content": "[assess] no IR -> default",
                }
            ],
        }

    llm = runtime.context.get("llm")
    if llm is None:
        logger.warning("LLM not configured, returning default assessment")
        assessment = SafetyAssessment(
            risk_level="low",
            safety_status="継続観測中",
            detected_hazards=[],
            action_type="monitor",
            reason="LLM 未設定のため継続監視",
            priority=0.0,
            temporal_status="unknown",
            confidence_score=0.0,
        )
        return {
            "assessment": assessment,
            "assessment_history": [assessment],
            "messages": [
                {
                    "role": "assistant",
                    "content": "[assess] LLM not configured -> default",
                }
            ],
        }

    # コンテキスト構築
    belief_state = state.get("belief_state")
    # context_history_size に基づき前回判断をスライシング
    context_history_size = runtime.context.get("context_history_size", 0)
    assessment_history = state.get("assessment_history", [])

    context_data = {
        "vision_analysis": (
            ir.vision_analysis.model_dump(exclude_none=True, by_alias=True)
            if ir.vision_analysis
            else None
        ),
        "depth_analysis": (
            ir.depth_analysis.model_dump(exclude_none=True)
            if ir.depth_analysis
            else None
        ),
        "infrared_analysis": (
            ir.infrared_analysis.model_dump(exclude_none=True)
            if ir.infrared_analysis
            else None
        ),
        "temporal_analysis": (
            ir.temporal_analysis.model_dump(exclude_none=True)
            if ir.temporal_analysis
            else None
        ),
        "audio_cues": [a.model_dump() for a in ir.audio],
        "belief_state": (
            belief_state.model_dump(exclude_none=True) if belief_state else None
        ),
    }

    # context_history_size > 0 のときのみ previous_assessments を追加（キー自体を除外）
    if context_history_size > 0 and assessment_history:
        context_data["previous_assessments"] = [
            a.model_dump() for a in assessment_history[-context_history_size:]
        ]

    # prompt.yaml の safety_assessment セクションから取得
    next_action_cfg = runtime.context["prompts"].get("safety_assessment", {})
    system = next_action_cfg.get("system", "").strip()

    try:
        chat_max_tokens = runtime.context["chat_max_tokens"]
        raw = llm.chat_json(
            system=system,
            user=json.dumps(context_data, ensure_ascii=False),
            max_tokens=chat_max_tokens,
            schema_type="safety_assessment",
        )

        # SafetyAssessment を直接取得
        assessment = SafetyAssessment.model_validate(raw)
        return {
            "assessment": assessment,
            "assessment_history": [assessment],
            "messages": [
                {
                    "role": "assistant",
                    "content": f"[assess] {assessment.action_type} risk={assessment.risk_level} priority={assessment.priority:.2f} (llm)",
                }
            ],
        }

    except Exception as e:
        logger.error(
            "LLM assessment failed, returning default assessment", exc_info=True
        )
        assessment = SafetyAssessment(
            risk_level="low",
            safety_status="継続観測中",
            detected_hazards=[],
            action_type="monitor",
            reason="LLM 失敗のため継続監視",
            priority=0.0,
            temporal_status="unknown",
            confidence_score=0.0,
        )
        # エラーメッセージを最初の 150 文字に制限（JSON パースエラー等が長いため）
        error_summary = f"{type(e).__name__}: {str(e)[:150]}"
        return {
            "assessment": assessment,
            "assessment_history": [assessment],
            "errors": [f"LLM assessment failed: {error_summary}"],
            "messages": [
                {
                    "role": "assistant",
                    "content": "[assess] LLM failed -> default",
                }
            ],
        }


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

    belief_state = state.get("belief_state")
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
        "infrared_analysis": (
            ir.infrared_analysis.model_dump(exclude_none=True)
            if ir and ir.infrared_analysis
            else None
        ),
        "temporal_analysis": (
            ir.temporal_analysis.model_dump(exclude_none=True)
            if ir and ir.temporal_analysis
            else None
        ),
        "audio": ir_dump.get("audio", []),
        "belief_state": (
            belief_state.model_dump(exclude_none=True) if belief_state else None
        ),
        "assessment": assessment.model_dump(exclude_none=True) if assessment else None,
        "errors": errors,
    }

    last_vision_summary = (
        ir.vision_analysis.scene_description
        if ir and ir.vision_analysis and ir.vision_analysis.scene_description
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

    # ノード登録（vlm/audio/depth に分割した fan-out/fan-in 対応）
    builder.add_node("ingest_observation", ingest_observation)
    builder.add_node("vlm_node", vlm_node)
    builder.add_node("audio_node", audio_node)
    builder.add_node("depth_node", depth_node)
    builder.add_node("infrared_node", infrared_node)
    builder.add_node("temporal_node", temporal_node)
    builder.add_node("join_modalities", join_modalities)  # ラッチ付き fan-in バリア
    builder.add_node("fuse_modalities", fuse_modalities)
    builder.add_node("update_belief_state_llm", update_belief_state_llm)
    builder.add_node("determine_next_action_llm", determine_next_action_llm)
    builder.add_node("emit_output", emit_output)
    builder.add_node("bump_step", bump_step)

    # エッジ設定
    builder.add_edge(START, "ingest_observation")
    # fan-out: vlm/audio ノードは Command で goto="join_modalities" するため静的エッジは不要
    builder.add_edge("fuse_modalities", "update_belief_state_llm")  # 信念状態更新
    builder.add_edge(
        "update_belief_state_llm", "determine_next_action_llm"
    )  # 信念状態 → 安全判断
    builder.add_edge("determine_next_action_llm", "emit_output")
    builder.add_edge("emit_output", "bump_step")
    builder.add_conditional_edges("bump_step", should_continue)

    return builder.compile()
