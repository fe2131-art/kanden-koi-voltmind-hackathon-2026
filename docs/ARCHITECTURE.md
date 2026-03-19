# Safety View Agent - システムアーキテクチャ

Safety View Agent は、LangGraph を使用したマルチモーダル安全支援システムです。複雑なロボット環境での安全性を動的に評価し、次に観測すべき視点を提案します。

## 全体構成

**LangGraph の fan-out/fan-in パイプライン** で複数モダリティを並列処理します：

```
Observation Input (image + audio + infrared)
         ↓
[ingest_observation] (観測データ取得・リセット)
    ↓ (fan-out: Command + Send API)
    ├─→ [vlm_node]           (画像分析: OpenAI Vision API)
    ├─→ [audio_node]         (音声解析: キーワード抽出)
    ├─→ [depth_node]         (深度推定: Depth Anything 3)
    ├─→ [infrared_node]      (赤外線分析: VLM による熱検出)
    └─→ [temporal_node]      (時系列変化: 前後フレーム比較)
    ↓ (各ノードが join_modalities へ)
[join_modalities] (fan-in バリア + ラッチ)
    ├─ 全モダリティ揃うまで待機
    └─ barrier_obs_id で重複実行を防止
    ↓
[fuse_modalities] (モダリティ統合)
    ├─ PerceptionIR 生成（vision_analysis, audio, depth_analysis, infrared_analysis, temporal_analysis）
    └─ モダリティエラーをまとめて記録
    ↓
[update_belief_state_llm] (信念状態更新 + BeliefState 生成)
    ├─ LLM あり: 前フレーム BeliefState + 新知覚情報から hazard_tracks を更新
    └─ LLM なし: 空の BeliefState を返す
    ↓
[determine_next_action_llm] (総合安全判断)
    ├─ LLM あり: BeliefState + PerceptionIR から SafetyAssessment を生成
    └─ LLM なし / エラー時: 固定値 SafetyAssessment（risk_level=low, action_type=monitor）を返す
    ↓
[emit_output] (フレーム出力)
    └─ latest_output に集約 → streaming 出力（フラット構造）
    ↓
[bump_step] (ステップ更新)
    ↓
(should_continue で分岐)
 ├─ step < max_steps → ingest_observation へループ
 └─ END
```

## コアコンポーネント

### 1. LangGraph グラフ（`src/safety_agent/agent.py`）

**fan-out/fan-in パイプライン**：
- **fan-out**: `Command(update={...}, goto=[Send(...), Send(...), Send(...)])`で 3 つのモダリティに並列送信
- **join_modalities** (fan-in バリア):
  - `expected_modalities` が全て揃うまで待機
  - `barrier_obs_id` で同フレーム内の重複実行を防止（ラッチ機構）
- **fuse_modalities**: `modality_results` 辞書から全結果を統合して PerceptionIR を生成
- **emit_output**: 各フレーム処理結果を `latest_output` に格納（streaming 対応）

### 2. モダリティ処理（`src/safety_agent/modality_nodes.py`）

5 つのモダリティを独立したクラスで処理（各 node として LangGraph に登録）：

- **VisionAnalyzer**: OpenAI Vision API で RGB 画像分析
  - OpenAI SDK を使用（httpx ではなく OpenAI 公式 SDK）
  - VisionAnalysisResult（scene_description, critical_points, blind_spots）返却
  - ノード: `vlm_node()` → PerceptionIR.vision_analysis に格納

- **AudioAnalyzer**: 音声テキストからキュー抽出
  - 実装: キーワードマッチング（Speech-to-Text 未実装）
  - `enable_audio: false` で無効化（デフォルト）
  - ノード: `audio_node()` → AudioCue リスト返却

- **DepthEstimator**: Depth Anything 3 による深度推定
  - `enable_depth: false` で無効化（デフォルト）
  - DepthAnalysisResult（depth_layers, overall_risk）返却
  - ノード: `depth_node()` → PerceptionIR.depth_analysis に格納

- **InfraredImageAnalyzer**: 赤外線フレーム分析（VLM）
  - 赤外線フレームから高温箇所を検出
  - InfraredAnalysisResult（hot_spots, overall_risk）返却
  - ノード: `infrared_node()` → PerceptionIR.infrared_analysis に格納

- **TemporalImageAnalyzer**: 時系列変化検出（前後フレーム比較）
  - 前フレーム（prev_image_path）と現フレームを比較
  - TemporalAnalysisResult（change_detected, changes）返却
  - ノード: `temporal_node()` → PerceptionIR.temporal_analysis に格納

### 3. Pydantic スキーマ（`src/safety_agent/schema.py`）

状態管理スキーマ：

**知覚統合結果**:
- `PerceptionIR`: フレーム単位の知覚統合結果（vision_analysis, audio, depth_analysis, infrared_analysis, temporal_analysis）
- `VisionAnalysisResult`: VLM の出力（scene_description, critical_points, blind_spots, overall_risk）
- `DepthAnalysisResult`: 深度推定結果（depth_layers, overall_risk）
- `InfraredAnalysisResult`: 赤外線分析結果（hot_spots, overall_risk）
- `TemporalAnalysisResult`: 時系列変化（change_detected, changes, overall_risk）

**信念状態**:
- `BeliefState`: 時系列で管理される危険状態（hazard_tracks, overall_risk, recommended_focus_regions）
- `HazardTrack`: 継続中の個別危険（hazard_id, hazard_type, status, severity, confidence_score）

**安全判断**:
- `SafetyAssessment`: LLM による総合安全判断（risk_level, safety_status, detected_hazards, action_type, target_region, reason, priority, temporal_status）

**入力**:
- `Observation`: 入力観測データ（image_path, prev_image_path, audio_path, infrared_image_path, camera_pose, video_timestamp）

## データフロー

### 1フレーム処理

```
Input: Observation (image_path + prev_image_path + audio_path + infrared_image_path)
  ↓
1. ingest_observation
   - provider から obs を取得
   - modality_results, received_modalities をリセット
   - barrier_obs_id, latest_output をリセット
   - vlm_node, audio_node, depth_node, infrared_node, temporal_node に Command + Send で並列送信
  ↓
2. vlm_node, audio_node, depth_node, infrared_node, temporal_node (真の並列実行)
   a) vlm_node (常に実行)
      - RGB フレームを OpenAI Vision API で分析
      - VisionAnalysisResult（scene_description, critical_points, blind_spots, overall_risk）返却
   b) audio_node (enable_audio=true の場合)
      - audio_path または audio_text からキーワード抽出
      - AudioCue リスト返却
   c) depth_node (enable_depth=true の場合)
      - RGB フレームから深度推定（Depth Anything 3）
      - DepthAnalysisResult（depth_layers, overall_risk）返却
   d) infrared_node (infrared_image_path がある場合)
      - 赤外線フレームを VLM で分析
      - InfraredAnalysisResult（hot_spots, overall_risk）返却
   e) temporal_node (prev_image_path がある場合)
      - 前後フレームを比較して変化検出
      - TemporalAnalysisResult（change_detected, changes）返却
   ↓ （各ノードが join_modalities へ）
3. join_modalities (fan-in バリア)
   - expected_modalities がすべて received_modalities に揃うまで待機
   - barrier_obs_id で同フレーム内の重複実行を防止
   - 全て揃ったら fuse_modalities へ遷移
  ↓
4. fuse_modalities
   - modality_results から vlm, audio, depth, infrared, temporal を取得
   - PerceptionIR（vision_analysis, audio, depth_analysis, infrared_analysis, temporal_analysis）生成
   - モダリティエラーをまとめて記録
  ↓
5. update_belief_state_llm
   a) LLM あり:
      - 前フレーム BeliefState + PerceptionIR から hazard_tracks を更新
      - status: new/persistent/worsening/improving/resolved を判定
      - BeliefState 生成
   b) LLM なし:
      - 空の BeliefState（hazard_tracks=[]）を返す
  ↓
6. determine_next_action_llm
   a) LLM あり:
      - BeliefState + PerceptionIR から SafetyAssessment 生成
      - risk_level: high/medium/low
      - action_type: emergency_stop/inspect_region/mitigate/monitor
      - temporal_status: new/persistent/worsening/improving/resolved/unknown
   b) LLM なし:
      - 固定値 SafetyAssessment（risk_level=low, action_type=monitor）を返す
  ↓
7. emit_output
   - フレーム出力（フラット構造）
   - frame_id, vision_analysis, depth_analysis, infrared_analysis, temporal_analysis, audio, belief_state, assessment, errors
   - streaming で即座に出力可能
  ↓
8. bump_step
   - step += 1
   - should_continue で分岐判定
   ├─ step < max_steps → ingest_observation へループ
   └─ END
```

## LLM フォールバック＆ Structured Outputs

### LLM 初期化

```
LLM = get_llm(config)
  ├─ OPENAI_API_KEY + "openai" → OpenAI API（json_object フォーマット）
  ├─ config の llm.vllm.base_url + "vllm" → ローカルサーバー（Structured Outputs で JSON スキーマ指定）
  └─ なし → None
```

### JSON 生成の安定化

**vLLM（推奨）**: Structured Outputs で JSON スキーマを指定
- `response_format` に `json_schema` 型で SafetyAssessment / BeliefState スキーマを指定
- LLM が厳密にスキーマに従う出力を生成（フォーマット違反なし）
- JSON パースエラーを事実上排除

**OpenAI API**: json_object フォーマット + プロンプト指示
- `response_format: {"type": "json_object"}` を指定
- プロンプトで JSON 形式を明確に指示
- `_robust_json_loads()` で複数フォーマットに対応

### BeliefState + SafetyAssessment 生成フロー

```
update_belief_state_llm:
  ├─ llm is not None:
  │  ├─ vLLM → Structured Outputs で確実に JSON 取得（BeliefState スキーマ）
  │  └─ OpenAI → json_object フォーマット + 抽出
  │      → BeliefState 検証 → belief_state に格納
  └─ llm is None → 空の BeliefState（hazard_tracks=[]）

determine_next_action_llm:
  ├─ llm is not None:
  │  ├─ vLLM → Structured Outputs で確実に JSON 取得（SafetyAssessment スキーマ）
  │  └─ OpenAI → json_object フォーマット + 抽出
  │      → SafetyAssessment 検証 → assessment に格納
  └─ llm is None → 固定値 SafetyAssessment（risk_level=low, action_type=monitor）
```

### ヒューリスティック判定

LLM 未設定時のフォールバック：
- risk_level: "low"（安全側を取る）
- action_type: "monitor"（継続観測）
- temporal_status: "unknown"（前回フレーム情報なし）
- 理由: 継続的な観測により安全性を確保

## 設定管理

### `configs/default.yaml`

```yaml
agent:
  max_steps: -1                   # フレーム処理数（-1: 全フレーム）
  enable_audio: true              # 音声解析の有効/無効
  enable_depth: true              # 深度推定の有効/無効
  enable_infrared: true           # 赤外線解析の有効/無効
  enable_temporal: true           # 時系列解析の有効/無効
  context_history_size: 0         # LLM に渡す前回判断の数（0=なし）

llm:
  provider: "vllm"                # "openai" or "vllm"

vlm:
  provider: "vllm"                # VLM プロバイダー

tokens:
  vision_max_completion_tokens: 8192   # VLM の最大トークン
  chat_max_tokens: 8192                # テキスト LLM の最大トークン
```

### `configs/prompt.yaml` (外部化)

```yaml
vision_analysis:
  default_prompt: |
    この画像の安全性を分析してください...

safety_assessment:
  system: |
    あなたは安全支援エージェントです...
    （知覚推論 + 安全判断を統合的に実行）
```

### エラーハンドリング

- **ファイル未検出**: 設定ファイル/プロンプトファイル欠落 → `FileNotFoundError` を raise
- **フレーム未検出**: manual モードで `data/videos/` に動画も `data/frames/` に画像もない → `FileNotFoundError` を raise
- **LLM 未設定**: `llm=None` → 固定値の `SafetyAssessment` / 既存 `BeliefState` でフォールバック
- **LLM 実行エラー**: JSON パース失敗など → 固定値の `SafetyAssessment` にフォールバック

## 並列化の仕組み

### fan-out: 複数ノードへ同時送信

```python
# ingest_observation() から
sends = [
    Send("vlm_node", {"observation": obs}),
    Send("audio_node", {"observation": obs}),
    Send("depth_node", {"observation": obs}),
    Send("infrared_node", {"observation": obs}),
    Send("temporal_node", {"observation": obs}),
]
return Command(
    update={"modality_results": {}},
    goto=sends
)
```

- 5 つのモダリティを真の並列実行
- VLM（HTTP I/O: ~20-30s）+ Depth（推定: ~5-10s）+ Audio（キーワード: <0.1s）+ Infrared（VLM: ~15-20s）+ Temporal（VLM: ~15-20s）
- 総処理時間 ≈ max(VLM, Infrared, Temporal) ≈ 20-30s（YOLO 削除による簡潔化）

### fan-in: ラッチ付きバリア機構

```python
def join_modalities(state, runtime):
    expected = set(runtime.context["expected_modalities"])  # enable_* に応じて動的
    received = set(state["received_modalities"])

    if expected.issubset(received):
        # 全てのモダリティが揃った
        if state["barrier_obs_id"] == state["observation"].obs_id:
            # 既に fuse 実行済み → 重複防止
            return Command(goto=END)
        # 初回 → fuse_modalities へ
        return Command(
            update={"barrier_obs_id": obs_id},
            goto="fuse_modalities"
        )
    else:
        # 未揃い → このサブタスクの枝を終了（他のノード完了待ち）
        return Command(goto=END)
```

- **barrier_obs_id**: ラッチ機構で同フレーム内の重複実行を防止
- **expected_modalities**: config に応じて動的に生成
  - 常に: ["vlm"]
  - enable_audio=true → ["vlm", "audio"]
  - enable_depth=true → ["vlm", "depth"]
  - infrared_image_path がある場合 → ["vlm", "infrared"]
  - prev_image_path がある場合 → ["vlm", "temporal"]
  - 例: enable_audio=true, enable_depth=true, prev_image_path がある場合 → ["vlm", "audio", "depth", "temporal"]

## パフォーマンス

### メモリ管理

- **modality_results**: Dict で最新結果のみ保有
- **messages**: スライディングウィンドウ（直近20件）
- **errors**: 最大50件保有

### スレッドセーフティ

VisionAnalyzer, DepthEstimator などで Lock を使用（OpenAI SDK の呼び出しが同期的なため）：

```python
class VisionAnalyzer:
    def __init__(self):
        self._lock = threading.Lock()

    def analyze(self, image_path: str, max_tokens: int = 4096) -> VisionAnalysisResult:
        with self._lock:
            # OpenAI API 呼び出しはスレッドセーフ（SDK が管理）
            # Lock は複数モダリティノードが同時アクセスする場合の保護
            return self._call_llm(...)
```

注: OpenAI SDK 自体はスレッドセーフですが、複数モダリティの並列処理でコンテキスト切り替えによるレート制限対策のため Lock を使用。

## 拡張ポイント

### 新センサー追加（例: LiDAR）

1. `modality_nodes.py` に LidarAnalyzer クラス追加
2. `agent.py` に lidar_node ノード関数追加
3. `ingest_observation` で `Send("lidar_node", ...)` 追加
4. `expected_modalities` に "lidar" を追加（config.get("enable_lidar", False) で動的に）
5. `fuse_modalities` で modality_results["lidar"] を PerceptionIR に統合

詳細は [EXTENDING.md](./EXTENDING.md) 参照。

## テスト

### E2E テスト

```bash
pytest tests/test_e2e.py -v
```

- LLM なしで動作（OpenAI API 不要）
- ~1秒で完了
- 決定的な結果（毎回同じ出力）

### 実行

```bash
# LLM ありで実行
export OPENAI_API_KEY="sk-..."
python src/run.py

# LLM なしで実行（ヒューリスティック）
python src/run.py
```

## ファイル構成

```
src/safety_agent/
├── agent.py          # LangGraph グラフビルダー
├── modality_nodes.py # VisionAnalyzer, AudioAnalyzer, DepthEstimator, InfraredImageAnalyzer, TemporalImageAnalyzer
└── schema.py         # Pydantic スキーマ
```

## 参考資料

- [LangGraph ドキュメント](https://langchain-ai.github.io/langgraph/)
- [QUICK_START.md](./QUICK_START.md) - はじめ方
- [EXTENDING.md](./EXTENDING.md) - 拡張ガイド
