# Safety View Agent - システムアーキテクチャ

Safety View Agent は、LangGraph を使用したマルチモーダル安全支援システムです。複雑なロボット環境での安全性を動的に評価し、次に観測すべき視点を提案します。

## 全体構成

**LangGraph の fan-out/fan-in パイプライン** で複数モダリティを並列処理します：

```
Observation Input (image + audio)
         ↓
[ingest_observation] (観測データ取得・リセット)
    ↓ (fan-out: Command + Send API)
    ├─→ [yolo_node]         (物体検出: ultralytics)
    ├─→ [vlm_node]          (画像分析: OpenAI Vision API)
    └─→ [audio_node]        (音声解析: キーワード抽出)
    ↓ (各ノードが join_modalities へ)
[join_modalities] (fan-in バリア + ラッチ)
    ├─ 全モダリティ揃うまで待機
    └─ barrier_obs_id で重複実行を防止
    ↓
[fuse_modalities] (モダリティ統合)
    ├─ PerceptionIR 生成（objects, audio, vision_description）
    └─ （Perceiver は LLM へ統合）
    ↓
[update_world_model] (世界状態更新)
    └─ ハザード融合 + 未確認領域リスク順
    ↓
[determine_next_action_llm] (知覚推論 + 総合安全判断【統合】)
    ├─ LLM あり:
    │  ├─ ステップ1: 知覚推論（YOLO/VLM/音声からハザード推定）
    │  ├─ ステップ2: 安全判断（SafetyAssessment 生成）
    │  └─ ir.hazards, ir.unobserved を LLM 出力で更新
    └─ LLM なし: ヒューリスティック判断
    ↓
[emit_output] (フレーム出力)
    └─ latest_output に集約 → streaming 出力
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

3 つのモダリティを独立したクラスで処理（各 node として LangGraph に登録）：

- **YOLODetector**: ultralytics YOLO で物体検出
  - `enable_yolo: true` で初期化（デフォルト: false）
  - スレッドセーフ Lock 付き（ultralytics が非スレッドセーフのため）
  - ノード: `yolo_node()` → ModalityResult 返却

- **VisionAnalyzer**: OpenAI互換 VLM で画像分析
  - `default_prompt` と `max_tokens` をコンストラクタ注入
  - httpx で HTTP I/O（~20-30s）
  - ノード: `vlm_node()` → YOLO 結果と VLM テキストを統合

- **AudioAnalyzer**: 音声テキストからキュー抽出
  - 実装: キーワードマッチング（Speech-to-Text 未実装）
  - `enable_audio: false` で無効化（デフォルト）
  - ノード: `audio_node()` → AudioCue リスト返却

### 3. 知覚エンジン（`src/safety_agent/perceiver.py`）

モダリティ結果からハザードと未確認領域を推定：

```python
Perceiver.estimate(obs, objects, audio_cues, vision_description) 
→ PerceptionIR (objects, hazards, unobserved, audio, vision_description)
```

### 4. 世界モデル（`src/safety_agent/schema.py`）

Pydantic スキーマで状態管理：
- `WorldModel`: 全体状態（fused_hazards, outstanding_unobserved, last_assessment）
- `PerceptionIR`: フレーム単位の内部表現
- `SafetyAssessment`: LLM による総合安全判断（risk_level, safety_status, detected_hazards, action_type, target_region, reason, priority）

## データフロー

### 1フレーム処理

```
Input: Observation (image_path + audio_text)
  ↓
1. ingest_observation
   - provider から obs を取得
   - modality_results, received_modalities をリセット
   - barrier_obs_id, latest_output をリセット
   - yolo_node, vlm_node, audio_node に Command + Send で並列送信
  ↓
2. yolo_node, vlm_node, audio_node (真の並列実行)
   a) yolo_node (enable_yolo=true の場合)
      - YOLO 物体検出（CPU: ~1-2s）
      - DetectedObject リスト返却
   b) vlm_node
      - VLM 画像分析（HTTP I/O: ~20-30s）
      - 検出結果と VLM テキストを統合
   c) audio_node (enable_audio=true の場合)
      - audio_text からキーワード抽出
      - AudioCue リスト返却
   ↓ （各ノードが join_modalities へ）
3. join_modalities (fan-in バリア)
   - expected_modalities がすべて received_modalities に揃うまで待機
   - barrier_obs_id で同フレーム内の重複実行を防止
   - 全て揃ったら fuse_modalities へ遷移
  ↓
4. fuse_modalities
   - modality_results から yolo, vlm, audio を取得
   - PerceptionIR（objects, audio, vision_description）生成
   - （Perceiver の推論は LLM に統合されたため、基本的な PerceptionIR のみ）
  ↓
5. update_world_model
   - fused_hazards を信度で統合
   - outstanding_unobserved をリスク順にソート
   - 前回の SafetyAssessment を保持
  ↓
6. determine_next_action_llm 【知覚推論 + 安全判断を統合実行】
   a) LLM あり: 2ステップで実行
      ステップ1: 知覚推論（Perceiver 機能を統合）
        - YOLO検出 + VLM分析 + 音声キューから直接ハザード推定
        - 音声 direction から未確認領域リスク評価
        - 複合条件判定（「人物 AND 車両接近」など）
        → ir.hazards, ir.unobserved を LLM 出力で更新

      ステップ2: 安全判断
        - 推定ハザード + 世界モデル + 前回判断から SafetyAssessment 生成
        * risk_level: high/medium/low
        * safety_status: 現在の状態説明
        * detected_hazards: 検出危険リスト
        * action_type: focus_region/increase_safety/continue_observation
        * priority: 0.0-1.0

   b) LLM なし: _heuristic_assessment() でフォールバック
      - Perceiver のヒューリスティック推論のみ実行
  ↓
7. emit_output
   - obs_id, ir, world, assessment, step, errors を latest_output に集約
   - streaming で即座に出力可能
  ↓
8. bump_step
   - step += 1
   - should_continue で分岐判定
   ├─ step < max_steps → ingest_observation へループ
   └─ END
```

## LLM フォールバック

```
LLM = get_llm(config)
  ├─ OPENAI_API_KEY + "openai" → OpenAI API
  ├─ LLM_BASE_URL + "vllm" → ローカルサーバー
  └─ なし → None

determine_next_action_llm:
  ├─ llm is not None → LLM で SafetyAssessment 生成
  └─ llm is None → _heuristic_assessment で代替
```

**ヒューリスティック**:
- 高リスク未確認領域あり → focus_region（観測指示）
- 低信度ハザード存在 → increase_safety（安全強化指示）
- その他 → continue_observation（継続観測指示）

## 設定管理

### `configs/default.yaml`

```yaml
agent:
  max_steps: 1                    # フレーム処理数（-1: 全フレーム）
  enable_yolo: false              # YOLO 物体検出の有効/無効
  enable_audio: false             # 音声解析の有効/無効（Speech-to-Text 未実装）
  max_outstanding_regions: 6      # LLM が検討する未確認領域の上限数
  context_history_size: 1         # LLM に渡す前回判断の数（0=なし, 1=前回のみ）

llm:
  provider: "openai"              # "openai" or "vllm"

vlm:
  provider: "openai"              # VLM プロバイダー

tokens:
  vision_max_completion_tokens: 5000   # VLM の最大トークン
  chat_max_tokens: 2000                # テキスト LLM の最大トークン
```

### `configs/prompt.yaml` (外部化)

```yaml
vision_analysis:
  default_prompt: |
    この画像の安全性を分析してください...

next_view_proposal:
  system: |
    あなたは安全支援エージェントです...
    （知覚推論 + 安全判断を統合的に実行）
```

### エラーハンドリング

- **ファイル未検出**: 設定ファイル/プロンプトファイル欠落 → `FileNotFoundError` を raise
- **フレーム未検出**: manual モード + data/frames 空 → `FileNotFoundError` を raise（ダミーデータフォールバック廃止）
- **LLM 未設定**: `llm=None` → `_heuristic_assessment()` で自動フォールバック
- **LLM 実行エラー**: JSON パース失敗など → ヒューリスティック判断に降格

## 並列化の仕組み

### fan-out: 複数ノードへ同時送信

```python
# ingest_observation() から
sends = [
    Send("yolo_node", {"observation": obs}),
    Send("vlm_node", {"observation": obs}),
    Send("audio_node", {"observation": obs}),
]
return Command(
    update={"modality_results": {}},
    goto=sends
)
```

- 3 つのモダリティを真の並列実行
- VLM（HTTP I/O: ~20-30s）+ YOLO（CPU: ~1-2s）+ Audio（キーワード: <0.1s）
- 総処理時間 ≈ max(VLM, YOLO) ≈ 20-30s（直列比 >20倍の並列化）

### fan-in: ラッチ付きバリア機構

```python
def join_modalities(state, runtime):
    expected = set(runtime.context["expected_modalities"])  # enable_audio に応じて動的
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
- **expected_modalities**: enable_audio に応じて動的に生成
  - enable_audio=false → ["yolo", "vlm"]
  - enable_audio=true → ["yolo", "vlm", "audio"]

## パフォーマンス

### メモリ管理

- **modality_results**: Dict で最新結果のみ保有
- **messages**: スライディングウィンドウ（直近20件）
- **errors**: 最大50件保有

### スレッドセーフティ

YOLODetector で Lock を使用：

```python
class YOLODetector:
    def __init__(self):
        self._lock = threading.Lock()
    
    def detect(self, image_path):
        with self._lock:
            return self.model.predict(...)
```

## 拡張ポイント

### 新センサー追加（例: LiDAR）

1. `modality_nodes.py` に LidarAnalyzer クラス追加
2. `agent.py` に lidar_node 追加
3. `ingest_observation` で `Send("lidar_node", ...)` 追加
4. `build_agent()` で `add_edge("lidar_node", "fuse_modalities")` 追加

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
├── modality_nodes.py # VisionAnalyzer, YOLODetector, AudioAnalyzer
├── perceiver.py      # ハザード推定エンジン
└── schema.py         # Pydantic スキーマ
```

## 参考資料

- [LangGraph ドキュメント](https://langchain-ai.github.io/langgraph/)
- [QUICK_START.md](./QUICK_START.md) - はじめ方
- [EXTENDING.md](./EXTENDING.md) - 拡張ガイド
