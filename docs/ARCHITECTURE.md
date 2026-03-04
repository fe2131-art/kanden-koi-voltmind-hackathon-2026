# Safety View Agent - システムアーキテクチャ

Safety View Agent は、LangGraph を使用したマルチモーダル安全支援システムです。複雑なロボット環境での安全性を動的に評価し、次に観測すべき視点を提案します。

## 全体構成

**LangGraph の fan-out/fan-in パイプライン** で複数モダリティを並列処理します：

```
Observation Input (image + audio)
         ↓
[ingest_observation]
    ↓ (fan-out: Send API)
    ├─→ [vision_node]  (VLM + YOLO)
    ├─→ [audio_node]   (音声解析)
    └─→ [future nodes] (拡張可能)
    ↓ (fan-in: 結果待機)
[fuse_modalities]
    ↓
[update_world_model]
    ↓
[propose_next_view_llm]
    ↓
[validate_and_guardrails]
    ↓
[select_view]
    ↓
[emit_output] → latest_output に集約
    ↓
[bump_step]
    ↓
(should_continue で分岐)
```

## コアコンポーネント

### 1. LangGraph グラフ（`src/safety_agent/agent.py`）

**fan-out/fan-in パイプライン**：
- **fan-out**: `Command(update={...}, goto=[Send(...), Send(...)])`で複数ノードに並列送信
- **fan-in**: `modality_results` 辞書で全結果を収集後に統合ノードへ
- **emit_output**: 各フレーム処理結果を `latest_output` に格納

### 2. モダリティ処理（`src/safety_agent/modality_nodes.py`）

独立したクラスで各モダリティを処理：

- **VisionAnalyzer**: OpenAI互換VLMで画像分析（httpx使用）
- **YOLODetector**: ultralytics YOLOで物体検出（スレッドセーフ Lock付き）
- **AudioAnalyzer**: 音声テキストからキュー抽出（ヒューリスティック）

### 3. 知覚エンジン（`src/safety_agent/perceiver.py`）

モダリティ結果からハザードと未確認領域を推定：

```python
Perceiver.estimate(obs, objects, audio_cues, vision_description) 
→ PerceptionIR (objects, hazards, unobserved, audio, vision_description)
```

### 4. 世界モデル（`src/safety_agent/schema.py`）

Pydantic スキーマで状態管理：
- `WorldModel`: 全体状態（fused_hazards, outstanding_unobserved）
- `PerceptionIR`: フレーム単位の内部表現
- `ViewCandidate`: 次ビュー候補

## データフロー

### 1フレーム処理

```
Input: Observation
  ↓
1. ingest_observation
   - obs を provider から取得
   - modality_results をリセット
   - vision_node, audio_node に Send
  ↓
2. vision_node & audio_node (並列)
   - VLM 分析（HTTP I/O）
   - YOLO 検出（CPU）
   - 音声解析（ヒューリスティック）
  ↓
3. fuse_modalities (fan-in)
   - Perceiver で統合推定
   - PerceptionIR を生成
  ↓
4. update_world_model
   - 世界状態を更新
  ↓
5. propose_next_view_llm
   - LLM 有→LLM提案、なし→ヒューリスティック
  ↓
6. validate_and_guardrails
   - 提案ビューの安全性チェック
  ↓
7. select_view
   - 最適ビュー選択
  ↓
8. emit_output
   - latest_output に集約
  ↓
9. bump_step → END or 次フレーム
```

## LLM フォールバック

```
LLM = get_llm(config)
  ├─ OPENAI_API_KEY + "openai" → OpenAI API
  ├─ LLM_BASE_URL + "vllm" → ローカルサーバー
  └─ なし → None

propose_next_view_llm:
  ├─ llm is not None → LLM で提案
  └─ llm is None → _heuristic_plan で代替
```

**ヒューリスティック**: 未確認領域をリスク順にソート → ビュー候補生成

## 設定管理

`configs/default.yaml` で外部化：

```yaml
agent:
  max_steps: 1           # フレーム処理数（-1: 全フレーム）

llm:
  provider: "openai"     # or "vllm"

view_planning:
  max_outstanding_regions: 6
  safety_priority_weight: 0.7
  info_gain_weight: 0.3
```

## 並列化の仕組み

### fan-out: 複数ノードへ同時送信

```python
return Command(
    update={"modality_results": {}},
    goto=[
        Send("vision_node", {"observation": obs}),
        Send("audio_node", {"observation": obs}),
    ]
)
```

- HTTP I/O（VLM: ~20-30s）と CPU処理（YOLO: <1s）を並列実行
- 総処理時間 ≈ max(VLM, YOLO) ≈ VLM時間

### fan-in: バリア機構

```python
def fuse_modalities(state):
    results = state["modality_results"]
    # expected_modalities が全て揃うまで待機
    if not all(m in results for m in expected):
        return
    # 揃ったら統合処理
```

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
