# アーキテクチャドキュメント

Safety View Agent のシステム設計と実装パターン

## システム概要

```
┌─────────────┐
│   入力画像   │
│ (input/)   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  Vision Analysis                 │
│  (src/safety_agent/perceiver.py)│
│  - YOLO検出（オプション）        │
│  - Vision API分析               │
│  - 未確認領域推定               │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  エージェントループ              │
│  (src/safety_agent/agent.py)    │
│  - LangGraph 状態機械           │
│  - 7段階の処理ノード            │
│  - LLM計画 + ヒューリスティック  │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────┐
│  出力ファイル    │
│ (output/)      │
│ - JSON         │
│ - テキスト     │
│ - 図表         │
└─────────────────┘
```

## ファイル構成

```
src/
├── run.py                      # エントリーポイント
│                                # - 設定読み込み
│                                # - 画像処理
│                                # - エージェント実行
│                                # - 結果出力
│
└── safety_agent/
    ├── __init__.py            # パッケージ初期化
    │
    ├── schema.py              # Pydantic モデル定義
    │   ├── BoundingBox        # 物体検出の矩形
    │   ├── DetectedObject     # 検出物体
    │   ├── Hazard             # ハザード情報
    │   ├── Observation        # 観測データ
    │   ├── PerceptionIR       # 知覚結果
    │   ├── WorldModel         # 世界モデル
    │   ├── ViewCandidate      # 次ビュー候補
    │   └── ViewCommand        # 実行ビュー
    │
    ├── perceiver.py           # 知覚処理
    │   ├── Perceiver          # メインクラス
    │   │   ├── run()          # 観測→知覚結果
    │   │   ├── _yolo_detect() # YOLO物体検出
    │   │   ├── _infer_hazards()      # ハザード推定
    │   │   ├── _infer_unobserved()   # 未確認領域推定
    │   │   └── _audio_to_cues()      # 音声処理
    │   │
    │   └── ObservationProvider # 観測ソース管理
    │       ├── next()         # 次の観測を取得
    │       └── reset()        # リセット
    │
    └── agent.py               # エージェント実装
        ├── OpenAICompatLLM    # LLM クライアント
        │   ├── chat_json()    # JSON生成
        │   └── chat_text()    # テキスト生成
        │
        ├── LangGraph ノード関数（7つ）
        │   ├── ingest_observation()
        │   ├── perceive_and_extract_ir()
        │   ├── update_world_model()
        │   ├── propose_next_view_llm()
        │   ├── validate_and_guardrails()
        │   ├── select_view()
        │   └── bump_step()
        │
        ├── _heuristic_plan()  # フォールバック計画
        ├── _robust_json_loads()       # JSON解析
        └── build_agent()      # グラフ構築
```

## 実行フロー

### 1. セットアップフェーズ（run.py）

```
┌─ 設定読み込み (load_config)
│  └─ configs/default.yaml を解析
│     ├─ agent.max_steps = 3
│     ├─ llm.provider = "openai"
│     └─ llm.openai.model = "gpt-5-nano-2025-08-07"
│
├─ 画像読み込み (load_images_from_input)
│  └─ input/ フォルダをスキャン
│
├─ LLM 初期化 (get_llm)
│  └─ OpenAI API キーを設定
│
└─ エージェント構築 (build_agent)
   └─ LangGraph ステートマシン
```

### 2. 知覚フェーズ（perceiver.py）

```
観測 (Observation)
 │
 ├─ YOLO 検出（オプション）
 │  └─ 物体検出
 │
 ├─ Vision API 分析
 │  └─ テキスト結果
 │
 ├─ ハザード推定
 │  ├─ 危険物体を特定
 │  └─ 信度スコア計算
 │
 ├─ 未確認領域推定
 │  ├─ blind_left (40% risk)
 │  ├─ blind_right (40% risk)
 │  └─ blind_back (30% risk)
 │
 └─ PerceptionIR （知覚結果）
    ├─ objects: [...]
    ├─ hazards: [...]
    ├─ unobserved: [...]
    └─ audio: [...]
```

### 3. エージェントループ（agent.py）

```
ステップ 1: ingest_observation
 │ 次の観測を取得
 │
 ├─ 観測ある？ YES → ステップ 2 へ
 │                 NO → [ingest] done
 │
 ▼
ステップ 2: perceive_and_extract_ir
 │ Perceiver で知覚処理
 │
 ▼
ステップ 3: update_world_model
 │ 検出結果を世界モデルに統合
 │
 ▼
ステップ 4: propose_next_view_llm ⭐ 重要
 │ 次のビュー（視点）を決定
 │
 ├─ LLM で計画生成？
 │  ├─ JSON 正常 → NextViewPlan 生成
 │  └─ JSON 失敗 → heuristic フォールバック
 │
 ▼
ステップ 5: validate_and_guardrails
 │ リスク閾値チェック
 │
 ├─ max_unobserved_risk < threshold？
 │  ├─ YES → done=True で終了
 │  └─ NO  → 次ステップ へ
 │
 ▼
ステップ 6: select_view
 │ 候補から最適ビューを選択
 │
 ▼
ステップ 7: bump_step
 │ ステップカウント++
 │
 ├─ step < max_steps？
 │  ├─ YES → ステップ 1 へ（ループ）
 │  └─ NO  → 終了
```

## LLM 互換性設計

### パラメータ自動判別

```python
# gpt-5-nano-2025-08-07 の場合
if "gpt-5" in model.lower():
    use_max_completion_tokens = True
    skip_temperature = True
else:
    use_max_completion_tokens = False
    skip_temperature = False
```

### Vision API 対応

```python
payload = {
    "model": "gpt-5-nano-2025-08-07",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
                },
                {
                    "type": "text",
                    "text": "Please analyze this image..."
                }
            ]
        }
    ],
    # gpt-5-nano の場合のみ：
    "max_completion_tokens": 500
}
```

## 状態管理（AgentState）

```python
@dataclass
class AgentState:
    messages: List[BaseMessage]      # メッセージログ
    step: int                         # ステップカウンタ
    max_steps: int                    # 最大ステップ数
    observation: Optional[Observation]# 現在の観測
    ir: Optional[PerceptionIR]       # 知覚結果
    world: WorldModel                # 世界モデル
    plan: Optional[NextViewPlan]     # 計画
    selected: Optional[ViewCommand]  # 選択ビュー
    done: bool                        # 終了フラグ
    errors: List[str]                # エラーログ
```

## グレースフルデグラデーション

### LLM が失敗した場合

```
LLM.chat_json() → Error (400, JSON parse fail, etc)
    │
    ├─ Exception catch
    │  └─ errors.append() でエラー記録
    │
    └─ _heuristic_plan() へフォールバック
       │
       └─ 未確認領域のリスク順で次ビュー候補を生成
          ├─ blind_left: 40% → pan=-30°
          ├─ blind_right: 40° → pan=30°
          └─ blind_back: 30% → pan=180°
```

### ヒューリスティック計画ロジック

```python
def _heuristic_plan(state: AgentState) -> NextViewPlan:
    # 未確認領域をリスク順にソート
    sorted_unobserved = sorted(
        state.world.outstanding_unobserved,
        key=lambda x: x.risk,
        reverse=True
    )

    # 最高リスク領域を次ビュー候補に
    return NextViewPlan(
        candidates=[
            ViewCandidate(
                view_id=f"heur_{u.region_id}",
                pan_deg=u.suggested_pan_deg,
                tilt_deg=u.suggested_tilt_deg or 0,
                zoom=1.0,
                why=f"未確認領域({u.region_id})のリスク({u.risk:.1%})が高い"
            )
            for u in sorted_unobserved
        ]
    )
```

## 出力形式

### 1. perception_results.json

```json
{
  "input_images": ["input/image.jpg"],
  "perception_results": [{
    "obs_id": "img_0",
    "objects": [{...}],
    "hazards": [{...}],
    "unobserved": [{...}],
    "audio": [],
    "vision_analysis": "Vision APIの応答"
  }]
}
```

### 2. agent_execution_summary.txt

人間向け形式：
```
SELECTED VIEW COMMAND:
  View ID: heur_blind_left
  Pan: -30.0°, Tilt: 0.0°, Zoom: 1.0x

DETECTED HAZARDS:
  • unidentified_foreground_object (50.00%)

OUTSTANDING UNOBSERVED REGIONS:
  • blind_left: 左側の死角（40.0%）
```

### 3. flow.md

LangGraph 実行フロー（Mermaid 図）

## パフォーマンス

| 処理段階 | 時間 | 説明 |
|---------|------|------|
| **Vision API 呼び出し** | 10-20秒 | 画像エンコード + API リクエスト |
| **LLM テキスト生成** | 3-10秒 | JSON パース失敗時は短い |
| **エージェントループ** | 5-20秒 | max_steps=3 の場合 |
| **出力生成** | 1-2秒 | ファイル書き込み |
| **合計** | 30-40秒 | |

## 拡張可能性

### 新しいハザード検出ロジックを追加

`perceiver.py` の `_infer_hazards()` を拡張：

```python
def _infer_hazards(self, objects: List[DetectedObject]) -> List[Hazard]:
    hazards = []

    # 既存: unidentified_foreground_object
    for obj in objects:
        if obj.label == "foreground_object":
            hazards.append(Hazard(...))

    # 新規: カスタムロジックを追加
    for obj in objects:
        if obj.label == "person" and obj.confidence > 0.8:
            hazards.append(Hazard(...))  # 人物検出

    return hazards
```

### 新しいビュー選択戦略

`agent.py` の `select_view()` をカスタマイズ：

```python
def select_view(state):
    if not state.plan or not state.plan.candidates:
        return {...}

    # デフォルト: 最初の候補を選択
    selected = state.plan.candidates[0]

    # カスタム: 最適化アルゴリズムを適用
    # selected = optimize_view_selection(state.plan.candidates)

    return {...}
```

## 依存関係

- **LangGraph**: 状態機械フレームワーク
- **Pydantic**: データモデル定義
- **OpenAI SDK**: Vision API 呼び出し
- **httpx**: HTTP クライアント
- **PyYAML**: 設定ファイル解析
- **Pillow**: 画像処理（フォールバック用）

