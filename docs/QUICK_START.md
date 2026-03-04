# クイックスタートガイド

**5分でプロジェクトを動かすガイド（LangGraph fan-out 並列マルチモーダル対応版）**

## TL;DR - コマンドのみ

```bash
# 1. セットアップ（初回のみ）
uv sync --extra dev
cp .env.example .env
# .env を編集して OPENAI_API_KEY を設定

# 2. テスト（LLM不要、1-2分）
pytest tests/ -v

# 3. 実行（Vision API 使用、30-40秒）
python src/run.py

# 4. 結果確認
cat data/perception_results.json
cat data/agent_execution_summary.txt
```

## 実行ログサンプル

fan-out 並列化により、実行ログには新しいノードタグが含まれます。

### LLM なし（ヒューリスティックフォールバック）

```
⚠️  OPENAI_API_KEY not set, using heuristic fallback
⚠️  Using example observations (no images in data/images/)

=== Running Safety View Agent ===

  [ingest] fan-out -> t0
  [vision] objects=0 vlm=none
  [audio] cues=1
  [fuse] hazards=1 unobserved=3 errors=0
  [world] fused_hazards=1 outstanding_unobserved=3
  [plan] LLM not configured -> heuristic fallback
  [validate] guardrails applied
  [select] heur_blind_right pan=30.0 tilt=0.0 done=False

=== Selected view command ===
view_id='heur_blind_right' pan_deg=30.0 tilt_deg=0.0 zoom=1.0 why='...'
```

### LLM あり + 画像入力

```
✅ Using OpenAI API (model=gpt-5-nano-2025-08-07)
✅ Using VisionAnalyzer (model=gpt-5-nano-2025-08-07)
✅ Found 2 image(s) in data/images/

=== Processing Input Images ===

🔍 Processing: scene_001.jpg
   - Objects detected: 3
     • person (85.20%)
     • forklift (72.50%)
     • foreground_object (50.00%)
   - Hazards identified: 2
     • human_present (60.00%)
     • unidentified_foreground_object (50.00%)
   - Unobserved regions: 3
   ✅ Vision Analysis Complete (456 chars)

=== Running Safety View Agent ===

  [ingest] fan-out -> img_0
  [vision] objects=3 vlm=ok
  [audio] cues=0
  [fuse] hazards=2 unobserved=3 errors=0
  [world] fused_hazards=2 outstanding_unobserved=3
  [plan] candidates=3 stop=False
  [validate] guardrails applied
  [select] view_left_scan pan=-30.0 tilt=0.0 done=False
```

### ログメッセージの読み方

| タグ | ノード | 意味 |
|------|--------|------|
| `[ingest]` | `ingest_observation` | 観測取得 + fan-out 開始 |
| `[vision]` | `vision_node` | VLM + YOLO の結果（objects 数、VLM 成否） |
| `[audio]` | `audio_node` | 音声キュー数 |
| `[fuse]` | `fuse_modalities` | 統合後のハザード・未確認領域数、モダリティエラー数 |
| `[world]` | `update_world_model` | 世界モデル更新後の状態 |
| `[plan]` | `propose_next_view_llm` | LLM 計画 or ヒューリスティック |
| `[validate]` | `validate_and_guardrails` | ガードレール適用 |
| `[select]` | `select_view` | 選択されたビューと終了判定 |

## モダリティの有効化

`configs/default.yaml` の `modalities` セクションで各モダリティの有効/無効を制御します。

```yaml
# Modalities configuration (LangGraph fan-out)
modalities:
  vision:
    enabled: true          # VLM による画像分析
    yolo_enabled: false    # YOLO 物体検出（ultralytics が必要）
  audio:
    enabled: true          # 音声テキスト解析
  # lidar:                 # 将来の拡張例
  #   enabled: false
```

### 設定の優先順位

| 設定項目 | YAML キー | 環境変数 | デフォルト |
|---------|----------|---------|----------|
| VLM プロバイダ | `vlm.provider` | - | `"openai"` |
| VLM モデル | `vlm.openai.model` | `VLM_MODEL` | `llm.openai.model` を流用 |
| YOLO 有効化 | `agent.enable_yolo` | - | `false` |
| LLM プロバイダ | `llm.provider` | - | `"openai"` |
| LLM モデル | `llm.openai.model` | `OPENAI_MODEL` | `gpt-5-nano-2025-08-07` |

## YOLO 有効化手順

YOLO を有効化すると、`vision_node` 内で VLM と同時に物体検出が実行されます。

### 1. ultralytics をインストール

```bash
uv pip install ultralytics
```

### 2. configs/default.yaml を編集

```yaml
agent:
  enable_yolo: true    # false → true に変更

modalities:
  vision:
    yolo_enabled: true  # false → true に変更
```

### 3. 実行

```bash
python src/run.py
```

### YOLO 有効化時の動作

```
vision_node の処理:
  1. YOLODetector.detect(image_path)  ← Lock 取得して推論
     → DetectedObject リスト（label, confidence, bbox）
  2. VisionAnalyzer.analyze(image_path)  ← HTTP API 呼び出し
     → テキスト分析結果

  両方の結果が ModalityResult に格納される
```

YOLO がインストールされていない場合、`_simple_image_analysis()` にフォールバックします（Pillow で簡易分析）。

## エラーメッセージ解釈

モダリティ処理中のエラーは `PerceptionIR.modality_errors` に記録され、ログの `[fuse]` タグで確認できます。

### 例 1: VLM API エラー

```
[vision] objects=0 vlm=none
[fuse] hazards=0 unobserved=3 errors=1
```

`errors=1` は VLM API の呼び出しに失敗したことを示します。エージェントは VLM なしで続行します（検出結果のみでハザード推定）。

### 例 2: 観測データなし

```
[vision] error: No observation provided
[audio] error: No observation provided
[fuse] hazards=0 unobserved=0 errors=2
```

`ingest_observation` から観測が渡されなかった場合のエラーです。

### エラー確認方法

実行後の出力で `errors` を確認できます。

```bash
# Python で確認
python -c "
import json
with open('data/perception_results.json') as f:
    data = json.load(f)
for r in data['perception_results']:
    if r.get('modality_errors'):
        print(f'{r[\"obs_id\"]}: {r[\"modality_errors\"]}')
"
```

## デバッグ用ログ

### ノード単位でのメッセージフィルタリング

実行結果のメッセージログから特定ノードの出力だけを抽出できます。

```bash
# 実行後、メッセージをファイルに出力
python src/run.py 2>&1 | tee run_log.txt

# 特定ノードのログだけ抽出
grep '\[vision\]' run_log.txt    # vision_node の結果
grep '\[audio\]' run_log.txt     # audio_node の結果
grep '\[fuse\]' run_log.txt      # fuse_modalities の結果
grep '\[plan\]' run_log.txt      # 計画ノードの結果
grep '\[select\]' run_log.txt    # ビュー選択結果
```

### Python コードでメッセージをフィルタリング

```python
from safety_agent.agent import build_agent

agent = build_agent()
out = agent.invoke(initial_state, context=context)

# ノード単位でフィルタリング
for msg in out["messages"]:
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
    if "[vision]" in content:
        print(f"Vision: {content}")
    elif "[fuse]" in content:
        print(f"Fuse: {content}")
```

### 詳細デバッグ（pytest）

```bash
# E2E テストを詳細出力で実行
pytest tests/test_e2e.py -vv -s

# 出力例:
# [ingest] fan-out -> t0
# [vision] objects=0 vlm=none
# [audio] cues=1
# [fuse] hazards=1 unobserved=3 errors=0
# ...
# Selected view: heur_blind_right
# Messages: 16
# Errors: []
```

## 主要コマンド一覧

| コマンド | 説明 |
|---------|------|
| `uv sync --extra dev` | 仮想環境をセットアップ |
| `pytest tests/ -v` | テストを実行（LLM 不要） |
| `python src/run.py` | エージェントを実行 |
| `python finetuning/train_dummy.py` | ダミー学習を実行 |

## 次のステップ

- [アーキテクチャ](ARCHITECTURE.md) - fan-out/fan-in の詳細設計
- [拡張ガイド](EXTENDING.md) - 新センサー追加手順
- [トラブルシューティング](TROUBLESHOOTING.md) - よくある問題
- [CLAUDE.md](../CLAUDE.md) - Claude Code 向け情報

## よくある質問（FAQ）

**Q: fan-out とは何ですか？**
A: `ingest_observation` ノードから `vision_node` と `audio_node` へ同時にデータを送信し、並列処理する LangGraph のパターンです。`Command` + `Send` API で実現しています。

**Q: Vision API は何をしているの？**
A: 入力画像を分析して、安全上の危険物や注意が必要な領域を自動検出しています。

**Q: なぜ `gpt-5-nano-2025-08-07` なの？**
A: Vision 対応モデルの中でコスト効率が最適だからです。変更しないでください。

**Q: オフラインで実行できる？**
A: はい、LLM を使わずヒューリスティックフォールバックで動作可能です。

**Q: modality_errors にエラーが記録されていても大丈夫？**
A: はい、モダリティエラーは記録されますが、エージェントは利用可能な結果のみで処理を続行します。VLM が失敗しても音声解析の結果は使われます。
