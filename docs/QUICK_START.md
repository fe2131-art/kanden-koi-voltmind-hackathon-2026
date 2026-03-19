# Safety View Agent - クイックスタート

5分で Safety View Agent を始められます。

## 前提条件

- Python 3.12+
- `uv` パッケージマネージャー
- 動画ファイル（optional）

## セットアップ（1分）

```bash
# リポジトリをクローン
git clone <repository-url>
cd kanden-koi-voltmind-hackathon-2026

# 依存関係をインストール
uv sync --extra dev

# 仮想環境を有効化
source .venv/bin/activate
```

## データセット配置

### 2 つのデータ入力モード

Safety View Agent は以下の 2 つのモードをサポートしています：

| モード | 入力ソース | 用途 |
|--------|-----------|------|
| **manual** | `data/videos/` | カスタム動画による評価 |
| **inspesafe** | `../InspecSafe-V1/` | InspecSafe-V1 産業用データセット |

### manual モード（デフォルト）

動画ファイルを `data/videos/` に配置して使用：

```bash
# 動画ファイルを配置
cp your_video.mp4 data/videos/

# エージェント実行
python src/run.py
```

### inspesafe モード

InspecSafe-V1 データセットをセッション指定で自動展開・処理。詳細は [INSPESAFE_INTEGRATION.md](INSPESAFE_INTEGRATION.md) を参照：

```yaml
# configs/default.yaml
data:
  mode: "inspesafe"
  inspesafe:
    dataset_path: "../InspecSafe-V1"
    session: "train/Other_modalities/58132919535743_20251118_session_1400_2#bowenguanshang-you"
```

```bash
# エージェント実行（動画・音声は自動展開）
python src/run.py
```

**詳細セットアップ:** [INSPESAFE_INTEGRATION.md](INSPESAFE_INTEGRATION.md)

## 最初の実行（2分）

### オプション A: LLM なし（推奨・テスト用）

```bash
# ヒューリスティックスで実行（LLM API不要）
python src/run.py
```

**出力例**:
```
2026-03-04 22:30:45 - safety_view_agent - INFO - Found 30 frame(s)
2026-03-04 22:30:45 - safety_view_agent - INFO - Running Safety View Agent
2026-03-04 22:30:45 - safety_view_agent - INFO - Selected view: view_blind_left (pan=-30.0°)
2026-03-04 22:30:45 - safety_view_agent - INFO - Results appended to data/perception_results.json
```

✅ **完了！** `data/perception_results.json` に結果が保存されました。

### オプション B: OpenAI API で実行

```bash
# 1. APIキーを設定
export OPENAI_API_KEY="sk-your-api-key-here"

# 2. 実行
python src/run.py
```

### オプション C: ローカル LLM（vLLM）で実行

vLLM は **Structured Outputs** 機能で JSON スキーマに厳密に従う出力を生成するため、JSON パースエラーが発生しません（推奨）。

```bash
# 1. 別ターミナルで vLLM サーバーを起動（JSON スキーマ対応モデルを使用）
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --port 8000 \
  --enable-prefix-caching  # オプション：キャッシング有効化で高速化

# 2. メインターミナルで実行
export LLM_BASE_URL="http://localhost:8000"
export LLM_MODEL="meta-llama/Llama-2-7b-hf"
export LLAMA_LOG_LEVEL="off"  # オプション：vLLM ログを抑制
python src/run.py
```

**vLLM の利点**:
- 📊 **Structured Outputs**: JSON スキーマで出力形式を保証
- ⚡ **高速**: CPU/GPU で高速推論
- 💰 **無料**: APIコストなし
- 🔒 **プライベート**: ローカル実行で機密性を確保

## `.env` ファイルで環境変数を管理（推奨）

毎回 export する代わりに、`.env` ファイルを作成：

```bash
# .env ファイルを作成
cat > .env << EOF
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-5-nano-2025-08-07
EOF

# これで python src/run.py が自動的に .env を読み込みます
python src/run.py
```

## 設定オプション（`configs/default.yaml`）

### エージェント設定

```yaml
agent:
  max_steps: 1              # 処理フレーム数（-1: 全フレーム）
  enable_audio: true        # 音声解析の有効/無効
  enable_depth: true        # 深度推定の有効/無効
  context_history_size: 1   # 前回判断の参照（0=なし, 1=前回のみ、推奨）
```

### context_history_size について

- **0**: フレーム独立判定（メモリ最小、トレンド検出不可）
- **1**: **推奨**。前フレームの判断を参照し、変化を検出（短期トレンド）
- **2+**: 将来拡張（複数フレーム履歴）

#### 効果の違い

```
context_history_size = 0:
  フレーム1: "high risk" → フレーム2: "low risk" → 見た目は改善

context_history_size = 1（推奨）:
  フレーム1: "high risk" → フレーム2: "low risk" →
  LLM: "前回は high だったが、今は low。リスク低下したか？"
  → より正確な判定（文脈認識）
```

## テスト実行（1分）

```bash
# E2E テスト実行（LLM 不要）
pytest tests/test_e2e.py -v
```

**期待される結果**:
```
tests/test_e2e.py::test_e2e_agent_no_llm PASSED [100%]
```

## 出力ファイル

実行後、以下のファイルが生成されます：

```
data/
├── frames/
│   ├── frame_0.000s.jpg
│   ├── frame_1.005s.jpg
│   └── ...
├── audio.wav
├── perception_results.json          ← 分析結果（JSON、frames 配列形式）
└── flow.md                          ← グラフ図（Mermaid）
```

### perception_results.json の例

```json
{
  "frames": [
    {
      "frame_id": "img_0",
      "timestamp": "2026-03-19T10:30:45.123456",
      "video_timestamp": 0.0,
      "vision_analysis": {
        "scene_description": "田園風景。道路沿いに建造物...",
        "critical_points": [
          {
            "region_id": "foreground_center",
            "description": "道路上の障害物",
            "severity": "high"
          }
        ],
        "blind_spots": [
          {
            "region_id": "blind_left",
            "description": "左側視界外",
            "position": "left",
            "severity": "medium"
          }
        ],
        "overall_risk": "medium",
        "confidence_score": 0.85
      },
      "depth_analysis": {
        "depth_layers": [
          {
            "zone": "near",
            "description": "前方5m以内に障害物"
          }
        ],
        "overall_risk": "medium",
        "confidence_score": 0.78
      },
      "infrared_analysis": null,
      "temporal_analysis": null,
      "audio": [
        {
          "cue": "vehicle_approaching",
          "severity": "high",
          "evidence": "エンジン音が聞こえる"
        }
      ],
      "belief_state": {
        "hazard_tracks": [
          {
            "hazard_id": "hazard_1",
            "hazard_type": "visible_hazard",
            "region_id": "foreground_center",
            "status": "new",
            "severity": "high",
            "confidence_score": 0.88,
            "supporting_modalities": ["vision", "depth"]
          }
        ],
        "overall_risk": "high",
        "recommended_focus_regions": ["foreground_center"]
      },
      "assessment": {
        "risk_level": "high",
        "safety_status": "危険な物体を検出",
        "detected_hazards": ["obstacle_in_path"],
        "action_type": "inspect_region",
        "target_region": "foreground_center",
        "reason": "視野内に検出された物体の詳細確認が必要",
        "priority": 0.9,
        "temporal_status": "new",
        "confidence_score": 0.88
      },
      "errors": []
    }
  ]
}
```

**フォーマットの特徴**:
- `frames` 配列で統一（フラット構造）
- 各フレームは独立した判断結果を含む
- `vision_analysis`, `depth_analysis`, `infrared_analysis`, `temporal_analysis` は各モダリティの結果
- `belief_state` は複数フレームにわたる危険状態の管理
- `assessment` は LLM による総合安全判断
- `errors` でモダリティ処理のエラーを記録

## 設定のカスタマイズ

`configs/default.yaml` で動作をカスタマイズ：

```yaml
# フレーム処理数
agent:
  max_steps: 1          # 1フレーム、-1で全フレーム

# LLM プロバイダー
llm:
  provider: "openai"    # or "vllm"

# ビュー選択戦略
view_planning:
  safety_priority_weight: 0.7  # 安全性の重み（0.7推奨）
  info_gain_weight: 0.3        # 情報利得の重み
```

## 複数フレーム処理

```bash
# 設定を編集
sed -i 's/max_steps: 1/max_steps: 3/' configs/default.yaml

# 実行
python src/run.py

# 結果確認
python -c "import json; d=json.load(open('data/perception_results.json')); print(f'Frames: {len(d[\"frames\"])}')"
```

**出力**: `Frames: 3`

**注**: `agent_execution` は廃止されました。すべての結果は `frames` 配列に統合されています。

## トラブルシューティング

### エラー: `OPENAI_API_KEY not set`

```bash
# APIキーを設定して再実行
export OPENAI_API_KEY="sk-..."
python src/run.py

# または .env ファイルを作成
echo "OPENAI_API_KEY=sk-..." > .env
```

### エラー: `No frames found in data/frames`

```bash
# 動画ファイルを data/videos/ に配置
cp your_video.mp4 data/videos/

# 実行（フレームが自動抽出されます）
python src/run.py
```

### テスト失敗

```bash
# 環境を再構築
uv sync --force
pytest tests/test_e2e.py -v
```

## 次のステップ

- 📖 [システムアーキテクチャ](./ARCHITECTURE.md) - 詳細な設計を学ぶ
- 🔧 [拡張ガイド](./EXTENDING.md) - 新しいセンサーやロジックを追加する
- 💡 [CLAUDE.md](../CLAUDE.md) - プロジェクト全体の仕様書

## よくある使用例

### 例 1: 単一画像の安全性分析

```bash
# data/frames/ に画像を配置
cp image.jpg data/frames/frame_0s.jpg

# 実行
python src/run.py
```

### 例 2: 動画から30フレームを分析

```bash
# 設定変更
echo "
agent:
  max_steps: 30
" > configs/default.yaml

# 動画を data/videos/ に配置
cp video.mp4 data/videos/

# 実行
python src/run.py

# 結果確認
python -c "import json; d=json.load(open('data/perception_results.json')); print(len(d['perception_results']))"
```

### 例 3: OpenAI API で高品質な判断を得る

```bash
# APIキー設定
export OPENAI_API_KEY="sk-..."

# 実行（LLM による安全判断を得られます）
python src/run.py

# JSON で判断内容を確認
python -c "import json; d=json.load(open('data/perception_results.json')); print(json.dumps(d['frames'][0]['assessment'], indent=2))"
```

**出力例**:
```json
{
  "risk_level": "high",
  "safety_status": "危険な物体を検出",
  "detected_hazards": ["obstacle_in_path"],
  "action_type": "inspect_region",
  "target_region": "foreground_center",
  "reason": "視野内に検出された物体の詳細確認が必要",
  "priority": 0.9,
  "temporal_status": "new"
}
```

## 支援が必要な場合

- 📚 [アーキテクチャドキュメント](./ARCHITECTURE.md)
- 🛠️ [拡張ガイド](./EXTENDING.md)
- 🐛 バグ報告: GitHub Issues
