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

```bash
# 1. 別ターミナルで vLLM サーバーを起動
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --port 8000

# 2. メインターミナルで実行
export LLM_BASE_URL="http://localhost:8000"
export LLM_MODEL="meta-llama/Llama-2-7b-hf"
python src/run.py
```

## `.env` ファイルで環境変数を管理（推奨）

毎回 export する代わりに、`.env` ファイルを作成：

```bash
# .env ファイルを作成
cat > .env << EOF
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o
EOF

# これで python src/run.py が自動的に .env を読み込みます
python src/run.py
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
│   ├── frame_0s.jpg
│   ├── frame_1s.jpg
│   └── ...
├── audio.wav
├── perception_results.json          ← 分析結果（JSON）
├── agent_execution_summary.txt      ← ログ
└── flow.md                          ← グラフ図（Mermaid）
```

### perception_results.json の例

```json
{
  "perception_results": [
    {
      "obs_id": "img_0",
      "video_timestamp": 0.0,
      "ir": {
        "objects": [],
        "hazards": [],
        "unobserved": [
          {
            "region_id": "blind_left",
            "risk": 0.4,
            "suggested_pan_deg": -30.0
          }
        ],
        "vision_description": "田園風景。道路沿いに..."
      },
      "selected_view": {
        "view_id": "view_blind_left",
        "pan_deg": -30.0,
        "tilt_deg": 0.0
      }
    }
  ],
  "agent_execution": [...]
}
```

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
python -c "import json; d=json.load(open('data/perception_results.json')); print(f'Frames: {len(d[\"perception_results\"])}')"
```

**出力**: `Frames: 3`

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

### 例 3: OpenAI API で高品質な提案を得る

```bash
# APIキー設定
export OPENAI_API_KEY="sk-..."

# 実行（LLM による次ビュー提案を得られます）
python src/run.py

# JSON で提案内容を確認
python -c "import json; d=json.load(open('data/perception_results.json')); print(d['agent_execution'][0]['selected_view'])"
```

## 支援が必要な場合

- 📚 [アーキテクチャドキュメント](./ARCHITECTURE.md)
- 🛠️ [拡張ガイド](./EXTENDING.md)
- 🐛 バグ報告: GitHub Issues
