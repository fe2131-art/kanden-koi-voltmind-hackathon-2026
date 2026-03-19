# Safety View Agent - クイックスタート

5分で Safety View Agent を始められます。

## 前提条件

- Python 3.12+
- `uv` パッケージマネージャー
- Slurm クラスタ（GPU 実行時）

## セットアップ（1分）

```bash
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

### manual モード

動画ファイルを `data/videos/` に配置して使用：

```bash
cp your_video.mp4 data/videos/
```

### inspesafe モード

`configs/default.yaml` でセッションを指定：

```yaml
data:
  mode: "inspesafe"
  inspesafe:
    dataset_path: "/home/team-005/data/hf_cache/hub/datasets--Tetrabot2026--InspecSafe-V1/snapshots/..."
    session: "train/Other_modalities/58132919535743_20251118_session_1400_2#bowenguanshang-you"
```

**詳細セットアップ:** [INSPESAFE_INTEGRATION.md](INSPESAFE_INTEGRATION.md)

## 実行方法

### オプション A: LLM なし（テスト・動作確認用）

vLLM サーバーなしでも動作します。LLM 未接続時はフォールバックとして
`risk_level="low"`, `action_type="monitor"` の固定値で全フレーム処理されます。

```bash
uv run python src/run.py
```

### オプション B: OpenAI API で実行

```bash
# APIキーを設定
export OPENAI_API_KEY="sk-your-api-key-here"

# configs/default.yaml の llm.provider を "openai" に変更してから実行
uv run python src/run.py
```

`.env` ファイルで管理することも可能：

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
  enable_infrared: true     # 赤外線解析の有効/無効
  enable_temporal: true     # 時系列解析の有効/無効
  context_history_size: 0   # 前回判断の参照数（0=なし）
```

### LLM 設定

```yaml
llm:
  provider: "vllm"   # "vllm"（ローカル）または "openai"（API）
  vllm:
    base_url: "http://localhost:8000/v1"
    model: "Qwen/Qwen3.5-9B"
```

## テスト実行

```bash
# E2E テスト（LLM 不要）
pytest tests/test_e2e.py -v
```

期待される結果：
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

### perception_results.json の構造

```json
{
  "frames": [
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

**フォーマットの特徴**:
- `frames` 配列で統一（フラット構造）
- 各フレームは独立した判断結果を含む
- `vision_analysis`, `depth_analysis`, `infrared_analysis`, `temporal_analysis` は各モダリティの結果
- `belief_state` は複数フレームにわたる危険状態の管理
- `assessment` は LLM による総合安全判断
- `errors` でモダリティ処理のエラーを記録

結果確認のワンライナー：

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

## TTS（音声案内）出力

エージェント実行後、フレームごとの状況報告を Qwen2.5-TTS で音声合成できます。

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

### `ConnectError: Connection refused`

vLLM サーバーが起動していません。エージェント実行前にサーバーを起動してください：

```bash
sbatch slurm/vllm_qwen3_light.sh
# サーバー ready を確認してから run_gpu.sh を実行
curl http://localhost:8000/v1/models
```

### `depth: VLM analysis failed`

VLM も LLM と同じ port 8000 を使用します。`vllm_qwen3_light.sh` を先に起動してください。

### `No frames found`

```bash
# manual モードの場合、data/videos/ に動画を配置
cp your_video.mp4 data/videos/
```

### テスト失敗

```bash
uv sync --force
pytest tests/test_e2e.py -v
```

## 次のステップ

- 📖 [システムアーキテクチャ](./ARCHITECTURE.md)
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
