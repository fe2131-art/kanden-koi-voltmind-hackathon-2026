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
| **inspesafe** | InspecSafe-V1 データセット | 産業用データセット |

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
cp .env.example .env
# .env に OPENAI_API_KEY を記入
uv run python src/run.py  # .env を自動読み込み
```

### オプション C: ローカル LLM（vLLM / Slurm）推奨

> **注意**: サーバーを起動してからエージェントを実行してください。
> 起動前に実行すると `Connection refused` エラーになります。

#### 使用モデルとポート

| 役割 | モデル | ポート | Slurm スクリプト |
|------|--------|--------|-----------------|
| LLM + VLM | Qwen/Qwen3.5-9B | 8000 | `slurm/vllm_qwen3_light.sh` |
| ALM（音声） | Qwen/Qwen2.5-Omni-7B | 8001 | `slurm/vllm_qwen2_audio.sh` |

#### 手順

```bash
# 1. LLM + VLM サーバーを起動（port 8000）
sbatch slurm/vllm_qwen3_light.sh

# 2. サーバーが ready になるまで待機（1〜2分）
curl http://localhost:8000/v1/models  # 応答が返ればOK

# 3. （音声解析も使う場合）ALM サーバーを起動（port 8001）
sbatch slurm/vllm_qwen2_audio.sh

# 4. エージェント実行
sbatch slurm/run_gpu.sh
```

サーバーを止めるには:

```bash
squeue -u $USER          # JOBID を確認
scancel <JOBID>
```

## 設定オプション（`configs/default.yaml`）

### エージェント設定

```yaml
agent:
  max_steps: -1             # 処理フレーム数（-1: 全フレーム、N: 最初のNフレームのみ）
  enable_audio: true        # 音声解析の有効/無効
  enable_depth: true        # 深度推定の有効/無効
  context_history_size: 1   # 前回判断の参照（0=なし, 1=前回のみ、推奨）
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
│   ├── frame_0.0s.jpg
│   ├── frame_1.0s.jpg
│   └── ...
├── audio/
│   └── audio.wav
├── depth/
├── voice/                           ← TTS 音声出力（qwen_tts.sh 実行後）
│   ├── frame_0.0s.wav
│   ├── frame_1.0s.wav
│   └── ...
├── perception_results.json          ← 分析結果（JSON）
└── results_archive/                 ← 実行ごとに自動アーカイブ
```

### perception_results.json の構造

```json
{
  "frames": [
    {
      "frame_id": "img_0",
      "video_timestamp": 0.0,
      "assessment": {
        "risk_level": "low",
        "action_type": "monitor",
        "reason": "...",
        "safety_status": "継続観測中",
        "detected_hazards": [],
        "priority": 0.0,
        "temporal_status": "unknown"
      },
      "vision_analysis": {
        "scene_description": "...",
        "critical_points": [],
        "blind_spots": []
      },
      "audio": [],
      "depth_analysis": null,
      "errors": []
    }
  ]
}
```

結果確認のワンライナー：

```bash
python -c "import json; d=json.load(open('data/perception_results.json')); print(f'Frames: {len(d[\"frames\"])}')"
```

## TTS（音声案内）出力

エージェント実行後、フレームごとの状況報告を Qwen2.5-TTS で音声合成できます。

```bash
# ドライラン（テキスト確認のみ、GPU不要）
uv run python src/tts/synthesize.py --dry-run

# GPU で音声合成実行
sbatch slurm/qwen_tts.sh
# → data/voice/frame_0.0s.wav, frame_1.0s.wav ... が生成される
```

TTS モデルは `configs/default.yaml` の `tts` セクションで設定します：

```yaml
tts:
  model: "Qwen/Qwen2.5-TTS-3B-Instruct"
  sample_rate: 24000
  voice: "Chelsie"
```

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
