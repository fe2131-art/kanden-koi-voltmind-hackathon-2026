# Safety View Agent Quick Start

このガイドは、**今の実装で最短で動作確認するための手順** です。
まだ環境構築をしていない場合は、先に [SETUP.md](./SETUP.md) を完了してください。

## 最短スモークテスト

外部依存の clone と `uv sync` は済んでいる前提です。
まずはモデルサーバー無しでパイプラインだけ確認したい場合、次の設定が最短です。

### 1. `configs/default.yaml` を最小構成にする

```yaml
data:
  mode: "manual"

agent:
  max_steps: 3

llm:
  provider: "openai"

vlm:
  provider: "openai"

alm:
  provider: "openai"
```

この状態で `OPENAI_API_KEY` を設定しなければ、LLM / VLM / ALM は無効化またはフォールバックされます。
パイプライン、出力保存、アーカイブ、デモ連携の確認に使えます。

### 2. 入力を置く

#### 動画を使う場合

```bash
cp your_video.mp4 data/videos/
```

#### 静止画フレームを使う場合

```bash
cp frame_0.0s.jpg data/frames/
```

### 3. 実行

```bash
uv run python src/run.py
```

### 4. 出力を確認

```bash
ls data/perception_results/frames
```

成功すると、以下が生成されます。

- `data/perception_results/manifest.json`
- `data/perception_results/frames/*.json`
- `data/flow.md`

## OpenAI で実行する

### 1. `.env` を設定

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
```

### 2. provider を OpenAI にする

```yaml
llm:
  provider: "openai"

vlm:
  provider: "openai"

alm:
  provider: "openai"
```

### 3. 実行

```bash
uv run python src/run.py
```

## vLLM で実行する

既定の `configs/default.yaml` は vLLM 前提です。

### 1. サーバーを起動

#### LLM / VLM 用

```bash
python -m vllm.entrypoints.openai.api_server \
  --model QuantTrio/Qwen3.5-9B-AWQ \
  --allowed-local-media-path /path/to/kanden-koi-voltmind-hackathon-2026 \
  --port 8000
```

#### ALM 用

既定値のまま `agent.enable_audio: true` で実行するなら、音声用サーバーも必要です。

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Omni-3B \
  --allowed-local-media-path /path/to/kanden-koi-voltmind-hackathon-2026 \
  --port 8002
```

`modality_nodes.py` の vLLM 経路は `file://` ベースでローカル画像・一時 WAV を渡すため、
`--allowed-local-media-path` は少なくともリポジトリの `data/` と `tmp/` を含むパスにしてください。
迷ったらリポジトリルート全体を許可するのが一番簡単です。

音声サーバーを立てない場合は、次のどちらかにしてください。

- `agent.enable_audio: false`
- `alm.provider: "openai"` に切り替えて OpenAI を使う

### 2. 必要なら `configs/default.yaml` を見直す

- `data.mode`
- `data.inspesafe.dataset_path`
- `sam3.checkpoint_path` (`null` なら公式 `facebook/sam3` を取得。ただし Hugging Face のアクセス承認と認証が必要)

### 3. 実行

```bash
uv run python src/run.py
```

## InspecSafe-V1 で実行する

```yaml
data:
  mode: "inspesafe"
  inspesafe:
    dataset_path: "/path/to/InspecSafe-V1"
    session: "train/Other_modalities/your_session"
```

```bash
uv run python src/run.py
```

`run.py` は次を自動で行います。

- RGB 動画を `data/frames/` へ展開
- 赤外線動画を `data/infrared_frames/` へ展開
- 音声を `data/audio/audio.wav` へコピー
- デモ用 `data/videos/video.mp4` を生成

詳細は [INSPESAFE_INTEGRATION.md](./INSPESAFE_INTEGRATION.md) を参照してください。

## 出力ディレクトリ

実行後の主な出力は以下です。

```text
data/
├── perception_results/
│   ├── manifest.json
│   └── frames/
├── results_archive/
├── frames/
├── audio/
├── depth/            # enable_depth=true の場合
├── infrared_frames/  # 赤外線入力がある場合
├── sam3_masks/       # agent.enable_sam3=true の場合
├── voice/            # tts.enabled=true の場合
└── flow.md
```

### 補足

- 新しい実行を始めると、既存の `data/perception_results/` は `data/results_archive/<timestamp>/` に退避されます
- `data/videos/video.mp4` は Demo UI 用です
- `manual` モードでも `data/videos/` に MP4 が 1 本あれば Demo UI で再生できます
- `src/apps/server.py` が接続時に実ファイル名を UI へ渡すため、`video.mp4` という名前に揃える必要はありません

## デモ UI を見る

```bash
uv sync --extra dev --extra demo
cd src/apps
npm install
```

別ターミナルで:

```bash
uv run python src/apps/server.py
```

さらに別ターミナルで:

```bash
cd src/apps
npm run dev
```

詳細は [DEMO_APP.md](./DEMO_APP.md) を参照してください。

## 困ったとき

- `uv sync` が失敗する → [SETUP.md](./SETUP.md)
- `session が見つからない` → `data.mode` を確認
- `Sam3Analyzer: model load failed` → `sam3.checkpoint_path`、Hugging Face 認証、外部依存を確認
- `Connection refused` → vLLM ポート設定を確認

詳しくは [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) を参照してください。
