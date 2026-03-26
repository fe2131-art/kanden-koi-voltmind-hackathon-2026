# Safety View Agent Setup

このガイドは、**このリポジトリを今の実装どおりに動かすための前提条件** をまとめたものです。
とくに `pyproject.toml` は local editable dependency を前提にしているため、外部 repo を clone せずに `uv sync` すると失敗します。

## 必須ツール

- Python 3.12
- `uv`
- `git`
- `ffmpeg` / `ffprobe`

### オプション

- OpenAI API キー
  - OpenAI provider を使う場合
- Node.js 18+
  - React デモ UI を使う場合
- CUDA 対応 GPU
  - vLLM、Depth-Anything-3、SAM3、Kokoro TTS を高速に動かしたい場合

## 1. リポジトリを clone

```bash
git clone https://github.com/fe2131-art/kanden-koi-voltmind-hackathon-2026.git
cd kanden-koi-voltmind-hackathon-2026
```

## 2. 外部依存 repo を clone

`pyproject.toml` は以下の 3 つを `external/` 配下に置く前提です。

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git external/Depth-Anything-3
git clone https://github.com/facebookresearch/sam3.git external/sam3
git clone https://github.com/vllm-project/vllm-omni.git external/vllm-omni
```

## 3. 互換性 patch を適用

Depth-Anything-3 と SAM3 は、このリポジトリ側の patch を当てる前提です。

```bash
git -C external/Depth-Anything-3 apply ../../scripts/depth_anything_3/patches/da3-numpy-compatibility.patch
git -C external/sam3 apply ../../scripts/sam3/patches/sam3-numpy-compatibility.patch
```

既に適用済みで失敗する場合は、そのまま次へ進んで問題ありません。

Depth-Anything-3 側だけは補助スクリプトもあります。

```bash
bash scripts/depth_anything_3/setup_external_deps.sh
```

## 4. 依存関係をインストール

コア開発用（Safety View Agent 全機能）:

```bash
uv sync --extra dev
```

動画生成のみ（外部 repo 不要）:

```bash
uv sync --extra video_generation
```

動画生成 + Safety View Agent 両方:

```bash
uv sync --extra dev --extra video_generation
```

React デモも使う場合:

```bash
uv sync --extra dev --extra demo
```

## 5. `ffmpeg` / `ffprobe` を確認

動画分割、音声抽出、InspecSafe のデモ動画生成に必要です。

```bash
ffmpeg -version
ffprobe -version
```

どちらも見つからない場合は、先にシステムへインストールしてください。
詳細は [UV_VENV.md](./UV_VENV.md) も参照してください。

## 6. 環境変数を設定

OpenAI provider を使う場合のみ必須です。

```bash
cp .env.example .env
```

`.env` の例:

```env
OPENAI_API_KEY=sk-...
```

以後は `uv run python src/run.py` 実行時に自動で読み込まれます。

## 7. `configs/default.yaml` を自分の環境に合わせる

そのままの既定値はチーム環境前提です。少なくとも以下を確認してください。

### 入力モード

- InspecSafe を使わない場合
  - `data.mode: "manual"` に変更
- InspecSafe を使う場合
  - `data.inspesafe.dataset_path`
  - `data.inspesafe.session`

### Provider

既定値は `llm.provider / vlm.provider / alm.provider: "vllm"` です。
ローカル vLLM サーバーが無い場合は、以下のどちらかにしてください。

- OpenAI を使う
  - 各 provider を `openai` に変更し、`OPENAI_API_KEY` を設定
- まずスモークテストだけしたい
  - 各 provider を `openai` に変更し、`OPENAI_API_KEY` は設定しない
  - この場合、VLM/ALM/LLM は無効化またはフォールバックされ、パイプライン確認に使えます

### SAM3

- 実行の ON/OFF は `agent.enable_sam3`
- `sam3:` セクションは threshold / prompts / mask 保存先などの analyzer 設定
- `sam3.checkpoint_path` の既定値はチーム環境のローカルパスです
- 他環境では **公式 checkpoint (`facebook/sam3`)** を使う前提で見直してください
- `checkpoint_path: null` にすると Hugging Face 上の公式 `facebook/sam3` から自動取得します
- `facebook/sam3` は gated model のため、事前に Hugging Face でアクセス承認を受け、`hf auth login` などで認証が必要です

設定例:

```yaml
agent:
  enable_sam3: true

sam3:
  checkpoint_path: null
  score_threshold: 0.35
  max_regions_per_prompt: 2
```

### Demo UI 用の補足

- `src/apps/server.py` を使うなら `uv sync --extra demo` が必要です
- React 側は別途 `cd src/apps && npm install` が必要です

## 8. 動作確認

### テスト

```bash
uv run pytest tests/ -v
```

### 実行

```bash
uv run python src/run.py
```

成功すると、少なくとも以下が生成されます。

- `data/perception_results/manifest.json`
- `data/perception_results/frames/*.json`
- `data/flow.md`

モダリティ設定に応じて以下も生成されます。

- `data/depth/`
- `data/infrared_frames/`
- `data/sam3_masks/`
- `data/voice/`

## よくある詰まりどころ

- `uv sync` が失敗する
  - `external/Depth-Anything-3` / `external/sam3` / `external/vllm-omni` のどれかが無い可能性が高いです
- `session が見つかりません`
  - `data.mode` が `inspesafe` のままです
- `ffmpeg not available`
  - システムパッケージとしての `ffmpeg` が未導入です
- `Sam3Analyzer: model load failed`
  - `checkpoint_path` が無効か、依存セットアップが未完了です

詳細は [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) を参照してください。
