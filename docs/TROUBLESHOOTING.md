# Troubleshooting

Safety View Agent でよく起きる問題を、現在の実装前提でまとめています。

## `uv sync` が失敗する

### `Distribution not found at: file:///.../external/Depth-Anything-3`

原因:

- `external/Depth-Anything-3` が無い

解決:

```bash
git clone https://github.com/DepthAnything/Depth-Anything-3.git external/Depth-Anything-3
git -C external/Depth-Anything-3 apply ../../scripts/depth_anything_3/patches/da3-numpy-compatibility.patch
uv sync --extra dev
```

### `Distribution not found at: file:///.../external/sam3`

原因:

- `external/sam3` が無い

解決:

```bash
git clone https://github.com/facebookresearch/sam3.git external/sam3
git -C external/sam3 apply ../../scripts/sam3/patches/sam3-numpy-compatibility.patch
uv sync --extra dev
```

### `Distribution not found at: file:///.../external/vllm-omni`

原因:

- `external/vllm-omni` が無い

解決:

```bash
git clone https://github.com/vllm-project/vllm-omni.git external/vllm-omni
uv sync --extra dev
```

## `FileNotFoundError: セッションが見つかりません`

原因:

- `configs/default.yaml` の `data.mode` が `inspesafe`
- しかし `data.inspesafe.dataset_path` / `session` が自分の環境に合っていない

解決:

- InspecSafe を使わないなら `data.mode: "manual"`
- InspecSafe を使うなら `dataset_path` と `session` を修正

## `OPENAI_API_KEY not set`

原因:

- provider が `openai`
- しかし `.env` または環境変数が未設定

補足:

- `llm.provider: openai` の場合、LLM はフォールバックされます
- `vlm.provider: openai` / `alm.provider: openai` の場合、VLM / ALM は無効化されます

解決:

```bash
cp .env.example .env
```

`.env`:

```env
OPENAI_API_KEY=sk-...
```

## vLLM に接続できない

### `Connection refused` / タイムアウト

原因:

- `llm.vllm.base_url` のサーバーが起動していない
- `vlm.vllm.base_url` または `alm.vllm.base_url` が違う

確認:

```bash
cat configs/default.yaml
```

既定値:

- LLM / VLM: `http://localhost:8000/v1`
- ALM: `http://localhost:8002/v1`

解決:

- 8000 のサーバーを起動
- 音声を有効にしたままなら 8002 も起動
- あるいは `agent.enable_audio: false`

### つながるのに画像 / 音声入力だけ失敗する

原因:

- vLLM 起動時に `--allowed-local-media-path` を付けていない

補足:

- この実装の vLLM 経路は `file://` でローカル画像と一時 WAV を渡します
- 少なくともリポジトリの `data/` と `tmp/` を許可する必要があります

解決例:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model QuantTrio/Qwen3.5-9B-AWQ \
  --allowed-local-media-path /path/to/kanden-koi-voltmind-hackathon-2026 \
  --port 8000
```

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Omni-3B \
  --allowed-local-media-path /path/to/kanden-koi-voltmind-hackathon-2026 \
  --port 8002
```

## `ffprobe not available` / `ffmpeg not available`

原因:

- システムに `ffmpeg` が入っていない

影響:

- 動画から音声抽出できない
- InspecSafe の `data/videos/video.mp4` 生成で音声 mux ができない

解決:

```bash
ffmpeg -version
ffprobe -version
```

未導入ならシステムパッケージとしてインストールしてください。

## `Sam3Analyzer: model load failed`

原因候補:

- `external/sam3` が無い
- patch 未適用
- `sam3.checkpoint_path` が無効
- GPU / 依存関係の問題

確認ポイント:

- 実行スイッチは `agent.enable_sam3`
- `sam3.enabled` ではありません
- `sam3.checkpoint_path` の既定値はチーム環境のローカルパスです

解決:

- `external/sam3` を clone
- patch を適用
- `sam3.checkpoint_path: null` にして Hugging Face から取得させる
- もしくは自分の checkpoint パスに変更する

## Demo UI で結果が出ない

### WebSocket がつながらない

確認:

```bash
uv run python src/apps/server.py
```

### フレームが増えない

確認:

```bash
cat data/perception_results/manifest.json
```

### 動画は見えるが BBox が出ない

原因:

- `vision_analysis.critical_points` に `normalized_bbox` が無い

補足:

- Demo UI のオーバーレイは `grounded_critical_points` ではなく `critical_points` を使います

### 音声が再生されない

確認:

- `tts.enabled: true`
- `data/voice/` に WAV がある

## `data/perception_results/` が消えたように見える

原因:

- `run.py` は新しい実行の前に既存の `data/perception_results/` を `data/results_archive/<timestamp>/` へ移動します

確認:

```bash
ls data/results_archive
```

## `python src/run.py` は動くのに Demo UI だけ失敗する

原因候補:

- `uv sync --extra demo` をしていない
- `npm install` をしていない

解決:

```bash
uv sync --extra dev --extra demo
cd src/apps
npm install
```

## まだ解決しないとき

次の順で確認すると切り分けしやすいです。

1. [SETUP.md](./SETUP.md) の外部依存と patch
2. `configs/default.yaml` の `data.mode` と provider
3. `ffmpeg` / `ffprobe`
4. `uv run pytest tests/ -v`
5. `uv run python src/run.py`
