# Safety View Agent - Project Guide

このファイルは、Safety View Agent の現在の実装に合わせた高信号ガイドです。
詳細な手順は `docs/` に分離し、ここでは構成・前提・重要コマンド・ハマりどころだけをまとめます。

## プロジェクト概要

Safety View Agent は、LangGraph ベースのマルチモーダル安全監視エージェントです。

- RGB 画像
- 音声
- Depth-Anything-3
- 赤外線フレーム
- 時系列差分
- SAM3 セグメンテーション

を統合して、`BeliefState` と `ActionWithGrounding` を生成します。

## 重要な前提

### 1. `uv sync` の前に外部 repo が必要

`pyproject.toml` は local editable dependency を前提にしています。

- `external/Depth-Anything-3`
- `external/sam3`
- `external/vllm-omni`

これらが無いと `uv sync` に失敗します。

### 2. 既定の `configs/default.yaml` はチーム環境向け

既定値のままだと、以下を前提にしています。

- `data.mode: "inspesafe"`
- `llm.provider / vlm.provider / alm.provider: "vllm"`
- `agent.enable_sam3: true`
- `sam3.checkpoint_path` はチーム環境のローカルパス

そのため、新しい環境では最初に `configs/default.yaml` を見直す必要があります。

### 3. `agent.enable_sam3` が実行スイッチ

`sam3:` セクションは analyzer の設定で、実行 ON/OFF は `agent.enable_sam3` 側です。

## よく使うコマンド

### セットアップ

```bash
uv sync --extra dev
```

Demo UI も使う場合:

```bash
uv sync --extra dev --extra demo
cd src/apps
npm install
```

### テスト

```bash
uv run pytest tests/ -v
```

### 実行

```bash
uv run python src/run.py
```

### Demo UI

```bash
uv run python src/apps/server.py
```

```bash
cd src/apps
npm run dev
```

## 主なファイル

- `src/run.py`
  - 設定読み込み
  - 入力データ準備
  - analyzer 初期化
  - 結果保存
  - TTS 実行
- `src/safety_agent/agent.py`
  - LangGraph の本体
- `src/safety_agent/modality_nodes.py`
  - Vision / Audio / Depth / Infrared / Temporal / SAM3
- `src/safety_agent/schema.py`
  - Pydantic スキーマ
- `configs/default.yaml`
  - 実行設定
- `configs/prompt.yaml`
  - 各モダリティと最終判断 prompt
- `src/apps/server.py`
  - WebSocket ストリーミング
- `src/apps/App.jsx`
  - Demo UI

## 入力モード

### manual

- `data/videos/` の動画
- または `data/frames/` の既存フレーム

動画がある場合は lazy frame extraction が使われます。

### inspesafe

- `data.inspesafe.dataset_path`
- `data.inspesafe.session`

をもとに RGB / 赤外線 / 音声を自動展開します。

## 出力

主な出力先:

```text
data/
├── perception_results/
│   ├── manifest.json
│   └── frames/*.json
├── results_archive/
├── frames/
├── depth/
├── infrared_frames/
├── sam3_masks/
├── voice/
└── flow.md
```

補足:

- 新しい実行の前に既存の `data/perception_results/` は `data/results_archive/` に退避されます
- `assessment.safety_status` は TTS 有効時に `data/voice/*.wav` になります
- Demo UI は `manifest.json` を監視して新規フレームだけ読み込みます

## 実装上のポイント

- `determine_next_action_llm()` は `ActionWithGrounding` を返します
  - `assessment`
  - `grounded_critical_points`
- `configs/prompt.yaml` の `safety_assessment` は最終判断だけでなく grounding も担います
- Demo UI のオーバーレイは `vision_analysis.critical_points` の `normalized_bbox` を使います
- `assessment.target_region` は UI ハイライトにも影響します

## よくある詰まりどころ

- `uv sync` が失敗する
  - `external/` の repo が足りません
- `session が見つかりません`
  - `data.mode: inspesafe` のままです
- `Connection refused`
  - vLLM サーバーが起動していません
- `Sam3Analyzer: model load failed`
  - patch / checkpoint / 依存が怪しいです
- `ffprobe not available`
  - ffmpeg が未導入です

## 関連ドキュメント

- [docs/README.md](./docs/README.md)
- [docs/SETUP.md](./docs/SETUP.md)
- [docs/QUICK_START.md](./docs/QUICK_START.md)
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)
- [docs/DEMO_APP.md](./docs/DEMO_APP.md)
- [docs/INSPESAFE_INTEGRATION.md](./docs/INSPESAFE_INTEGRATION.md)
- [docs/EXTENDING.md](./docs/EXTENDING.md)
- [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)
- [docs/UV_VENV.md](./docs/UV_VENV.md)
