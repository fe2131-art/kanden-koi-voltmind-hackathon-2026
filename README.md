# Safety View Agent

Safety View Agent は、LangGraph ベースのマルチモーダル安全監視エージェントです。
RGB 画像、音声、深度、赤外線、時系列差分、SAM3 セグメンテーションを統合し、産業設備や巡回点検の映像から危険状態を要約して `BeliefState` と `ActionWithGrounding` を生成します。

詳細ドキュメントは [docs/README.md](./docs/README.md) に整理しています。この README では、ハッカソン納品向けに全体像と必要事項を 1 か所で確認できるようにまとめています。

## プロジェクト概要

このリポジトリは、産業現場の巡回動画や InspecSafe-V1 セッションを入力として、

- 各フレームの視覚・音声・深度・赤外線・時系列変化を分析する
- フレームごとの安全判断と根拠領域を出力する
- 継続的な危険状態を `BeliefState` として追跡する
- 結果を JSON と Demo UI で確認できる形に保存する

ためのコードです。

関連詳細:

- 全体設計: [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)
- 実行方法: [docs/QUICK_START.md](./docs/QUICK_START.md)
- セットアップ: [docs/SETUP.md](./docs/SETUP.md)

## 主な機能 / ユースケース

- 産業設備の巡回動画から危険箇所をフレーム単位で検出・要約
- RGB、音声、深度、赤外線、時系列差分、SAM3 のマルチモーダル統合
- InspecSafe-V1 セッションの直接処理
- `data/perception_results/` への逐次保存と `data/results_archive/` への自動退避
- React + WebSocket Demo UI による可視化
- TTS による `assessment.safety_status` の音声化
- LTX-2 を使ったテスト用動画生成
- InspecSafe 音声への背景音・異常音合成

詳細:

- Demo UI: [docs/DEMO_APP.md](./docs/DEMO_APP.md)
- InspecSafe 連携: [docs/INSPESAFE_INTEGRATION.md](./docs/INSPESAFE_INTEGRATION.md)
- 動画生成: [docs/VIDEO_GENERATION.md](./docs/VIDEO_GENERATION.md)
- 音声加工: [docs/AUDIO_PROCESSING.md](./docs/AUDIO_PROCESSING.md)

## 技術スタック

### 言語

- Python 3.12
- JavaScript / JSX
- YAML

### 主なライブラリ / フレームワーク

- LangGraph
- Pydantic
- httpx
- OpenCV
- vLLM
- Hugging Face Hub
- React
- Vite
- websockets
- diffusers
- PyTorch / torchaudio
- librosa / soundfile
- Kokoro TTS

依存関係の詳細は [pyproject.toml](./pyproject.toml) と [docs/SETUP.md](./docs/SETUP.md) を参照してください。

## モデル情報

このリポジトリは複数のモデルを用途別に使い分けます。既定値は `configs/default.yaml` と `configs/gen_prompt.yaml` にあります。

| 役割 | 既定 / 代表モデル | 取得先 |
|---|---|---|
| LLM / VLM（vLLM） | `QuantTrio/Qwen3.5-9B-AWQ` | [Hugging Face](https://huggingface.co/QuantTrio/Qwen3.5-9B-AWQ) |
| ALM（vLLM） | `Qwen/Qwen2.5-Omni-3B` | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) |
| OpenAI 代替 provider | `gpt-5-nano` | [OpenAI Models Docs](https://platform.openai.com/docs/models) |
| Depth 推定 | Depth-Anything-3 | [GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3) |
| セグメンテーション | SAM 3 | [GitHub](https://github.com/facebookresearch/sam3) |
| SAM3 checkpoint 例 | `facebook/sam3` | [Hugging Face](https://huggingface.co/facebook/sam3) |
| TTS | `hexgrad/Kokoro-82M` | [Hugging Face](https://huggingface.co/hexgrad/Kokoro-82M) |
| 動画生成 | `Lightricks/LTX-2` | [Hugging Face](https://huggingface.co/Lightricks/LTX-2) |

補足:

- Safety View Agent 本体の既定 provider は `vllm` です
- OpenAI provider は任意で切り替え可能です
- SAM3 は `agent.enable_sam3` が実行スイッチです
- SAM3 checkpoint は公式の `facebook/sam3` を前提にしています
- `facebook/sam3` は Hugging Face の gated model なので、利用前にアクセス承認と `hf auth login` などの認証が必要です
- 動画生成で使う LTX-2 は本体パイプラインと独立しています

詳細:

- 本体推論: [docs/QUICK_START.md](./docs/QUICK_START.md)
- セットアップと provider 設定: [docs/SETUP.md](./docs/SETUP.md)
- 動画生成: [docs/VIDEO_GENERATION.md](./docs/VIDEO_GENERATION.md)

## 使用データの概要と取得方法

### 1. InspecSafe-V1

本体の `inspesafe` モードでは、InspecSafe-V1 のセッションを直接処理します。

- 代表モダリティ: 可視動画、赤外線動画、音声
- 想定入力: `DATA_PATH/<split>/Other_modalities/<session>`
- 取得先: [Tetrabot2026/InspecSafe-V1](https://huggingface.co/datasets/Tetrabot2026/InspecSafe-V1)

このリポジトリにはデータセット本体は含まれません。ローカルに展開した上で `data.inspesafe.dataset_path` と `session` を設定してください。

詳細: [docs/INSPESAFE_INTEGRATION.md](./docs/INSPESAFE_INTEGRATION.md)

### 2. manual モード入力

手元の動画やフレーム画像を直接使うこともできます。

- 動画: `data/videos/`
- 静止画: `data/frames/`

詳細: [docs/QUICK_START.md](./docs/QUICK_START.md)

### 3. 補助データ

- `audio_processing` 用の背景音 / 異常音素材は `data/audio_materials/` に配置します
- LTX-2 モデル重みや公式 SAM3 checkpoint（`facebook/sam3`）は別途取得が必要です

詳細:

- [docs/AUDIO_PROCESSING.md](./docs/AUDIO_PROCESSING.md)
- [docs/VIDEO_GENERATION.md](./docs/VIDEO_GENERATION.md)

## セットアップ手順

最短のセットアップ手順:

1. リポジトリを clone
2. `external/Depth-Anything-3`, `external/sam3`, `external/vllm-omni` を配置
3. 必要な patch を適用
4. `uv sync --extra dev` を実行
5. `ffmpeg` / `ffprobe` を確認
6. `configs/default.yaml` を自分の環境向けに修正

詳細な手順は [docs/SETUP.md](./docs/SETUP.md) を参照してください。

## 実行方法（再現手順）

### 最短スモークテスト

1. [docs/SETUP.md](./docs/SETUP.md) を完了
2. `configs/default.yaml` を `manual + openai provider` の最小構成へ変更
3. 動画を `data/videos/` またはフレームを `data/frames/` に配置
4. `uv run python src/run.py`
5. `data/perception_results/manifest.json` と `data/perception_results/frames/*.json` を確認

### Demo UI を使う場合

1. `uv sync --extra dev --extra demo`
2. `cd src/apps && npm install`
3. `uv run python src/apps/server.py`
4. `cd src/apps && npm run dev`
5. 別ターミナルで `uv run python src/run.py`

詳細:

- 実行全般: [docs/QUICK_START.md](./docs/QUICK_START.md)
- Demo UI: [docs/DEMO_APP.md](./docs/DEMO_APP.md)
- トラブル時: [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)

## ディレクトリ構成

```text
.
├── audio_processing/   # InspecSafe 音声合成スクリプト
├── configs/            # 実行設定と prompt
├── data/               # 入出力データ置き場
├── dataset/            # 補助データ / データ関連資産
├── docs/               # 詳細ドキュメント
├── finetuning/         # ファインチューニング関連
├── scripts/            # 外部依存セットアップや補助スクリプト
├── slurm/              # ジョブ投入関連
├── src/                # 本体コードと Demo UI
├── tests/              # テスト
├── video_generation/   # LTX-2 動画生成
├── CLAUDE.md           # プロジェクトガイド
├── LICENSE             # ライセンス本文
├── NOTICE              # 著作権表示
└── pyproject.toml      # Python パッケージ / 依存定義
```

主要ファイルの詳細は [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) と [docs/README.md](./docs/README.md) を参照してください。

## 制約・注意事項

- `uv sync` 前に `external/Depth-Anything-3`, `external/sam3`, `external/vllm-omni` が必要です
- 既定の `configs/default.yaml` はチーム環境前提です
- `data.mode: "inspesafe"`、`vllm` provider、`agent.enable_sam3: true` が既定です
- `sam3.checkpoint_path` や `configs/gen_prompt.yaml` の `local_path` は環境依存値を含みます
- `sam3.checkpoint_path: null` は公式 `facebook/sam3` の自動取得ですが、事前に Hugging Face 側でアクセス承認と認証が必要です
- `ffmpeg` / `ffprobe` はシステムパッケージとして別途必要です
- vLLM、Depth-Anything-3、SAM3、Kokoro TTS、LTX-2 は GPU があると大幅に高速化します
- LTX-2 は H100 クラスの GPU と大きな VRAM を前提にした設定例を含みます
- サードパーティのモデル、checkpoint、データセットはこのリポジトリに同梱されません

詳細:

- セットアップ制約: [docs/SETUP.md](./docs/SETUP.md)
- 実行時の注意: [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)
- 動画生成の計算資源: [docs/VIDEO_GENERATION.md](./docs/VIDEO_GENERATION.md)

## ライセンス

このリポジトリ内のソースコード、ドキュメント、設定ファイルは Apache License 2.0 です。

- ライセンス本文: [LICENSE](./LICENSE)
- 著作権表示: [NOTICE](./NOTICE)

補足:

- サードパーティのモデル、checkpoint、データセット、外部リポジトリは元ライセンスに従います
- 詳細な注意書きは [CLAUDE.md](./CLAUDE.md) にも記載しています

