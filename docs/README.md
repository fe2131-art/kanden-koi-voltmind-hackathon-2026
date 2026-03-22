# Safety View Agent Docs

この `docs/` は、現在の実装に合わせて整理したドキュメント集です。
セットアップ、実行、UI デモ、InspecSafe 連携、拡張ポイントをここから辿れます。

## 最初に読む順番

1. [SETUP.md](./SETUP.md)
   - 外部依存の clone、patch 適用、`uv sync`、ffmpeg などの前提を確認します
2. [QUICK_START.md](./QUICK_START.md)
   - `manual` / `inspesafe`、OpenAI / vLLM / フォールバック実行の最短手順を見ます
3. [ARCHITECTURE.md](./ARCHITECTURE.md)
   - LangGraph の fan-out/fan-in、BeliefState、SAM3、出力構造を理解します

## ドキュメント一覧

- [SETUP.md](./SETUP.md)
  - 実行環境の構築、外部依存、SAM3/Depth-Anything-3、環境変数
- [QUICK_START.md](./QUICK_START.md)
  - 最短実行手順と結果の見方
- [ARCHITECTURE.md](./ARCHITECTURE.md)
  - システム構成、主要ファイル、データフロー、出力構造
- [DEMO_APP.md](./DEMO_APP.md)
  - React + WebSocket デモ UI の起動方法と確認ポイント
- [INSPESAFE_INTEGRATION.md](./INSPESAFE_INTEGRATION.md)
  - `data.mode: inspesafe` の設定とデータセット構造
- [EXTENDING.md](./EXTENDING.md)
  - 新モダリティ追加、prompt/schema 拡張、UI 拡張の入口
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
  - `uv sync` 失敗、ffmpeg 不足、vLLM 接続失敗、SAM3 チェックポイント問題など
- [UV_VENV.md](./UV_VENV.md)
  - `uv run` と仮想環境の使い分け
- [../CLAUDE.md](../CLAUDE.md)
  - AI コーディングツール向けの高信号プロジェクトガイド

## 現在の実装前提

- `pyproject.toml` は以下の local editable dependency を前提にしています
  - `external/Depth-Anything-3`
  - `external/sam3`
  - `external/vllm-omni`
- 既定の `configs/default.yaml` は以下を前提にしています
  - `data.mode: "inspesafe"`
  - `llm.provider / vlm.provider / alm.provider: "vllm"`
  - `agent.enable_sam3: true`
- 初回セットアップなしでそのまま `uv sync` や `uv run python src/run.py` を実行すると、環境によっては失敗します

## 代表的な作業パス

### 1. まずローカルでパイプラインだけ確認したい

- [SETUP.md](./SETUP.md) で外部依存と `uv sync` を完了
- [QUICK_START.md](./QUICK_START.md) の「最短スモークテスト」を実行

### 2. OpenAI か vLLM で実際に推論したい

- [SETUP.md](./SETUP.md) の provider 設定を確認
- [QUICK_START.md](./QUICK_START.md) の OpenAI / vLLM 手順へ

### 3. ブラウザ UI で結果を見たい

- [DEMO_APP.md](./DEMO_APP.md) を参照

### 4. InspecSafe-V1 を使いたい

- [INSPESAFE_INTEGRATION.md](./INSPESAFE_INTEGRATION.md) を参照

### 5. 新しいノードや出力を追加したい

- [ARCHITECTURE.md](./ARCHITECTURE.md) で流れを確認
- [EXTENDING.md](./EXTENDING.md) のチェックリストを使う

## セットアップ完了チェック

- [ ] `external/Depth-Anything-3`、`external/sam3`、`external/vllm-omni` を配置した
- [ ] 必要な patch を適用した
- [ ] `uv sync --extra dev` が通った
- [ ] `ffmpeg -version` と `ffprobe -version` が通る
- [ ] `uv run pytest tests/ -v` が通る
- [ ] `configs/default.yaml` を自分の環境に合わせた
- [ ] `uv run python src/run.py` で `data/perception_results/` が生成される

## 補足

- 実行のたびに既存の `data/perception_results/` は `data/results_archive/<timestamp>/` に退避されます
- Demo UI は `data/perception_results/manifest.json` を監視して新規フレームだけ読み込みます
- `assessment.safety_status` は TTS 有効時に `data/voice/` の WAV に変換されます
