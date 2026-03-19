# Safety View Agent - セットアップガイド

チームメンバがプロジェクトを正しくセットアップするための完全ガイドです。

## 前提条件

- **Python 3.12 以上** がシステムにインストール済み
- **uv** パッケージマネージャー（最新版推奨）
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **OpenAI API キー** （Vision API を使用する場合）

## ステップ 1: リポジトリをクローン

```bash
git clone <repository-url>
cd kanden-koi-voltmind-hackathon-2026
```

## ステップ 2: 仮想環境をセットアップ

```bash
# uv で仮想環境を構築・有効化
uv sync --extra dev
```

このコマンドで以下が自動実行されます：
- Python 仮想環境の作成
- `pyproject.toml` の依存関係をインストール
- dev グループの追加依存関係をインストール（テスト、リント等）

## ステップ 3: 環境変数を設定

### Option A: .env ファイルを使用（推奨）

```bash
# .env.example をコピー
cp .env.example .env

# .env を編集（テキストエディタで開く）
# OPENAI_API_KEY="sk-proj-YOUR-KEY-HERE" を設定
```

**例：**
```bash
OPENAI_API_KEY="sk-proj-your-api-key-here"
```

### Option B: 環境変数を直接設定

```bash
export OPENAI_API_KEY="sk-proj-YOUR-KEY-HERE"
```

## ステップ 4: 設定を確認

```bash
# configs/default.yaml を確認
cat configs/default.yaml
```

**重要:** 以下が確定していることを確認：
```yaml
agent:
  max_steps: -1

llm:
  provider: "openai"
  openai:
    model: "gpt-5-nano"
```

## ステップ 5: テストで動作確認

```bash
# LLM 不要のテストを実行（1-2分）
pytest tests/ -v

# 成功時の出力例：
# tests/test_schema.py::test_observation_creation PASSED
# tests/test_e2e.py::test_e2e_agent_no_llm PASSED
```

## ステップ 6: 入力データを準備（オプション）

### 静止画を使う場合：
```bash
# data/frames/ フォルダに画像を配置
cp /path/to/your/image.jpg data/frames/frame_0.0s.jpg
```

### 動画を使う場合（推奨）：
```bash
# data/videos/ フォルダに動画ファイルを配置
cp /path/to/your/video.mp4 data/videos/

# エージェントが自動的に以下を実行します：
# - フレーム分割 (data/frames/ に保存)
# - 音声抽出 (data/audio/ に保存)
```

## ステップ 7: エージェントを実行

```bash
# .env ファイルから自動的に環境変数を読み込んで実行
python src/run.py
```

**注**: `.env` ファイルは自動的に読み込まれます（`python-dotenv` ライブラリにより）

**実行後、以下のファイルが `data/` に生成されます：**
- `perception_results.json` - 構造化データ（フレーム単位の分析結果、frames 配列形式）
- `flow.md` - LangGraph 実行フロー図（Mermaid 形式）
- `frames/` - 抽出されたフレーム画像
- `audio/audio.wav` - 抽出された音声ファイル
- `depth/` - 深度解析の可視化画像
- `infrared_frames/` - 赤外線動画から展開したフレーム（inspesafe モード時）

## 一般的な問題

### ImportError: No module named 'safety_agent'

```bash
# 仮想環境を再構築
uv sync --force
```

### OPENAI_API_KEY is not set

```bash
# .env ファイルが存在するか確認
ls -la .env

# .env の内容を確認
grep OPENAI_API_KEY .env

# .env を編集してAPIキーを設定
nano .env
```

### Vision API が空の応答を返す

- モデルが利用可能か確認：https://platform.openai.com/account/api-keys
- API キーの権限確認（Vision API にアクセス可能か）

## 次のステップ

- [クイックスタートガイド](QUICK_START.md) - 5分でプロジェクトを理解
- [アーキテクチャ](ARCHITECTURE.md) - システム設計の詳細
- [トラブルシューティング](TROUBLESHOOTING.md) - よくある問題と解決策
- [CLAUDE.md](../CLAUDE.md) - Claude Code 向け詳細情報

## オプション: React デモアプリのセットアップ

ブラウザで動画とリアルタイム検出結果を確認したい場合：

### 前提条件
- Node.js 18+ と npm

### セットアップ手順

#### 1. デモ依存をインストール
```bash
uv sync --extra demo
```

#### 2. WebSocket サーバーを起動（ターミナル 1）
```bash
python src/apps/server.py
# 出力例：
# ws server: ws://localhost:8001
# monitoring: /path/to/repo/data/perception_results.json
```

#### 3. React アプリをセットアップ・起動（ターミナル 2）
```bash
cd src/apps
npm install
npm run dev
# 出力例：
#   VITE v5.0.8  ready in 123 ms
#   ➜  Local:   http://localhost:5173/
```

#### 4. ブラウザで開く
```
http://localhost:5173
```

#### 5. エージェントを実行（ターミナル 3）
```bash
# .env から自動的に環境変数を読み込んで実行
python src/run.py
```

### 動作確認チェックリスト

- [ ] React アプリが `http://localhost:5173` で起動している
- [ ] WebSocket が接続状態（Status パネルで `ws: 1 (OPEN)` が表示）
- [ ] 動画 `/video.mp4` が表示され再生可能
- [ ] エージェント実行後、検出結果が Canvas に BBox として描画される

詳細は [DEMO_APP.md](DEMO_APP.md) を参照。

## オプション: Depth-Anything-3 セットアップ

深度推定機能を使用する場合は、Depth-Anything-3 をセットアップしてください。

### セットアップ手順

#### 1. Depth-Anything-3 をクローン

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git external/Depth-Anything-3
```

#### 2. パッチを自動適用

```bash
# scripts/depth_anything_3/ 内のセットアップスクリプトを実行
bash scripts/depth_anything_3/setup_external_deps.sh
```

このスクリプトは以下を自動実行します：
- numpy バージョン制約を緩和（`numpy<2` → `numpy`）
- pyproject.toml と requirements.txt を修正

#### 3. 動作確認

```bash
# デモ画像を使用してテスト実行
uv run python scripts/depth_anything_3/smoke_test_da3.py \
  --image scripts/depth_anything_3/depth_anything_3_demo.png \
  --model metric \
  --focal-px 1000
```

### 使用方法

```bash
# Metric Depth（実メートル深度）
uv run python scripts/depth_anything_3/smoke_test_da3.py \
  --image path/to/your/image.jpg \
  --model metric \
  --focal-px 1000

# Monocular Depth（相対深度）
uv run python scripts/depth_anything_3/smoke_test_da3.py \
  --image path/to/your/image.jpg \
  --model mono
```

**詳細なオプション:**
```bash
uv run python scripts/depth_anything_3/smoke_test_da3.py --help
```

### 修正内容

Depth-Anything-3 では以下の小規模な修正を適用しています：

| ファイル | 修正内容 | 理由 |
|--------|--------|------|
| `pyproject.toml` | `numpy<2` → `numpy` | numpy 2.0+ との互換性 |
| `requirements.txt` | `numpy<2` → `numpy` | numpy 2.0+ との互換性 |

修正内容は `scripts/depth_anything_3/patches/da3-numpy-compatibility.patch` で管理されています。

### トラブルシューティング

**Patch 適用エラー**
```bash
# 既に適用済みの可能性がある場合、初期状態に戻す
cd external/Depth-Anything-3
git reset --hard HEAD
cd ../..
bash scripts/depth_anything_3/setup_external_deps.sh
```

**モデルダウンロードエラー**
```bash
# インターネット接続を確認し、Hugging Face キャッシュをクリア
rm -rf ~/.cache/huggingface/hub

# 再度実行
uv run python scripts/depth_anything_3/smoke_test_da3.py \
  --image scripts/depth_anything_3/depth_anything_3_demo.png \
  --model metric
```

## オプション: vLLM-Omni セットアップ

マルチモーダル推論機能（Vision + Audio 統合）を使用する場合は、vLLM-Omni をセットアップしてください。

### セットアップ手順

#### 1. vLLM-Omni をクローン

```bash
git clone https://github.com/vllm-project/vllm-omni.git external/vllm-omni
```

#### 2. 依存関係をインストール

既に `pyproject.toml` に vLLM-Omni が記載されているため、以下を実行：

```bash
uv sync --extra dev
```

このコマンドで自動的に vLLM-Omni がビルド・インストールされます。

#### 3. 動作確認

```bash
# vLLM-Omni がインストールされたか確認
python -c "import vllm_omni; print('vLLM-Omni installed successfully')"
```

### 使用方法

vLLM-Omni は以下の機能を提供します：

- **マルチモーダル推論**: Vision + Audio 入力の統合処理
- **効率的な推論**: vLLM の最適化技術を活用
- **拡張可能な設計**: カスタムモデルの統合

詳細は [vLLM-Omni 公式ドキュメント](https://github.com/vllm-project/vllm-omni) を参照。

## ヘルプが必要な場合

- Issues: プロジェクトの GitHub Issues を確認
- Team Lead: チームリーダーに連絡

---

**最終更新:** 2026-03-19
**対象バージョン:** Safety View Agent v1.0
