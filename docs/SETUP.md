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
  max_steps: 1

llm:
  provider: "openai"
  openai:
    model: "gpt-5-nano-2025-08-07"  # ← 絶対に変更しない
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
# data/images/ フォルダに画像を配置
cp /path/to/your/image.jpg data/images/
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
- `perception_results.json` - 構造化データ（Vision 分析結果 + video_timestamp 含む）
- `agent_execution_summary.txt` - 人間向けレポート
- `flow.md` - LangGraph 実行フロー図
- `frames/` - 抽出されたフレーム画像
- `audio/audio.wav` - 抽出された音声

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
# monitoring: /path/to/output/perception_results.json
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

## ヘルプが必要な場合

- Issues: プロジェクトの GitHub Issues を確認
- Team Lead: チームリーダーに連絡
