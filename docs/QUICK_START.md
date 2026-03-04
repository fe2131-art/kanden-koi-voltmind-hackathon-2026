# クイックスタートガイド

**5分でプロジェクトを動かすガイド**

## TL;DR - コマンドのみ

```bash
# 1. セットアップ（初回のみ）
uv sync --extra dev
cp .env.example .env
# .env を編集して OPENAI_API_KEY を設定

# 2. テスト（LLM不要、1-2分）
pytest tests/ -v

# 3. 実行（Vision API 使用、30-40秒）
set -a && source .env && set +a && python src/run.py

# 4. 結果確認
cat data/perception_results.json
cat data/agent_execution_summary.txt
```

## 詳細版（初心者向け）

### 1. リポジトリをクローン

```bash
git clone <repository-url>
cd kanden-koi-voltmind-hackathon-2026
```

### 2. 環境をセットアップ

```bash
# uv で仮想環境を構築
uv sync --extra dev
```

**何が起こる？**
- Python 仮想環境が `.venv` に作成される
- 依存パッケージ（LangGraph, OpenAI, Pydantic等）がインストールされる

### 3. API キーを設定

```bash
# テンプレートをコピー
cp .env.example .env

# エディタで開いて OPENAI_API_KEY を設定
# vi .env または nano .env
```

**例:**
```env
OPENAI_API_KEY="sk-proj-your-api-key-here"
```

### 4. テストして動作確認

```bash
# LLM 不要のテストを実行
pytest tests/ -v
```

**期待される出力:**
```
test_schema.py::test_observation_creation PASSED
test_e2e.py::test_e2e_agent_no_llm PASSED
```

### 5. 本体を実行

```bash
# 環境変数を読み込んで実行
set -a && source .env && set +a && python src/run.py
```

**実行中のログ:**
```
✅ Found video: free-video7-rice-cafinet.mp4
✅ Extracted 30 frames from free-video7-rice-cafinet.mp4 at 1.0 FPS
✅ Extracted audio to audio.wav
✅ Found 1 image(s) in data/images/
🔍 Processing: test_image.jpg
   - Objects detected: 2
   - Hazards identified: 1
   - Unobserved regions: 3
=== Running Safety View Agent ===
✅ Agent execution summary saved to data/agent_execution_summary.txt
✅ Graph diagram saved to data/flow.md
```

### 6. 結果を確認

```bash
# 構造化データ（video_timestamp 含む）
cat data/perception_results.json

# 人間向けレポート
cat data/agent_execution_summary.txt

# グラフ図（Markdown）
cat data/flow.md

# 抽出されたフレーム
ls -lh data/frames/

# 抽出された音声
ls -lh data/audio/
```

## 主要なコマンド

| コマンド | 説明 |
|---------|------|
| `uv sync --extra dev` | 仮想環境をセットアップ |
| `pytest tests/ -v` | テストを実行 |
| `python src/run.py` | エージェントを実行 |
| `python finetuning/train_dummy.py` | ダミー学習を実行 |

## ファイルの役割

```
src/
├── run.py                    # ← エージェント実行スクリプト + 動画処理関数
├── apps/                     # ← React + Vite デモアプリ
├── safety_agent/
│   ├── agent.py             # LLM・グラフノード
│   ├── perceiver.py         # Vision処理
│   └── schema.py            # データモデル（video_timestamp 含む）
│
configs/
└── default.yaml             # ← モデル設定（gpt-5-nano 固定、video セクション追加）

data/                         # ← 統合データフォルダ
├── videos/                  # 入力動画
├── frames/                  # 抽出フレーム（自動生成）
├── audio/                   # 抽出音声（自動生成）
├── images/                  # 静止画入力
└── perception_results.json  # 実行結果（video_timestamp 含む）
```

## トラブルシューティング

### 「ModuleNotFoundError: No module named 'safety_agent'」

```bash
# 仮想環境を再構築
uv sync --force
```

### 「OPENAI_API_KEY is not set」

```bash
# 環境変数を確認
echo $OPENAI_API_KEY

# .env が正しく読み込まれているか確認
set -a && source .env && set +a && echo $OPENAI_API_KEY
```

### テストが失敗する

```bash
# 詳細ログを表示
pytest tests/ -vv -s

# 特定テストのみ実行
pytest tests/test_schema.py -v
```

## 次のステップ

- [詳細セットアップガイド](SETUP.md) - より詳しい説明
- [アーキテクチャ](ARCHITECTURE.md) - システム設計
- [トラブルシューティング](TROUBLESHOOTING.md) - よくある問題
- [CLAUDE.md](../CLAUDE.md) - Claude Code 向け情報

## よくある質問（FAQ）

**Q: Vision API は何をしているの？**
A: 入力画像を分析して、安全上の危険物や注意が必要な領域を自動検出しています。

**Q: なぜ `gpt-5-nano-2025-08-07` なの？**
A: Vision 対応モデルの中でコスト効率が最適だからです。変更しないでください。

**Q: オフラインで実行できる？**
A: はい、LLM を使わずヒューリスティックフォールバックで動作可能です。

**Q: 実行時間はどのくらい？**
A: 30-40秒（Vision API + エージェントループ）

**Q: 複数の画像・動画を処理できる？**
A: はい、`data/images/` に複数画像を置くと順番に処理します。動画は自動的にフレーム分割されます。

**Q: 動画からのフレーム分割はどのくらい時間がかかる？**
A: 30秒の動画をデフォルト設定（1.0 FPS、30フレーム上限）で処理するのに約 5-10 秒です。

**Q: ffmpeg がない場合はどうする？**
A: 動画処理機能は使えませんが、画像のみなら問題なく動作します。`docs/UV_VENV.md` を参照してください。
