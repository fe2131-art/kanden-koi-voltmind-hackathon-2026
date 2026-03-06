# Safety View Agent - Claude Code 向けプロジェクト情報

## プロジェクト概要
**Safety View Agent** は、LangGraph を使用した安全支援エージェントです。複雑なマルチスタジアムロボット環境での安全性を評価し、次に観測すべき視点（画角）を動的に提案します。

### 特徴
- **LLM 不要**: OpenAI互換LLMが利用できない場合、ヒューリスティックスで動作
- **フラット構造**: 複雑なプラグインアーキテクチャを廃止し、シンプルな単一パッケージ設計
- **E2E 動作検証**: `pytest` で LLM なしでもエンドツーエンド通過

### 前提条件
- **実行環境**: `uv` で仮想環境が有効 (`uv sync` 実行後)
- **実行コマンド**: すべて `python src/run.py` で統一
- **モデル**: `gpt-5-nano-2025-08-07` 固定（変更禁止）

## ディレクトリ構造

```
kanden-koi-voltmind-hackathon-2026/
├── src/
│   ├── run.py                 # 実行エントリポイント（main()関数）
│   ├── safety_agent/          # メインパッケージ
│   │   ├── __init__.py
│   │   ├── schema.py          # Pydantic モデル定義
│   │   ├── modality_nodes.py  # モダリティ処理クラス（VisionAnalyzer, YOLODetector, AudioAnalyzer）
│   │   ├── perceiver.py       # Perceiver クラス（ハザード推定）
│   │   └── agent.py           # LLM・グラフノード・ビルダー（fan-out/fan-in 実装）
│   └── apps/                  # React + Vite デモアプリ
│       ├── server.py             # WebSocket サーバー
│       ├── App.jsx               # React メインコンポーネント
│       ├── vite.config.js        # Vite 設定
│       ├── package.json          # npm 依存
│       └── index.html            # エントリーポイント
│
├── tests/
│   ├── test_schema.py         # スキーマ検証テスト
│   └── test_e2e.py            # E2Eスモークテスト（LLM不要）
│
├── data/                       # 統合データフォルダ
│   ├── videos/                # 入力動画
│   ├── frames/                # 抽出フレーム
│   ├── audio/                 # 抽出音声
│   ├── images/                # 静止画入力
│   ├── perception_results.json # 分析結果（追記式）
│   ├── agent_execution_summary.txt
│   └── flow.md                # グラフ図
│
├── finetuning/
│   ├── data/samples/
│   │   └── dummy_instructions.jsonl
│   └── train_dummy.py          # CPU ダミー学習スクリプト
│
├── configs/
│   └── default.yaml
│
├── pyproject.toml              # uv / hatchling 設定
├── .gitignore
└── CLAUDE.md                   # このファイル
```

## 基本コマンド

### セットアップ
```bash
uv sync --extra dev
```

### 環境をアクティベート
```
source .venv/bin/activate
```

### テスト実行（LLM 不要）
```bash
pytest tests/ -v
```

### エージェント実行

#### 1. LLM なしで実行（ヒューリスティックフォールバック）
```bash
python src/run.py
```

#### 2. OpenAI API で実行
```bash
# 1. OpenAI APIキーを環境変数に設定
export OPENAI_API_KEY="sk-your-api-key-here"
export OPENAI_MODEL="gpt-4o"

# 2. configs/default.yaml の llm.provider を "openai" に設定
# 3. エージェント実行
python src/run.py
```

#### 3. vLLM (ローカルサーバー) で実行
```bash
# 1. vLLM サーバーを起動（別ターミナル）
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --port 8000

# 2. LLM 環境変数を設定
export LLM_BASE_URL="http://localhost:8000"
export LLM_MODEL="meta-llama/Llama-2-7b-hf"

# 3. configs/default.yaml の llm.provider を "vllm" に設定
# 4. エージェント実行
python src/run.py
```

**注**: `.env` ファイルを作成することで、環境変数を一度に読み込むこともできます：
```bash
# .env を作成（.env.example を参考）
cp .env.example .env
# .env を編集してAPIキーを入力

# 実行（.env から自動的に環境変数を読み込み）
python src/run.py
```

### ダミー学習実行
```bash
python finetuning/train_dummy.py --epochs 3
```

## 重要なコードパス

### スキーマ定義（`src/safety_agent/schema.py`）
- **Pydantic モデル**: `BoundingBox`, `Hazard`, `PerceptionIR`, `SafetyAssessment` など
- **Observation / ObservationProvider**: データソース抽象化

### モダリティ処理ノード（`src/safety_agent/modality_nodes.py`）
- **`VisionAnalyzer`**: OpenAI互換 Vision API による画像テキスト分析
- **`YOLODetector`**: ultralytics YOLO 物体検出（threading.Lock で並列実行対応）
- **`AudioAnalyzer`**: 音声テキストからヒューリスティックで AudioCue を抽出
- **`ModalityResult`**: 各モダリティの統一結果型（objects, audio_cues, description, error フィールド）

### エージェント実装（`src/safety_agent/agent.py`）
- **`OpenAICompatLLM`**: OpenAI互換 LLM クライアント（httpx 使用）
- **グラフノード関数（fan-out/fan-in パイプライン）**:
  - `ingest_observation`: 観測データ取得 + fan-out 並列送信（Command + Send）
  - `yolo_node`: 物体検出（並列実行）
  - `vlm_node`: 画像分析（並列実行）
  - `audio_node`: 音声キュー抽出（並列実行）
  - `join_modalities`: fan-in バリア（ラッチ機構）
  - `fuse_modalities`: モダリティ結果の統合
  - `update_world_model`: 世界モデル更新
  - `determine_next_action_llm`: 知覚推論 + 総合安全判断を統合実行（**LLM がない場合は `_heuristic_assessment` にフォールバック**）
    * ステップ1: YOLO/VLM/音声からハザード推定 → ir.hazards, ir.unobserved を LLM 出力で上書き
    * ステップ2: 推定ハザード + 世界モデル + 前回判断から SafetyAssessment 生成
  - `emit_output`: フレーム出力
  - `bump_step`: ステップカウント
- **`build_agent()`**: LangGraph グラフ構築（fan-out/fan-in 実装、状態拡張）

### 知覚処理（`src/safety_agent/perceiver.py`）
- **`Perceiver.estimate()`**: オブジェクト・音声キューからハザード推定 + 未確認領域推定
- **`Perceiver.run()`**: 後方互換ラッパー（テスト用）

## LLM フォールバック設計

### 動作フロー

```
1. run.py#get_llm(): YAML + 環境変数から LLM を初期化
   - provider = "openai": OPENAI_API_KEY が必須
   - provider = "vllm": LLM_BASE_URL が必須
   - どちらも設定なし → llm = None

2. agent.py#determine_next_action_llm():
   - runtime.context["llm"] が None → _heuristic_assessment(state) へフォールバック
   - LLM 実行エラー → 例外キャッチして _heuristic_assessment へ降格

3. ヒューリスティック: 世界モデルから SafetyAssessment を生成
   - 高リスク未確認領域あり → focus_region（観測指示）
   - 低信度ハザード存在 → increase_safety（安全強化指示）
   - その他 → continue_observation（継続観測指示）
```

### OpenAI vs vLLM

| 項目 | OpenAI API | vLLM (ローカル) |
|------|-----------|-----------------|
| **設定** | `provider: "openai"` | `provider: "vllm"` |
| **認証** | `OPENAI_API_KEY` | `EMPTY` (または不要) |
| **ベースURL** | `https://api.openai.com/v1` | `http://localhost:8000` |
| **費用** | トークンベースの課金 | 無料（自ホスト） |
| **応答速度** | 遅い (API レイテンシ) | 速い (ローカル) |
| **オフライン対応** | ✗ | ✓ |

**ポイント**: LLM なしでも E2E テストが通るため、初期開発・CI/CD が容易

## input / output フォルダ用途

### `input/`
- 観測画像 (`.jpg`, `.png`)
- 音声ファイル (`.wav`, `.mp4`) の入力先
- .gitkeep により git 管理（ファイルは .gitignore）

### `output/`
- エージェント実行結果 (JSON, CSV)
- グラフ図（`flow.md` の Mermaid テキスト）
- エージェント実行ログ

## コーディング規約

### フォーマッター / リンター
```bash
# ruff でチェック・修正
uv run ruff check --fix .
uv run ruff format .

# pyright で型チェック
uv run pyright src/ tests/
```

### 命名規則
- **クラス**: `PascalCase` (例: `ViewCandidate`, `PerceptionIR`)
- **関数**: `snake_case` (例: `propose_next_view_llm`, `_heuristic_plan`)
- **プライベート**: `_` プレフィックス (例: `_robust_json_loads`)
- **定数**: `UPPER_SNAKE_CASE`

## 依存関係の方向

```
finetuning/train_dummy.py (独立)
         ↓
run.py
         ↓
src/safety_agent/agent.py
         ↓
src/safety_agent/perceiver.py
         ↓
src/safety_agent/schema.py
```

## トラブルシューティング

### テスト失敗
```bash
# 詳細なエラーメッセージを見る
pytest tests/ -vv -s

# 特定のテストのみ実行
uv run pytest tests/test_e2e.py::test_e2e_agent_no_llm -v
```

### OpenAI API エラー
```bash
# APIキーが設定されているか確認
echo $OPENAI_API_KEY

# APIキーが有効か確認
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### vLLM 接続エラー
```bash
# LLM_BASE_URL を確認
echo $LLM_BASE_URL

# ローカルサーバーが起動しているか確認
curl http://localhost:8000/v1/models

# サーバーを起動
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --port 8000
```

### 設定ファイルエラー
```bash
# YAML 構文をチェック
python -c "import yaml; yaml.safe_load(open('configs/default.yaml'))"

# configs/default.yaml の llm.provider を確認
cat configs/default.yaml | grep -A 5 "llm:"
```

### import エラー
```bash
# 環境を再構築
uv sync --force
```

## 参考資料
- LangGraph: https://langchain-ai.github.io/langgraph/
- Pydantic: https://docs.pydantic.dev/
- OpenAI API: https://platform.openai.com/docs/api-reference
