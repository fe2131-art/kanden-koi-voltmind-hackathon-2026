# Safety View Agent - Claude Code 向けプロジェクト情報

## プロジェクト概要
**Safety View Agent** は、LangGraph を使用した安全支援エージェントです。複雑なマルチスタジアムロボット環境での安全性を評価し、次に観測すべき視点（画角）を動的に提案します。

### 特徴
- **LLM 不要**: OpenAI互換LLMが利用できない場合、固定値 SafetyAssessment で継続動作
- **フラット構造**: 複雑なプラグインアーキテクチャを廃止し、シンプルな単一パッケージ設計
- **E2E 動作検証**: `pytest` で LLM なしでもエンドツーエンド通過

### 前提条件
- **実行環境**: `uv` で仮想環境が有効 (`uv sync` 実行後)
- **実行コマンド**: すべて `python src/run.py` で統一
- **モデル**: `configs/default.yaml` で切り替え（既定は `gpt-5-nano` / `Qwen/Qwen3.5-9B`）

## ディレクトリ構造

```
kanden-koi-voltmind-hackathon-2026/
├── src/
│   ├── run.py                 # 実行エントリポイント（main()関数）
│   ├── safety_agent/          # メインパッケージ
│   │   ├── __init__.py
│   │   ├── schema.py          # Pydantic モデル定義
│   │   ├── modality_nodes.py  # モダリティ処理クラス（VisionAnalyzer, AudioAnalyzer）
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
│   ├── perception_results.json # 分析結果（追記式）
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
├── external/                   # 外部依存パッケージ（editable install）
│   ├── Depth-Anything-3/       # 深度推定モデル
│   └── vllm-omni/              # マルチモーダル推論フレームワーク
│
├── scripts/
│   └── depth_anything_3/       # Depth-Anything-3 セットアップ・テスト
│       ├── setup_external_deps.sh
│       ├── smoke_test_da3.py
│       └── patches/
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

**外部依存パッケージのセットアップ（初回のみ）:**

```bash
# 1. Depth-Anything-3（深度推定機能）
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git external/Depth-Anything-3

# 2. vLLM-Omni（マルチモーダル推論）
git clone https://github.com/vllm-project/vllm-omni.git external/vllm-omni

# 3. 再度 uv sync で両方をビルド・インストール
uv sync --extra dev
```

詳細は [SETUP.md](docs/SETUP.md) を参照。

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
# 1. APIキーを環境変数に設定
export OPENAI_API_KEY="sk-your-api-key-here"

# 2. configs/default.yaml を設定
# llm.provider: "openai"
# llm.openai.model: "gpt-4o"  （または任意の Vision 対応モデル）

# 3. エージェント実行
python src/run.py
```

#### 3. vLLM (ローカルサーバー) で実行
```bash
# 1. vLLM サーバーを起動（別ターミナル）
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --port 8000

# 2. configs/default.yaml を設定
# llm.provider: "vllm"
# llm.vllm.base_url: "http://localhost:8000/v1"
# llm.vllm.model: "meta-llama/Llama-2-7b-hf"

# 3. エージェント実行
python src/run.py
```

**注**: `.env` ファイルを作成することで、環境変数（OPENAI_API_KEY など）を一度に読み込むこともできます：
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
- **Pydantic モデル**: `PerceptionIR`, `SafetyAssessment`, `AudioCue`, `VisionAnalysisResult` など
- **Observation / ObservationProvider**: データソース抽象化

### モダリティ処理ノード（`src/safety_agent/modality_nodes.py`）
- **`VisionAnalyzer`**: OpenAI互換 Vision API による画像テキスト分析
- **`AudioAnalyzer`**: 音声テキストから AudioCue を抽出（危険キュー検出）
- **`DepthEstimator`**: 深度推定モデルによる 3D シーン分析
- **`ModalityResult`**: 各モダリティの統一結果型（modality_name, audio_cues, description, extra, error フィールド）

### エージェント実装（`src/safety_agent/agent.py`）
- **`OpenAICompatLLM`**: OpenAI互換 LLM クライアント（httpx 使用）
- **グラフノード関数（fan-out/fan-in パイプライン）**:
  - `ingest_observation`: 観測データ取得 + fan-out 並列送信（Command + Send）
  - `vlm_node`: 画像分析（並列実行）
  - `audio_node`: 音声キュー抽出（並列実行）
  - `depth_node`: 深度推定・分析（並列実行）
  - `join_modalities`: fan-in バリア（ラッチ機構）
  - `fuse_modalities`: モダリティ結果の統合
  - `update_belief_state_llm`: BeliefState 更新
  - `determine_next_action_llm`: 知覚推論 + 総合安全判断を統合実行（**LLM がない場合は固定値 SafetyAssessment を返す**）
    * ステップ1: VLM/音声からハザード推定 → ir に格納
    * ステップ2: 推定ハザード + 世界モデル + 前回判断から SafetyAssessment 生成
    * フォールバック: LLM 未設定/失敗時は `risk_level="low"`, `action_type="monitor"` の固定値を返す
  - `emit_output`: フレーム出力（`frame_id`, `assessment`, `vision_analysis`, `audio`, `depth_analysis` を含む）
  - `bump_step`: ステップカウント
- **`build_agent()`**: LangGraph グラフ構築（fan-out/fan-in 実装、状態拡張）

## LLM フォールバック設計

### 動作フロー

```
1. run.py#get_llm(): YAML + 環境変数から LLM を初期化
   - provider = "openai": OPENAI_API_KEY が必須
   - provider = "vllm": `configs/default.yaml` の `llm.vllm.base_url` が必須
   - どちらも設定なし → llm = None

2. agent.py#determine_next_action_llm():
   - runtime.context["llm"] が None → 固定値 SafetyAssessment を返す
   - LLM 実行エラー → 例外をキャッチして固定値 SafetyAssessment を返す
   - ir=None (モダリティ処理失敗) → 固定値 SafetyAssessment で継続動作

3. 固定値フォールバック:
   - risk_level="low", action_type="monitor", priority=0.0
   - safety_status="継続観測中"
   - temporal_status="unknown", evidence=None
   → 安全側を取り、継続的に観測可能にする設計
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

### データ出力構造（`data/perception_results.json`）

フレーム単位の分析結果を `frames` 配列で保存：

```json
{
  "frames": [
    {
      "frame_id": "t0",
      "video_timestamp": 0.0,
      "assessment": {
        "risk_level": "low",
        "action_type": "monitor",
        "reason": "..."
      },
      "vision_analysis": {
        "scene_description": "...",
        "critical_points": [...]
      },
      "audio": [...],
      "depth_analysis": {...},
      "errors": []
    }
  ]
}
```

**構造の特徴**:
- `frame_id`: フレーム一意ID
- `assessment`: LLM / 固定値による安全判断
- `vision_analysis`: VLM 分析結果（scene_description, critical_points, blind_spots）
- `audio`: 音声キュー（AudioCue リスト）
- `depth_analysis`: 深度推定結果（scene_description, depth_layers）
- `errors`: モダリティ処理エラー

## `data/` フォルダ用途

- `data/videos/`: 入力動画
- `data/frames/`: 抽出フレーム、または手動配置した静止画フレーム
- `data/audio/`: 抽出・コピーした音声
- `data/depth/`: 深度可視化画像
- `data/infrared_frames/`: 赤外線フレーム
- `data/perception_results.json`: エージェント実行結果
- `data/flow.md`: グラフ図（`Mermaid`）

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
