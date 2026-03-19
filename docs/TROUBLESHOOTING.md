# トラブルシューティングガイド

よくある問題と解決策を集めました。

## セットアップ関連

### 「ModuleNotFoundError: No module named 'safety_agent'」

**原因:** 仮想環境が正しく初期化されていない

**解決策:**
```bash
# 仮想環境を強制再構築
uv sync --force

# または
rm -rf .venv uv.lock
uv sync --extra dev
```

### 「command not found: uv」

**原因:** uv がインストールされていない

**解決策:**
```bash
# uv をインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# PATH に追加（必要に応じて）
export PATH="$HOME/.cargo/bin:$PATH"

# インストール確認
uv --version
```

### 「python: command not found」（なぜか python3 コマンドもない）

**原因:** Python がシステムにインストールされていない

**解決策:**
```bash
# Python 3.11 以上をインストール
# macOS:
brew install python@3.12

# Ubuntu/Debian:
sudo apt update
sudo apt install python3.12 python3.12-venv

# Windows:
# https://www.python.org/downloads/ からインストール
```

---

## 実行関連

### 「OPENAI_API_KEY is not set」

**原因:** 環境変数が設定されていない

**解決策:**

**方法 1: .env ファイルを使用（推奨）**
```bash
# .env が存在するか確認
ls -la .env

# 存在しない場合、作成
cp .env.example .env

# .env を編集
export OPENAI_API_KEY="sk-proj-YOUR-KEY-HERE"
```

**方法 2: 直接環境変数を設定**
```bash
export OPENAI_API_KEY="sk-proj-YOUR-KEY-HERE"
python src/run.py
```

**方法 3: .env ファイルを確認する**
```bash
# .env ファイルが存在するか確認
ls -la .env

# .env に OPENAI_API_KEY が設定されているか確認
grep OPENAI_API_KEY .env
```

### 「OPENAI_API_KEY が含まれた形式が間違っている」

**原因:** API キーの形式が不正

**解決:**
```bash
# API キーを再確認
# https://platform.openai.com/account/api-keys

# .env に正しく設定
OPENAI_API_KEY="sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# クォートが必要な場合
export OPENAI_API_KEY='sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
```

### 実行時に「httpx.HTTPStatusError: Client error '401 Unauthorized'」

**原因:** OpenAI API キーが無効

**解決策:**
```bash
# API キーの有効性を確認
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# 応答がない場合、API キーが無効
# → https://platform.openai.com/account/api-keys で新しいキーを生成
```

---

## Vision API 関連

### 「Vision analysis returned empty response. Please check model availability.」

**原因:** Vision API が空の応答を返している

**解決策:**

**1. モデルが有効か確認**
```bash
# モデル一覧を取得
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | grep gpt-5-nano
```

**2. 画像ファイルが存在するか確認**
```bash
# data/frames/ に画像があるか確認
ls -la data/frames/

# 画像がない場合はテスト画像を配置
cp /path/to/test/image.jpg data/frames/frame_0.0s.jpg
```

**3. Vision API の権限確認**
- OpenAI Dashboard: https://platform.openai.com/account/api-keys
- `configs/default.yaml` で指定した OpenAI モデルで Vision 呼び出し可能か確認
- API キーに実行権限があるか確認

**4. モデル可用性確認**
```bash
# 代わりに configs/default.yaml の vlm.openai.model を変更して再実行
python src/run.py
```

### 「Vision API error 400: Unsupported parameter」

**原因:** モデルがパラメータをサポートしていない（Phase 10.6 で修正済み）

**解決策:**
```bash
# 設定を確認
cat configs/default.yaml | grep -A 5 "llm:"

# agent.py の vlm_node, depth_node で max_tokens が自動設定されているか確認
# run.py で context["config"] が agent に渡されているか確認
```

**詳細:**
- `max_tokens` は config から自動取得（`vision_max_completion_tokens`）
- vlm_node, depth_node で Vision API 呼び出し時に指定
- デフォルト値: 4096 トークン（2 枚画像処理に対応）

**デバッグ方法:**
```bash
# VLM レスポンスの finish_reason を確認（ログに出力）
# finish_reason が "stop" なら正常、"length" なら max_tokens 不足
python src/run.py 2>&1 | grep -A 2 "finish_reason"
```

---

## パフォーマンス関連

### 「実行が遅い（60秒以上）」

**原因:** `max_steps` が大きすぎる

**解決策:**
```bash
# configs/default.yaml で確認
cat configs/default.yaml | grep max_steps

# 修正（3-5 推奨）
# agent:
#   max_steps: 3
```

### 「Vision API 呼び出しがタイムアウトする」

**原因:** ネットワーク遅延 or API が遅い

**解決策:**
```bash
# タイムアウトを延長（configs/default.yaml）
# openai:
#   timeout_s: 120.0
```

---

## テスト関連

### 「pytest: command not found」

**原因:** dev グループが インストールされていない

**解決策:**
```bash
# dev グループを含めてインストール
uv sync --extra dev

# または
pip install pytest
```

### 「test_e2e.py::test_e2e_agent_no_llm FAILED」

**原因:** LLM がなくても動作する設定になっていない

**解決策:**
```bash
# LLM なしで実行するようテストが設定されているか確認
# tests/test_e2e.py を開いて llm=None になっているか確認

# 手動実行
export OPENAI_API_KEY=""  # 空に設定
pytest tests/test_e2e.py -v
```

---

## ファイル I/O 関連

### 「FileNotFoundError: フレームが見つかりません」

**原因:** `data/videos/` に動画がなく、`data/frames/` にも画像がない

**解決策:**
```bash
# フレーム格納先を作成
mkdir -p data/frames

# テスト画像を配置
cp /path/to/test/image.jpg data/frames/frame_0.0s.jpg
```

### 「PermissionError: [Errno 13] Permission denied: 'data/...'」

**原因:** `data/` 配下に書き込み権限がない

**解決策:**
```bash
# 権限を修正
chmod -R 755 data/

# または再作成
rm -rf data/frames data/depth data/voice
mkdir -p data/frames data/depth data/voice
```

---

## 設定ファイル関連

### 「yaml.scanner.ScannerError: mapping values are not allowed」

**原因:** YAML 構文エラー

**例:**
```yaml
# 間違い
model: "gpt-5-nano"  bad space

# 正しい
model: "gpt-5-nano"
```

**解決策:**
```bash
# YAML を検証
python -c "import yaml; yaml.safe_load(open('configs/default.yaml'))"

# エラーが出たら修正
```

---

## デバッグテクニック

### ログを詳しく表示

```bash
# Python スタックトレースを表示
python src/run.py 2>&1 | tail -100

# または stderr をキャプチャ
python src/run.py 2>&1 | tee debug.log
```

### LLM リクエスト/レスポンスをログに出力

**方法 1: agent.py の determine_next_action_llm で確認**
```python
# agent.py の determine_next_action_llm() 関数内で以下を追加
print(f"DEBUG: LLM request - system prompt length: {len(system_prompt)}")
print(f"DEBUG: LLM request - max_tokens: {max_tokens}")

# レスポンス取得後
try:
    response = llm.chat_json(...)
    print(f"DEBUG: LLM response: {response}")
except Exception as e:
    print(f"DEBUG: LLM error: {e}")
```

**方法 2: ログレベルを上げる**
```bash
# PYTHONUNBUFFERED で標準出力を即座にフラッシュ
PYTHONUNBUFFERED=1 python src/run.py 2>&1 | tee debug.log

# ログを確認
grep "DEBUG" debug.log
```

### Vision API レスポンスを確認

**方法 1: modality_nodes.py の VisionAnalyzer で確認**
```python
# modality_nodes.py の analyze() メソッド内で以下を追加
print(f"DEBUG: Vision API request - image paths: {image_paths}")
print(f"DEBUG: Vision API request - max_tokens: {max_tokens}")
print(f"DEBUG: Vision API response status: {r.status_code}")

if r.status_code >= 400:
    print(f"DEBUG: Vision API error: {r.text[:500]}")
else:
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    print(f"DEBUG: Vision API response (first 300 chars): {content[:300]}")
```

**方法 2: JSON パースエラーをキャッチ**
```bash
# JSON パースエラーが発生した場合、modality_nodes.py でログを確認
# 418行: logger.warning で最初の 300 文字を表示
python src/run.py 2>&1 | grep -A 1 "Failed to parse JSON"
```

---

## それでも解決しない場合

1. **チームリーダーに報告**
   - エラーメッセージのスクリーンショット
   - `debug.log` ファイル
   - OS/Python バージョン情報

2. **GitHub Issues で検索**
   - 類似の問題がないか確認

3. **詳細ドキュメント**
   - [SETUP.md](SETUP.md) - セットアップガイド
   - [ARCHITECTURE.md](ARCHITECTURE.md) - システム設計
   - [CLAUDE.md](../CLAUDE.md) - Claude Code 向け情報

