# uv 仮想環境の使用方法

## 方法 1: `uv run` で自動実行（推奨）

```bash
# 仮想環境を自動有効化して実行
uv run python src/run.py
```

**メリット:**
- 簡単（1つのコマンド）
- 自動で依存関係を確認

## 方法 2: 仮想環境に直接入る

### macOS / Linux

```bash
# 仮想環境に入る
source .venv/bin/activate

# 確認（プロンプトが変わる）
(環境名) $ python src/run.py

# 仮想環境を出る
deactivate
```

### Windows

```bash
# 仮想環境に入る
.venv\Scripts\activate

# または PowerShell の場合
.venv\Scripts\Activate.ps1

# 確認（プロンプトが変わる）
(環境名) > python src/run.py

# 仮想環境を出る
deactivate
```

## 仮想環境の状態を確認

```bash
# どの Python が使われているか確認
which python        # または where python (Windows)

# 仮想環境が有効か確認
echo $VIRTUAL_ENV   # または echo %VIRTUAL_ENV% (Windows)

# インストール済みパッケージ一覧
pip list
```

## トラブルシューティング

### 「python: command not found」（仮想環境で）

```bash
# 仮想環境を再構築
uv sync --force

# または
source .venv/bin/activate
```

### 「仮想環境が見つからない」

```bash
# 最初のセットアップから実行
uv sync --extra dev
```

## 推奨：uv run を使う理由

| 方法 | 簡単さ | 管理 | 推奨 |
|------|------|------|------|
| `uv run python ...` | ⭐⭐⭐ | 自動 | ✅ |
| `source .venv/bin/activate` | ⭐⭐ | 手動 | 開発時 |

**結論:** 日常的には `uv run python ...` を使い、長時間開発する場合は仮想環境に入るのが効率的です。
