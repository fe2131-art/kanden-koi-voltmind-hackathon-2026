#!/bin/bash
# 外部依存（Depth-Anything-3）のセットアップスクリプト
# 用途: CI/CD や新規開発者が修正を自動適用する

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DA3_DIR="${REPO_ROOT}/external/Depth-Anything-3"
PATCHES_DIR="${SCRIPT_DIR}/patches"

echo "[INFO] Depth-Anything-3 セットアップ開始..."

# 1. リポジトリ存在確認
if [ ! -d "$DA3_DIR" ]; then
    echo "[ERROR] $DA3_DIR が見つかりません"
    echo "[INFO] 以下のコマンドで clone してください:"
    echo "  git clone https://github.com/DepthAnything/Depth-Anything-3.git $DA3_DIR"
    exit 1
fi

# 2. パッチを適用
if [ -f "$PATCHES_DIR/da3-numpy-compatibility.patch" ]; then
    echo "[INFO] Patch 適用: da3-numpy-compatibility.patch"
    cd "$DA3_DIR"
    git apply "$PATCHES_DIR/da3-numpy-compatibility.patch" || {
        echo "[WARN] Patch 適用に失敗（既に適用済みの可能性）"
    }
    cd "$REPO_ROOT"
else
    echo "[WARN] Patch ファイルが見つかりません: $PATCHES_DIR/da3-numpy-compatibility.patch"
fi

echo "[INFO] セットアップ完了 ✓"
echo ""
echo "[使用方法]"
echo "  - 開発時: uv run python scripts/depth_anything_3/smoke_test_da3.py --image path/to/image.jpg"
echo "  - 詳細: uv run python scripts/depth_anything_3/smoke_test_da3.py --help"
