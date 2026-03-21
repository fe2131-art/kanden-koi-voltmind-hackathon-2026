#!/usr/bin/env bash
# バッチデモ実行スクリプト
# inspesafe × 7 + manual × 7 を configs/default.yaml を書き換えながら順次実行する

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$REPO_ROOT/configs/default.yaml"
CONFIG_BAK="$REPO_ROOT/configs/default.yaml.bak"
RESULTS_DIR="/home/team-005/data/output"
VIDEOS_DIR="$REPO_ROOT/data/videos"
DATASET_PATH="/home/team-005/data/InspecSafe-V1"

cd "$REPO_ROOT"

# --- 後片付け: スクリプト終了時に config を復元 ---
cleanup() {
    if [[ -f "$CONFIG_BAK" ]]; then
        cp "$CONFIG_BAK" "$CONFIG_FILE"
        rm -f "$CONFIG_BAK"
        echo "[batch] configs/default.yaml を復元しました"
    fi
    # data/videos/ のシンボリックリンクをクリア
    if [[ -d "$VIDEOS_DIR" ]]; then
        find "$VIDEOS_DIR" -maxdepth 1 -type l -delete 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- 事前準備 ---
cp "$CONFIG_FILE" "$CONFIG_BAK"
mkdir -p "$RESULTS_DIR"
mkdir -p "$VIDEOS_DIR"

echo "[batch] 開始: $(date)"
echo "[batch] 結果保存先: $RESULTS_DIR"

# --- YAML 書き換え関数（Python yaml モジュール使用）---
update_config_inspesafe() {
    local session_rel="$1"
    python3 - "$CONFIG_BAK" "$CONFIG_FILE" "$session_rel" <<'PYEOF'
import sys, yaml

src, dst, session_rel = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("data", {})["mode"] = "inspesafe"
cfg["data"].setdefault("inspesafe", {})["session"] = session_rel

with open(dst, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
              sort_keys=False)
PYEOF
}

update_config_manual() {
    python3 - "$CONFIG_BAK" "$CONFIG_FILE" <<'PYEOF'
import sys, yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("data", {})["mode"] = "manual"

with open(dst, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
              sort_keys=False)
PYEOF
}

# --- data/ 以下の全出力を指定ディレクトリへコピー ---
# videos/（入力シンボリックリンク）と results_archive/（自動アーカイブ）は除外
copy_data_outputs() {
    local out_dir="$1"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"
    local copied=0
    for src in "$REPO_ROOT/data"/*/; do
        local dirname
        dirname="$(basename "$src")"
        case "$dirname" in
            videos|results_archive) continue ;;
        esac
        if [[ -d "$src" ]]; then
            cp -r "$src" "$out_dir/$dirname"
            copied=$((copied + 1))
        fi
    done
    echo "[batch] $copied ディレクトリをコピー → $out_dir"
}

# --- inspesafe セッション実行関数 ---
run_inspesafe() {
    local session_name="$1"
    echo ""
    echo "=========================================="
    echo "[inspesafe] セッション: $session_name"
    echo "=========================================="

    # セッションディレクトリを動的に検索
    local session_rel
    session_rel=$(find "$DATASET_PATH/DATA_PATH" -type d -name "$session_name" \
        -path "*/Other_modalities/*" 2>/dev/null | head -1 \
        | sed "s|$DATASET_PATH/DATA_PATH/||")

    if [[ -z "$session_rel" ]]; then
        echo "[inspesafe] 警告: セッションが見つかりません: $session_name (スキップ)"
        return 0
    fi
    echo "[inspesafe] セッションパス: $session_rel"

    update_config_inspesafe "$session_rel"
    python src/run.py

    # 結果コピー（data/ 以下の全出力ディレクトリをセッション名のディレクトリへ）
    local out_dir="$RESULTS_DIR/$session_name"
    copy_data_outputs "$out_dir"
    echo "[inspesafe] 結果保存: $out_dir"
}

# --- manual 動画実行関数 ---
run_manual() {
    local video_path="$1"
    local video_stem
    video_stem="$(basename "$video_path" .mp4)"

    echo ""
    echo "=========================================="
    echo "[manual] 動画: $video_path"
    echo "=========================================="

    if [[ ! -f "$video_path" ]]; then
        echo "[manual] 警告: 動画ファイルが見つかりません: $video_path (スキップ)"
        return 0
    fi

    # data/videos/ のシンボリックリンクをクリアして対象動画のリンクを張る
    find "$VIDEOS_DIR" -maxdepth 1 -type l -delete 2>/dev/null || true
    ln -sf "$video_path" "$VIDEOS_DIR/$(basename "$video_path")"

    update_config_manual
    python src/run.py

    # 結果コピー（data/ 以下の全出力ディレクトリを動画名のディレクトリへ）
    local out_dir="$RESULTS_DIR/$video_stem"
    copy_data_outputs "$out_dir"
    echo "[manual] 結果保存: $out_dir"
}

# ==========================================================
# InspecSafe セッション × 7
# ==========================================================
INSPESAFE_SESSIONS=(
    "58132919741958_20251112_session_1200_beiyongguanAyalibiao"
    "58132919535743_20251118_session_1400_16#refengweiguan-you"
    "58132919535777_20251114_session_1400_K76+681 xiaofangshoubaoxiang2"
    "58132919742054_20251112_session0200#16pidaiA-quyu1-xiacenghuichengtuogun（1-2haotuogun）"
    "58132919742054_20251112_session0200#16pidaiA-quyu30-shangcengtuogun（58-59haotuogun）"
    "58132919742126_20251114_session_0500_1haozhubianAxiangbileiqi"
    "58132919742224_20251111_session_1600_tuogun24"
)

for session in "${INSPESAFE_SESSIONS[@]}"; do
    run_inspesafe "$session"
done

# ==========================================================
# manual 動画 × 7
# ==========================================================
MANUAL_VIDEOS=(
    "/home/team-005/data/hazard-detection/dataset/videos/cable.mp4"
    "/home/team-005/data/hazard-detection/dataset/videos/human.mp4"
    "/home/team-005/data/hazard-detection/dataset/videos/cones.mp4"
    "/home/team-005/data/gen_video/blind_spot_forklift_42_20260321_125558.mp4"
    "/home/team-005/data/gen_video/welding_sparks_hazard_123_20260321_125817.mp4"
    "/home/team-005/data/gen_video/gemini.mp4"
    "/home/team-005/data/gen_video/sora.mp4"
)

for video in "${MANUAL_VIDEOS[@]}"; do
    run_manual "$video"
done

echo ""
echo "=========================================="
echo "[batch] 完了: $(date)"
echo "[batch] 結果一覧: $RESULTS_DIR"
ls "$RESULTS_DIR" 2>/dev/null | sed 's/^/  /'
echo "=========================================="
