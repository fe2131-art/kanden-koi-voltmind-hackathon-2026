"""InspecSafe-V1 データビューワ (Gradio)

起動:
    uv run python src/vis/viewer.py [--port 7860] [--share]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from datasets import Dataset
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# パス定義
# ---------------------------------------------------------------------------
CACHE_BASE = Path(
    "/home/team-005/data/hf_cache"
    "/Tetrabot2026___inspec_safe-v1/default/0.0.0"
    "/6d58d012079ba6441c474be54d7a93dd9d70c01c"
)
TAR_BASE = Path(
    "/home/team-005/work/team-005/.cache/huggingface/hub"
    "/datasets--Tetrabot2026--InspecSafe-V1/snapshots"
    "/6d58d012079ba6441c474be54d7a93dd9d70c01c"
)
TAR_PATHS: dict[str, Path] = {
    "train": TAR_BASE / "train.tar",
    "test": TAR_BASE / "test.tar",
}
SHARD_NAMES: dict[str, list[str]] = {
    "train": [f"inspec_safe-v1-train-{i:05d}-of-00011.arrow" for i in range(11)],
    "test": [f"inspec_safe-v1-test-{i:05d}-of-00004.arrow" for i in range(4)],
}

# ---------------------------------------------------------------------------
# ラベルカラーパレット (LabelMe polygon 用)
# ---------------------------------------------------------------------------
PALETTE = [
    (255, 82, 82),
    (82, 175, 255),
    (82, 255, 130),
    (255, 210, 82),
    (220, 82, 255),
    (82, 255, 220),
    (255, 130, 82),
    (130, 82, 255),
    (255, 255, 82),
    (82, 230, 180),
    (200, 82, 130),
    (130, 200, 82),
]
_label_color_cache: dict[str, tuple[int, int, int]] = {}


def _label_color(label: str) -> tuple[int, int, int]:
    if label not in _label_color_cache:
        idx = len(_label_color_cache) % len(PALETTE)
        _label_color_cache[label] = PALETTE[idx]
    return _label_color_cache[label]


# ---------------------------------------------------------------------------
# Arrow データセット管理
# ---------------------------------------------------------------------------
class AnnotationStore:
    """全 Arrow シャードを遅延ロードし、アノテーション付きフレームを提供する。"""

    def __init__(self, split: str) -> None:
        self.split = split
        self._rows: list[dict] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        rows: list[dict] = []
        for shard in SHARD_NAMES[self.split]:
            path = CACHE_BASE / shard
            if not path.exists():
                continue
            ds = Dataset.from_file(str(path))
            for i in range(len(ds)):
                row = ds[i]
                jpg = row.get("jpg")
                if jpg is None:
                    continue  # 画像なし行はスキップ
                rows.append(row)
        self._rows = rows
        self._loaded = True
        logger.info("%s: %d annotated frames loaded", self.split, len(rows))

    # ------------------------------------------------------------------ #
    def all_rows(self) -> list[dict]:
        self._ensure_loaded()
        return self._rows

    def anomaly_types(self) -> list[str]:
        self._ensure_loaded()
        types: set[str] = set()
        for r in self._rows:
            key = r.get("__key__", "")
            atype = _anomaly_type(key)
            types.add(atype)
        return sorted(types)

    def filter(self, anomaly_type: str | None) -> list[dict]:
        self._ensure_loaded()
        if not anomaly_type or anomaly_type == "すべて":
            return self._rows
        return [r for r in self._rows if _anomaly_type(r.get("__key__", "")) == anomaly_type]


_stores: dict[str, AnnotationStore] = {
    "train": AnnotationStore("train"),
    "test": AnnotationStore("test"),
}


def _anomaly_type(key: str) -> str:
    """キーからカテゴリ名を抽出する。"""
    if "Anomaly_data" in key:
        filename = key.split("/")[-1]
        for suffix in ("_frame_", "_visible_"):
            if suffix in filename:
                return filename.split(suffix)[0]
        return filename
    if "Normal_data" in key:
        return "Normal"
    return "Other"


# ---------------------------------------------------------------------------
# 画像描画ユーティリティ
# ---------------------------------------------------------------------------
def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def draw_annotations(pil_img: Image.Image, shapes: list[dict] | None) -> Image.Image:
    """LabelMe shape (polygon / rectangle) をオーバーレイ描画する。"""
    if not shapes:
        return pil_img

    overlay = pil_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = shape.get("label") or "?"
        points = shape.get("points") or []
        shape_type = shape.get("shape_type", "polygon")
        color = _label_color(label)
        fill_color = color + (50,)   # 半透明塗り
        line_color = color + (220,)  # 線

        if len(points) < 2:
            continue
        xy = [(float(p[0]), float(p[1])) for p in points]

        if shape_type in ("polygon", "rectangle"):
            draw.polygon(xy, fill=fill_color, outline=line_color)
        elif shape_type == "linestrip":
            draw.line(xy, fill=line_color, width=2)
        else:
            draw.polygon(xy, fill=fill_color, outline=line_color)

        # ラベルテキスト
        tx, ty = xy[0]
        draw.rectangle([tx, ty - 16, tx + len(label) * 8, ty], fill=color + (200,))
        draw.text((tx + 2, ty - 15), label, fill=(255, 255, 255, 255))

    return Image.alpha_composite(overlay.convert("RGBA"), Image.new("RGBA", overlay.size, (0, 0, 0, 0))).convert("RGB")


def row_to_annotated_image(row: dict) -> np.ndarray:
    pil_img = row["jpg"]
    json_data = row.get("json") or {}
    shapes = json_data.get("shapes") if isinstance(json_data, dict) else None
    img = draw_annotations(pil_img, shapes)
    return _pil_to_numpy(img)


def shapes_to_text(shapes: list[dict] | None) -> str:
    if not shapes:
        return "(アノテーションなし)"
    label_counts: dict[str, int] = defaultdict(int)
    for s in shapes:
        if isinstance(s, dict) and s.get("label"):
            label_counts[s["label"]] += 1
    return "\n".join(f"  {label}: {cnt} 個" for label, cnt in sorted(label_counts.items()))


# ---------------------------------------------------------------------------
# セッションインデックス (tar.gz 対応)
# ---------------------------------------------------------------------------
# tar ファイルは gzip 圧縮 (magic=1f8b) のためランダムシーク不可。
# Arrow ファイルの __key__ からメンバーパスを再構築し、
# システムの tar コマンドで抽出する。抽出結果はディスクにキャッシュ。
# ---------------------------------------------------------------------------

# モダリティ → 拡張子のマッピング
_MODALITY_EXT: dict[str, str] = {
    "_visible_": ".mp4",
    "_infrared_": ".mp4",
    "_audio_": ".wav",
    "_point_cloud_": ".bag",
    "_sensor_": ".txt",
}

# キャッシュディレクトリ
_CACHE_ROOT = Path(tempfile.gettempdir()) / "inspec_safe_v1_cache"


class SessionIndex:
    """Arrow ファイルから Other_modalities セッション一覧を構築し、
    システム tar コマンドで個別ファイルを抽出するクラス。"""

    def __init__(self, split: str) -> None:
        self.split = split
        self.tar_path = TAR_PATHS[split]
        self._cache_dir = _CACHE_ROOT / split
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # session_dir -> {modality_key: tar_member_path}
        # 例: "visible" -> "DATA_PATH/train/Other_modalities/session/.../foo_visible_20251101.mp4"
        self._sessions: dict[str, dict[str, str]] = defaultdict(dict)
        self._built = False

    def _build(self) -> None:
        if self._built:
            return
        logger.info("[%s] Arrow からセッション一覧を構築中...", self.split)
        for shard_name in SHARD_NAMES[self.split]:
            shard_path = CACHE_BASE / shard_name
            if not shard_path.exists():
                continue
            ds = Dataset.from_file(str(shard_path))
            for i in range(len(ds)):
                key: str = ds[i].get("__key__", "")
                if "Other_modalities" not in key:
                    continue
                # パス構造: DATA_PATH/{split}/Other_modalities/{session_folder}/{filename_no_ext}
                parts = key.split("/")
                if len(parts) < 5:
                    continue
                session_dir = parts[3]  # 例: "58132919742254_20251118_session_1000_..."
                filename = parts[4]     # 例: "#0 biangaoyacechuxian_sensor_20251101"
                for mod_key, ext in _MODALITY_EXT.items():
                    if mod_key in filename:
                        modality = mod_key.strip("_")  # e.g. "visible"
                        self._sessions[session_dir][modality] = key + ext
                        break
        self._built = True
        logger.info("[%s] セッション数: %d", self.split, len(self._sessions))

    def session_list(self) -> list[str]:
        self._build()
        return sorted(self._sessions.keys())

    def session_modalities(self, session: str) -> dict[str, str]:
        """{modality: tar_member_path} を返す。"""
        self._build()
        return dict(self._sessions.get(session, {}))

    def _cache_path(self, member_path: str) -> Path:
        return self._cache_dir / member_path.replace("/", "__")

    def extract_bytes(self, member_path: str) -> bytes:
        """tar.gz から指定メンバーをキャッシュ込みで抽出する。"""
        cached = self._cache_path(member_path)
        if cached.exists() and cached.stat().st_size > 0:
            return cached.read_bytes()

        logger.info("tar から抽出中: %s", member_path)
        result = subprocess.run(
            ["tar", "-xOf", str(self.tar_path), member_path],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")[:300]
            raise RuntimeError(f"tar 抽出失敗 ({member_path}): {err}")

        data = result.stdout
        cached.write_bytes(data)
        return data

    def extract_session(self, session: str) -> dict[str, Path]:
        """セッションの全ファイルを 1 回の tar 呼び出しで一括抽出し、キャッシュパスを返す。

        tar.gz はシーケンシャル読み出しのため、複数ファイルを別々に抽出すると
        その都度アーカイブ全体をスキャンする。一括抽出で所要時間を 1/n に削減する。
        """
        modalities = self.session_modalities(session)
        result: dict[str, Path] = {}
        missing: list[tuple[str, str]] = []  # [(modality, member_path)]

        for mod, member_path in modalities.items():
            cached = self._cache_path(member_path)
            if cached.exists() and cached.stat().st_size > 0:
                result[mod] = cached
            else:
                missing.append((mod, member_path))

        if not missing:
            return result

        # 一括抽出: tar -xzf archive -C tmpdir file1 file2 ...
        tmpdir = Path(tempfile.mkdtemp(prefix="inspec_extract_"))
        try:
            member_names = [mp for _, mp in missing]
            logger.info("tar 一括抽出 (%d ファイル): session=%s", len(member_names), session)
            subprocess.run(
                ["tar", "-xzf", str(self.tar_path), "-C", str(tmpdir)] + member_names,
                capture_output=True,
                timeout=180,
            )
            # 抽出されたファイルをキャッシュに移動
            for mod, member_path in missing:
                extracted = tmpdir / member_path
                if extracted.exists() and extracted.stat().st_size > 0:
                    cached = self._cache_path(member_path)
                    shutil.copy2(extracted, cached)
                    result[mod] = cached
                    logger.info("  キャッシュ: %s (%d KB)", mod, cached.stat().st_size // 1024)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return result


_session_indices: dict[str, SessionIndex] = {}


def _get_session_index(split: str) -> SessionIndex:
    if split not in _session_indices:
        _session_indices[split] = SessionIndex(split)
    return _session_indices[split]


def extract_video_frame(video_bytes: bytes, frame_idx: int = 0) -> np.ndarray | None:
    """mp4 バイト列から指定フレームを RGB numpy 配列として返す。"""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp_path = f.name
    try:
        return _read_frame(tmp_path, frame_idx)
    finally:
        os.unlink(tmp_path)


def extract_video_frame_from_path(video_path: Path, frame_idx: int = 0) -> np.ndarray | None:
    """キャッシュ済み mp4 ファイルから指定フレームを RGB numpy 配列として返す。"""
    return _read_frame(str(video_path), frame_idx)


def _read_frame(path: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def parse_sensor_txt(txt: str) -> str:
    """sensor JSON テキストを読みやすい Markdown テーブルに変換する。"""
    try:
        data = json.loads(txt)
    except Exception:
        return txt[:500]

    env = data.get("env", [])
    if not env:
        return "(センサデータなし)"

    lines = ["| センサ | 値 | 単位 | アラーム |", "| --- | --- | --- | --- |"]
    for e in env:
        name = e.get("name", "?")
        val = e.get("showValue", e.get("value", "-"))
        unit = e.get("unit", "")
        warn = e.get("warn", 0)
        alarm = "⚠️" if warn else ""
        lines.append(f"| {name} | {val} | {unit} | {alarm} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="InspecSafe-V1 ビューワ") as demo:
        gr.Markdown(
            "# InspecSafe-V1 データビューワ\n"
            "**Tetrabot2026/InspecSafe-V1** — 電力設備点検ロボットのマルチモーダルデータセット"
        )

        with gr.Tabs():
            # ================================================================
            # Tab 1: アノテーション閲覧 (RGB + LabelMe polygon)
            # ================================================================
            with gr.Tab("アノテーション閲覧 (RGB)"):
                gr.Markdown(
                    "LabelMe アノテーション付き RGB フレームを閲覧します。"
                    "ポリゴンは各ラベルに対応した色でオーバーレイ表示されます。"
                )
                with gr.Row():
                    ann_split = gr.Dropdown(
                        ["train", "test"], value="test", label="Split", scale=1
                    )
                    ann_type = gr.Dropdown(
                        ["すべて"], value="すべて", label="カテゴリフィルタ", scale=2
                    )
                with gr.Row():
                    ann_prev_btn = gr.Button("◀ 前へ", scale=1)
                    ann_idx = gr.Slider(
                        minimum=0, maximum=0, step=1, value=0, label="フレームインデックス", scale=5
                    )
                    ann_next_btn = gr.Button("次へ ▶", scale=1)

                with gr.Row():
                    ann_image = gr.Image(
                        label="RGB + アノテーション (1920×1080)", type="numpy", scale=3
                    )
                    with gr.Column(scale=1):
                        ann_key = gr.Textbox(label="データキー", lines=3)
                        ann_labels = gr.Textbox(label="検出ラベル", lines=6)
                        ann_desc = gr.Textbox(label="シーン説明 (txt)", lines=6)

                # --- コールバック ---
                def on_split_change(split: str):
                    store = _stores[split]
                    types = ["すべて"] + store.anomaly_types()
                    rows = store.filter(None)
                    n = max(0, len(rows) - 1)
                    return (
                        gr.update(choices=types, value="すべて"),
                        gr.update(maximum=n, value=0),
                    )

                def on_type_change(split: str, atype: str):
                    rows = _stores[split].filter(atype)
                    n = max(0, len(rows) - 1)
                    return gr.update(maximum=n, value=0)

                def on_frame_change(split: str, atype: str, idx: int):
                    rows = _stores[split].filter(atype)
                    if not rows:
                        blank = np.zeros((480, 640, 3), dtype=np.uint8)
                        return blank, "", "(フレームなし)", "(フレームなし)"
                    idx = min(idx, len(rows) - 1)
                    row = rows[idx]
                    img = row_to_annotated_image(row)
                    key = row.get("__key__", "")
                    json_data = row.get("json") or {}
                    shapes = json_data.get("shapes") if isinstance(json_data, dict) else None
                    label_txt = shapes_to_text(shapes)
                    desc = row.get("txt") or "(説明なし)"
                    return img, key, label_txt, desc

                def ann_step(split: str, atype: str, idx: int, delta: int) -> int:
                    rows = _stores[split].filter(atype)
                    n = max(0, len(rows) - 1)
                    return int(max(0, min(n, idx + delta)))

                ann_split.change(
                    on_split_change,
                    inputs=ann_split,
                    outputs=[ann_type, ann_idx],
                )
                ann_type.change(
                    on_type_change,
                    inputs=[ann_split, ann_type],
                    outputs=ann_idx,
                )
                ann_idx.change(
                    on_frame_change,
                    inputs=[ann_split, ann_type, ann_idx],
                    outputs=[ann_image, ann_key, ann_labels, ann_desc],
                )
                ann_split.change(
                    lambda split, atype, idx: on_frame_change(split, atype, idx),
                    inputs=[ann_split, ann_type, ann_idx],
                    outputs=[ann_image, ann_key, ann_labels, ann_desc],
                )
                ann_prev_btn.click(
                    lambda split, atype, idx: ann_step(split, atype, idx, -1),
                    inputs=[ann_split, ann_type, ann_idx],
                    outputs=ann_idx,
                ).then(
                    on_frame_change,
                    inputs=[ann_split, ann_type, ann_idx],
                    outputs=[ann_image, ann_key, ann_labels, ann_desc],
                )
                ann_next_btn.click(
                    lambda split, atype, idx: ann_step(split, atype, idx, +1),
                    inputs=[ann_split, ann_type, ann_idx],
                    outputs=ann_idx,
                ).then(
                    on_frame_change,
                    inputs=[ann_split, ann_type, ann_idx],
                    outputs=[ann_image, ann_key, ann_labels, ann_desc],
                )

                # 初期表示ボタン
                ann_load_btn = gr.Button("データ読み込み / 更新", variant="primary")
                ann_load_btn.click(
                    on_split_change,
                    inputs=ann_split,
                    outputs=[ann_type, ann_idx],
                ).then(
                    on_frame_change,
                    inputs=[ann_split, ann_type, ann_idx],
                    outputs=[ann_image, ann_key, ann_labels, ann_desc],
                )

            # ================================================================
            # Tab 2: セッション閲覧 (可視光 + 赤外線)
            # ================================================================
            with gr.Tab("セッション閲覧 (可視光 + 赤外線)"):
                gr.Markdown(
                    "点検セッションごとに **可視光カメラ** と **赤外線カメラ** の映像フレームを並べて閲覧します。\n\n"
                    "> **注意**: tar からのストリーミング読み出しのため、初回ロードに数秒かかります。"
                )
                with gr.Row():
                    ses_split = gr.Dropdown(
                        ["train", "test"], value="test", label="Split", scale=1
                    )
                    ses_session = gr.Dropdown(
                        choices=[], label="セッション", scale=4
                    )
                with gr.Row():
                    ses_prev_btn = gr.Button("◀ 前へ", scale=1)
                    ses_frame_idx = gr.Slider(
                        minimum=0, maximum=30, step=1, value=0, label="フレーム番号", scale=5
                    )
                    ses_next_btn = gr.Button("次へ ▶", scale=1)

                with gr.Row():
                    ses_vis_img = gr.Image(label="可視光 (RGB)", type="numpy", scale=1)
                    ses_ir_img = gr.Image(label="赤外線 (Infrared)", type="numpy", scale=1)

                with gr.Row():
                    ses_sensor = gr.Markdown(label="環境センサデータ", value="(セッションを選択してください)")

                ses_load_btn = gr.Button("セッション一覧を読み込む", variant="primary")

                # --- コールバック ---
                def on_ses_split_change(split: str):
                    idx = _get_session_index(split)
                    sessions = idx.session_list()
                    if sessions:
                        return gr.update(choices=sessions, value=sessions[0])
                    return gr.update(choices=[], value=None)

                def on_session_select(split: str, session: str | None, frame_idx: int):
                    blank = np.zeros((360, 640, 3), dtype=np.uint8)
                    if not session:
                        return blank, blank, "(セッションを選択してください)"

                    idx = _get_session_index(split)
                    # 一括抽出（初回のみ tar を 1 回スキャン、以降はキャッシュから即時返却）
                    cached_files = idx.extract_session(session)

                    vis_frame: np.ndarray | None = None
                    ir_frame: np.ndarray | None = None
                    sensor_md = "(センサデータなし)"

                    # 可視光 mp4
                    if "visible" in cached_files:
                        try:
                            vis_frame = extract_video_frame_from_path(
                                cached_files["visible"], frame_idx
                            )
                        except Exception as e:
                            logger.warning("可視光フレーム読み込みエラー: %s", e)

                    # 赤外線 mp4
                    if "infrared" in cached_files:
                        try:
                            ir_frame = extract_video_frame_from_path(
                                cached_files["infrared"], frame_idx
                            )
                        except Exception as e:
                            logger.warning("赤外線フレーム読み込みエラー: %s", e)

                    # 環境センサ txt
                    if "sensor" in cached_files:
                        try:
                            txt = cached_files["sensor"].read_text(errors="replace")
                            sensor_md = parse_sensor_txt(txt)
                        except Exception as e:
                            logger.warning("センサデータ読み込みエラー: %s", e)

                    return (
                        vis_frame if vis_frame is not None else blank,
                        ir_frame if ir_frame is not None else blank,
                        sensor_md,
                    )

                ses_load_btn.click(
                    on_ses_split_change,
                    inputs=ses_split,
                    outputs=ses_session,
                ).then(
                    on_session_select,
                    inputs=[ses_split, ses_session, ses_frame_idx],
                    outputs=[ses_vis_img, ses_ir_img, ses_sensor],
                )
                ses_session.change(
                    on_session_select,
                    inputs=[ses_split, ses_session, ses_frame_idx],
                    outputs=[ses_vis_img, ses_ir_img, ses_sensor],
                )
                ses_frame_idx.change(
                    on_session_select,
                    inputs=[ses_split, ses_session, ses_frame_idx],
                    outputs=[ses_vis_img, ses_ir_img, ses_sensor],
                )
                ses_prev_btn.click(
                    lambda idx: max(0, idx - 1),
                    inputs=ses_frame_idx,
                    outputs=ses_frame_idx,
                ).then(
                    on_session_select,
                    inputs=[ses_split, ses_session, ses_frame_idx],
                    outputs=[ses_vis_img, ses_ir_img, ses_sensor],
                )
                ses_next_btn.click(
                    lambda idx: idx + 1,
                    inputs=ses_frame_idx,
                    outputs=ses_frame_idx,
                ).then(
                    on_session_select,
                    inputs=[ses_split, ses_session, ses_frame_idx],
                    outputs=[ses_vis_img, ses_ir_img, ses_sensor],
                )

            # ================================================================
            # Tab 3: カメラ仕様
            # ================================================================
            with gr.Tab("センサ仕様"):
                gr.Markdown("""
## 搭載センサ仕様

### RGB カメラ (可視光)
| 項目 | 値 |
| --- | --- |
| センサ | 1/2.8 inch CMOS |
| 最大解像度 | 2560×1440 @ 25fps |
| 最低照度 | 0.005 lux (カラー) |
| 光学手ブレ補正 | あり |
| 圧縮方式 | SmartH.265 / H.264 / MJPEG |

### 赤外線カメラ (IR)
| 項目 | 値 |
| --- | --- |
| センサ | 非冷却型赤外線検出器 |
| 解像度 | 256×192 (標準) / 384×288 (Pro) |
| 視野角 (FOV) | 水平 53.7° × 垂直 39.7° |
| 温度計測範囲 | -20℃ 〜 +150℃ |
| 圧縮方式 | H.265 / H.264 |

### 深度カメラ
| 項目 | 値 |
| --- | --- |
| モデル | TM265-E1 |
| 計測距離 | 0.05m 〜 5m |
| 解像度 | 240×96 |
| フレームレート | 25fps (BASIC) / 15fps (MEDIUM) |
| Z軸精度 | ±10mm + 0.5% |
| FOV | 水平 100° × 垂直 50° |
| データ形式 | **ROS bag (.bag) ファイルに収録** |

> 深度データは `.bag` 形式で格納されています。
> 展開には `rosbag` または `rosbags` ライブラリが必要です。

### LiDAR
| 項目 | 値 |
| --- | --- |
| モデル | **Livox MID-360** |
| 水平視野角 | 360° |
| 垂直視野角 | -7° 〜 +52° |
| 点群密度 | 200,000 点/秒 |
| フレームレート | 10 Hz |
| 最大計測距離 | 70m (反射率 80%) |
| IMU | 内蔵 (ICM40609) |
| データ形式 | **ROS bag (.bag) ファイルに収録** |
| 防塵防水 | IP67 |

### 環境センサ
| センサ種別 | 単位 | 用途 |
| --- | --- | --- |
| CO (一酸化炭素) | PPM | 有毒ガス検知 |
| O₂ (酸素) | %VOL | 酸欠検知 |
| CH₄ (メタン) | %VOL | 爆発性ガス |
| H₂S (硫化水素) | PPM | 有毒ガス |
| PM1.0 / PM2.5 / PM10 | μg/m³ | 粉塵 |
| 煙 (smog) | - | 火災前兆 |
| 温度 | ℃ | 環境温度 |
| 湿度 | % | 環境湿度 |
""")

    return demo


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="InspecSafe-V1 データビューワ")
    parser.add_argument("--port", type=int, default=7860, help="起動ポート (デフォルト: 7860)")
    parser.add_argument("--share", action="store_true", help="Gradio share リンクを発行する")
    parser.add_argument("--host", default="0.0.0.0", help="バインドアドレス")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    '''
    uv run python src/vis/viewer.py --port 7860
    '''    
    main()
