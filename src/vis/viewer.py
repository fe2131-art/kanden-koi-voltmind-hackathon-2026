"""InspecSafe-V1 データビューワ (Gradio)

起動:
    uv run python src/vis/viewer.py [--port 7860] [--share]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# セッションインデックス
# ---------------------------------------------------------------------------

# モダリティ → 拡張子のマッピング
_MODALITY_EXT: dict[str, str] = {
    "_visible_": ".mp4",
    "_infrared_": ".mp4",
    "_audio_": ".wav",
    "_point_cloud_": ".bag",
    "_sensor_": ".txt",
}

# 永続キャッシュディレクトリ（再起動後も保持される）
_EXTRACT_ROOT = Path("/home/team-005/data/inspec_extracted")

# セッション モダリティ CSV
CSV_PATH = Path(__file__).parent.parent.parent / "data" / "session_modalities.csv"
_CSV_FIELDNAMES = [
    "split", "robot_id", "modality_dir_path",
    "has_rgb_video", "has_infrared_video", "has_audio", "has_point_cloud", "has_sensor_log",
]


class SessionIndex:
    """Other_modalities セッション一覧を管理するクラス。

    _EXTRACT_ROOT/{split}/{session_dir}/{filename} のファイルシステムを直接スキャンする。
    """

    def __init__(self, split: str) -> None:
        self.split = split
        self._cache_dir = _EXTRACT_ROOT / split
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # session_dir -> {modality: file_path}
        self._sessions: dict[str, dict[str, Path]] = defaultdict(dict)
        self._built = False

    # ------------------------------------------------------------------ #
    # セッション一覧構築
    # ------------------------------------------------------------------ #

    def _build(self) -> None:
        if self._built:
            return
        found = 0
        for session_dir_path in self._cache_dir.iterdir():
            if not session_dir_path.is_dir():
                continue
            session_dir = session_dir_path.name
            for cached_file in session_dir_path.iterdir():
                if not cached_file.is_file() or cached_file.stat().st_size == 0:
                    continue
                filename = cached_file.name
                for mod_key in _MODALITY_EXT:
                    if mod_key in filename:
                        modality = mod_key.strip("_")
                        self._sessions[session_dir][modality] = cached_file
                        found += 1
                        break
        self._built = True
        logger.info("[%s] %d セッション構築", self.split, len(self._sessions))

    def session_list(self) -> list[str]:
        self._build()
        return sorted(self._sessions.keys())

    def session_modalities(self, session: str) -> dict[str, Path]:
        """{modality: file_path} を返す。"""
        self._build()
        return dict(self._sessions.get(session, {}))

    def extract_session(self, session: str) -> dict[str, Path]:
        """セッションのファイルパスを返す。"""
        return self.session_modalities(session)


_session_indices: dict[str, SessionIndex] = {}


def _get_session_index(split: str) -> SessionIndex:
    if split not in _session_indices:
        _session_indices[split] = SessionIndex(split)
    return _session_indices[split]


# ---------------------------------------------------------------------------
# セッション モダリティ CSV 管理
# ---------------------------------------------------------------------------

def generate_session_csv(splits: list[str] | None = None) -> str:
    """SessionIndex から CSV を生成/更新し、結果メッセージを返す。"""
    if splits is None:
        splits = ["train", "test"]

    rows: list[dict[str, str]] = []
    for split in splits:
        idx = _get_session_index(split)
        idx._build()
        for session_dir, modalities in idx._sessions.items():
            modality_dir_path = (
                f"../InspecSafe-V1/DATA_PATH/{split}/Other_modalities/{session_dir}"
            )
            rows.append({
                "split": split,
                "robot_id": session_dir,
                "modality_dir_path": modality_dir_path,
                "has_rgb_video": "Yes" if "visible" in modalities else "No",
                "has_infrared_video": "Yes" if "infrared" in modalities else "No",
                "has_audio": "Yes" if "audio" in modalities else "No",
                "has_point_cloud": "Yes" if "point_cloud" in modalities else "No",
                "has_sensor_log": "Yes" if "sensor" in modalities else "No",
            })

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    msg = f"CSV 生成完了: {len(rows)} セッション → {CSV_PATH}"
    logger.info(msg)
    return msg


def load_session_csv() -> dict[tuple[str, str], dict[str, str]]:
    """CSV を読み込み {(split, robot_id): row_dict} を返す。CSV がなければ空 dict。"""
    result: dict[tuple[str, str], dict[str, str]] = {}
    if not CSV_PATH.exists():
        return result
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["split"], row["robot_id"])
            result[key] = dict(row)
    logger.info("CSV 読み込み: %d 件", len(result))
    return result


def filter_sessions_by_modality(
    split: str,
    sessions: list[str],
    csv_data: dict[tuple[str, str], dict[str, str]],
    need_rgb: bool,
    need_ir: bool,
    need_audio: bool,
    need_lidar: bool,
) -> list[str]:
    """CSV を参照してモダリティ条件に合うセッションだけ返す。
    CSV が空の場合はフィルタせずそのまま返す。
    """
    if not csv_data:
        return sessions

    filtered: list[str] = []
    for session in sessions:
        row = csv_data.get((split, session))
        if row is None:
            continue
        if need_rgb and row.get("has_rgb_video") != "Yes":
            continue
        if need_ir and row.get("has_infrared_video") != "Yes":
            continue
        if need_audio and row.get("has_audio") != "Yes":
            continue
        if need_lidar and row.get("has_point_cloud") != "Yes":
            continue
        filtered.append(session)
    return filtered


def extract_video_frame_from_path(video_path: Path, frame_idx: int = 0) -> np.ndarray | None:
    """キャッシュ済み mp4 ファイルから指定フレームを RGB numpy 配列として返す。"""
    return _read_frame(str(video_path), frame_idx)


def _read_frame(path: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # total==0 はコンテナにフレーム数メタデータがない場合（H.264/H.265 で頻発）。
    # その場合でも cap.set() を呼ぶことでシークを試みる。
    seek_to = min(frame_idx, total - 1) if total > 0 else frame_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
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
# LiDAR BEV 描画
# ---------------------------------------------------------------------------

# PointCloud2 datatype -> numpy dtype のマッピング (sensor_msgs/PointField 定数)
_PC2_DTYPE: dict[int, type] = {
    1: np.uint8, 2: np.int8, 3: np.uint16, 4: np.int16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}

# Livox/一般的な LiDAR トピック候補
_LIDAR_TOPICS = [
    "/livox/lidar", "/livox/lidar1", "/points", "/lidar_points",
    "/velodyne_points", "/scan_3d", "/rslidar_points",
]


def _extract_xyz_from_pc2(msg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PointCloud2 メッセージから x, y, z の numpy 配列を返す。"""
    field_map: dict[str, tuple[int, type]] = {}
    for f in msg.fields:
        dt = _PC2_DTYPE.get(int(f.datatype), np.float32)
        field_map[f.name] = (int(f.offset), dt)

    if "x" not in field_map or "y" not in field_map:
        return np.array([]), np.array([]), np.array([])

    n_points = int(msg.width) * int(msg.height)
    step = int(msg.point_step)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n_points, step)

    def _field(name: str) -> np.ndarray:
        off, dt = field_map[name]
        size = np.dtype(dt).itemsize
        return np.frombuffer(raw[:, off : off + size].copy().tobytes(), dtype=dt)

    xs = _field("x")
    ys = _field("y")
    zs = _field("z") if "z" in field_map else np.zeros(n_points, dtype=np.float32)
    return xs, ys, zs


# LiDAR フレーム数キャッシュ (bag_path -> total_frames)
_lidar_frame_count_cache: dict[Path, int] = {}


def _render_bev_from_points(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    max_range: float,
    img_size: int,
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    """点群から BEV 画像 (RGB numpy) を生成する。cv2 ベースで確実に描画される。

    セル単位の平均高さをカラー、密度を輝度としてエンコードする。
    """
    bins = img_size
    x_edges = np.linspace(-max_range, max_range, bins + 1)
    y_edges = np.linspace(-max_range, max_range, bins + 1)

    density, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    z_sum, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=zs)

    # セルごとの平均高さ
    with np.errstate(invalid="ignore", divide="ignore"):
        z_mean = np.where(density > 0, z_sum / density, np.nan)

    # 高さを [0, 255] に正規化
    valid = np.isfinite(z_mean)
    if valid.any():
        z_min, z_max = float(np.nanmin(z_mean)), float(np.nanmax(z_mean))
        if z_max > z_min:
            z_norm = np.where(valid, (z_mean - z_min) / (z_max - z_min), 0.0)
        else:
            z_norm = np.where(valid, 0.5, 0.0)
    else:
        z_norm = np.zeros_like(z_mean)

    # 輝度: log(density) で正規化
    density_log = np.log1p(density)
    bright = (density_log / density_log.max()) if density_log.max() > 0 else density_log

    # 高さカラー × 密度輝度
    # histogram2d の軸は (x_bins, y_bins)。画像座標に変換: 転置 + Y 反転（前方が上）
    z_uint8 = (z_norm * 255).astype(np.uint8).T[::-1]
    bright_2d = bright.T[::-1].astype(np.float32)

    colored = cv2.applyColorMap(z_uint8, cv2.COLORMAP_INFERNO)  # H×W×BGR
    result = np.clip(
        colored.astype(np.float32) * bright_2d[:, :, np.newaxis], 0, 255
    ).astype(np.uint8)
    rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # テキストオーバーレイ
    valid_pts = int(valid.sum())  # 有効セル数ではなく点数
    cv2.putText(
        rgb,
        f"fr {frame_idx}/{total_frames - 1}  {len(xs):,} pts  z:[{zs.min():.1f},{zs.max():.1f}]m",
        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        rgb,
        f"range +/-{max_range:.0f}m",
        (6, img_size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA,
    )
    _ = valid_pts  # suppress unused warning
    return rgb


def render_lidar_bev(
    bag_path: Path,
    frame_idx: int = 0,
    max_range: float = 30.0,
    img_size: int = 512,
) -> tuple[np.ndarray | None, int]:
    """ROS bag から LiDAR 点群を読み取り、鳥瞰図 (BEV) 画像と総フレーム数を返す。

    フレーム数は初回スキャン後にキャッシュされ、2 回目以降は bag_path を最大
    frame_idx まで走査するだけで済む。

    Returns:
        (image_array, total_frames) — 画像取得失敗時は (None, 0)
    """
    try:
        from rosbags.rosbag1 import Reader as Reader1
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        logger.warning("rosbags がインストールされていません")
        return None, 0

    cached_total = _lidar_frame_count_cache.get(bag_path)

    try:
        typestore = get_typestore(Stores.ROS1_NOETIC)

        with Reader1(bag_path) as reader:
            # LiDAR トピックを探す
            lidar_conns: list = []
            for topic, info in reader.topics.items():
                if topic in _LIDAR_TOPICS or "PointCloud2" in (info.msgtype or ""):
                    lidar_conns.extend(info.connections)

            if not lidar_conns:
                all_topics = {t: i.msgtype for t, i in reader.topics.items()}
                logger.warning("LiDAR トピックなし: %s  利用可能: %s", bag_path.name, all_topics)
                _lidar_frame_count_cache[bag_path] = 0
                return None, 0

            target_xs = target_ys = target_zs = np.array([])
            found = False
            total = 0

            for connection, _ts, rawdata in reader.messages(connections=lidar_conns):
                if total == frame_idx and not found:
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    target_xs, target_ys, target_zs = _extract_xyz_from_pc2(msg)
                    found = True
                    if cached_total is not None:
                        # フレーム数はキャッシュ済み → 以降スキャン不要
                        break
                total += 1

            if cached_total is None:
                _lidar_frame_count_cache[bag_path] = total
                cached_total = total

    except Exception as exc:
        logger.warning("LiDAR 読み込みエラー (ROS1): %s", exc)
        return None, _lidar_frame_count_cache.get(bag_path, 0)

    total_frames = cached_total or 0

    if len(target_xs) == 0:
        logger.info("LiDAR: frame_idx=%d に点群なし (total=%d)", frame_idx, total_frames)
        return None, total_frames

    # 有効点フィルタ（NaN・範囲外を除去）
    mask = (
        (np.abs(target_xs) < max_range)
        & (np.abs(target_ys) < max_range)
        & np.isfinite(target_xs)
        & np.isfinite(target_ys)
        & np.isfinite(target_zs)
    )
    xs_f, ys_f, zs_f = target_xs[mask], target_ys[mask], target_zs[mask]

    if len(xs_f) == 0:
        # 範囲外にすべての点がある場合 → max_range を実データに合わせて再計算
        actual_range = float(np.nanpercentile(np.abs(target_xs[np.isfinite(target_xs)]), 95)) if np.any(np.isfinite(target_xs)) else max_range
        logger.info("LiDAR: max_range=%.1f 内に点なし、実レンジ推定=%.1f m", max_range, actual_range)
        mask2 = (
            (np.abs(target_xs) < actual_range * 1.1)
            & (np.abs(target_ys) < actual_range * 1.1)
            & np.isfinite(target_xs)
            & np.isfinite(target_ys)
            & np.isfinite(target_zs)
        )
        xs_f, ys_f, zs_f = target_xs[mask2], target_ys[mask2], target_zs[mask2]
        max_range = actual_range * 1.1
        if len(xs_f) == 0:
            return None, total_frames

    img = _render_bev_from_points(xs_f, ys_f, zs_f, max_range, img_size, frame_idx, total_frames)
    return img, total_frames


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
            # Tab 2: セッション閲覧 (可視光 + 赤外線 + LiDAR + 音声)
            # ================================================================
            with gr.Tab("セッション閲覧 (可視光 + 赤外線 + LiDAR + 音声)"):
                gr.Markdown(
                    "点検セッションごとに **可視光カメラ**・**赤外線カメラ** の映像フレームと"
                    " **LiDAR 鳥瞰図 (BEV)**・**音声** を並べて閲覧します。\n\n"
                    f"> **データディレクトリ**: `{_EXTRACT_ROOT}`"
                )

                # --- モダリティ フィルタ チェックボックス ---
                ses_modality_filter = gr.CheckboxGroup(
                    choices=["可視光 (RGB)", "赤外線 (IR)", "音声", "LiDAR"],
                    value=["可視光 (RGB)", "赤外線 (IR)", "音声"],
                    label="モダリティ フィルタ（ONのデータが存在するセッションのみ表示）",
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
                        minimum=0, maximum=300, step=1, value=0, label="フレーム番号", scale=5
                    )
                    ses_next_btn = gr.Button("次へ ▶", scale=1)
                ses_frame_info = gr.Textbox(
                    label="フレーム情報 (可視光 / 赤外線 / LiDAR / 音声)", value="", interactive=False
                )

                with gr.Row():
                    ses_vis_img = gr.Image(label="可視光 (RGB)", type="numpy", scale=1)
                    ses_ir_img = gr.Image(label="赤外線 (Infrared)", type="numpy", scale=1)
                    ses_lidar_img = gr.Image(label="LiDAR 鳥瞰図 (BEV)", type="numpy", scale=1)

                with gr.Row():
                    ses_audio = gr.Audio(label="音声", type="filepath", scale=1)

                with gr.Row():
                    ses_sensor = gr.Markdown(label="環境センサデータ", value="(セッションを選択してください)")

                with gr.Row():
                    ses_load_btn = gr.Button("セッション一覧を読み込む (CSV 生成)", variant="primary")
                    ses_csv_status = gr.Textbox(
                        label="CSV ステータス", value="", interactive=False, scale=3
                    )

                # 動画の最大フレーム数をセッション間で保持するステート
                ses_frame_max = gr.State(value=300)

                # --- コールバック ---

                def _modality_flags(selected: list[str]) -> tuple[bool, bool, bool, bool]:
                    """CheckboxGroup の選択値をフラグに変換する。"""
                    return (
                        "可視光 (RGB)" in selected,
                        "赤外線 (IR)" in selected,
                        "音声" in selected,
                        "LiDAR" in selected,
                    )

                def _filtered_session_list(
                    split: str, selected_modalities: list[str]
                ) -> list[str]:
                    idx = _get_session_index(split)
                    sessions = idx.session_list()
                    csv_data = load_session_csv()
                    need_rgb, need_ir, need_audio, need_lidar = _modality_flags(selected_modalities)
                    return filter_sessions_by_modality(
                        split, sessions, csv_data, need_rgb, need_ir, need_audio, need_lidar
                    )

                def on_filter_change(split: str, selected_modalities: list[str]):
                    """チェックボックス / Split 変更時: セッション一覧を更新する。"""
                    sessions = _filtered_session_list(split, selected_modalities)
                    if sessions:
                        return gr.update(choices=sessions, value=sessions[0])
                    return gr.update(choices=[], value=None)

                def _video_frame_count(path: Path) -> int:
                    cap = cv2.VideoCapture(str(path))
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    return total

                def on_session_select(split: str, session: str | None, frame_idx: int):
                    blank = np.zeros((360, 640, 3), dtype=np.uint8)
                    lidar_blank = np.zeros((512, 512, 3), dtype=np.uint8)
                    if not session:
                        return blank, blank, lidar_blank, None, "(セッションを選択してください)", "", gr.update(maximum=300), 300

                    idx = _get_session_index(split)
                    cached_files = idx.extract_session(session)

                    vis_frame: np.ndarray | None = None
                    ir_frame: np.ndarray | None = None
                    lidar_img: np.ndarray | None = None
                    audio_path: str | None = None
                    sensor_md = "(センサデータなし)"

                    # フレーム数を取得して対応ズレを検出
                    vis_total = _video_frame_count(cached_files["visible"]) if "visible" in cached_files else 0
                    ir_total = _video_frame_count(cached_files["infrared"]) if "infrared" in cached_files else 0
                    max_frames = max(vis_total, ir_total, 1) - 1
                    frame_info = f"可視光: {vis_total} フレーム / 赤外線: {ir_total} フレーム"
                    if vis_total > 0 and ir_total > 0 and vis_total != ir_total:
                        frame_info += f"  ⚠️ フレーム数が異なります（差: {abs(vis_total - ir_total)} フレーム）"

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

                    # LiDAR bag
                    if "point_cloud" in cached_files:
                        try:
                            lidar_img, lidar_total = render_lidar_bev(
                                cached_files["point_cloud"], frame_idx
                            )
                            frame_info += f" / LiDAR: {lidar_total} フレーム"
                            if lidar_total > 0:
                                max_frames = max(max_frames, lidar_total - 1)
                        except Exception as e:
                            logger.warning("LiDAR フレーム読み込みエラー: %s", e)

                    # 音声 wav
                    if "audio" in cached_files:
                        audio_path = str(cached_files["audio"])
                        frame_info += " / 音声: あり"

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
                        lidar_img if lidar_img is not None else lidar_blank,
                        audio_path,
                        sensor_md,
                        frame_info,
                        gr.update(maximum=max_frames),
                        max_frames,
                    )

                _ses_outputs = [
                    ses_vis_img, ses_ir_img, ses_lidar_img, ses_audio,
                    ses_sensor, ses_frame_info, ses_frame_idx, ses_frame_max,
                ]

                # セッション一覧読み込み + CSV 生成
                def on_load_btn(split: str, selected_modalities: list[str]):
                    csv_msg = generate_session_csv()
                    sessions = _filtered_session_list(split, selected_modalities)
                    session_update = (
                        gr.update(choices=sessions, value=sessions[0])
                        if sessions
                        else gr.update(choices=[], value=None)
                    )
                    return session_update, csv_msg

                ses_load_btn.click(
                    on_load_btn,
                    inputs=[ses_split, ses_modality_filter],
                    outputs=[ses_session, ses_csv_status],
                ).then(
                    on_session_select,
                    inputs=[ses_split, ses_session, ses_frame_idx],
                    outputs=_ses_outputs,
                )

                # チェックボックス変更 → セッション一覧を再フィルタ
                ses_modality_filter.change(
                    on_filter_change,
                    inputs=[ses_split, ses_modality_filter],
                    outputs=ses_session,
                ).then(
                    on_session_select,
                    inputs=[ses_split, ses_session, ses_frame_idx],
                    outputs=_ses_outputs,
                )

                # Split 変更 → セッション一覧を再フィルタ
                ses_split.change(
                    on_filter_change,
                    inputs=[ses_split, ses_modality_filter],
                    outputs=ses_session,
                ).then(
                    on_session_select,
                    inputs=[ses_split, ses_session, ses_frame_idx],
                    outputs=_ses_outputs,
                )

                ses_session.change(
                    on_session_select,
                    inputs=[ses_split, ses_session, ses_frame_idx],
                    outputs=_ses_outputs,
                )
                ses_frame_idx.change(
                    on_session_select,
                    inputs=[ses_split, ses_session, ses_frame_idx],
                    outputs=_ses_outputs,
                )
                ses_prev_btn.click(
                    lambda idx: max(0, idx - 1),
                    inputs=ses_frame_idx,
                    outputs=ses_frame_idx,
                ).then(
                    on_session_select,
                    inputs=[ses_split, ses_session, ses_frame_idx],
                    outputs=_ses_outputs,
                )
                ses_next_btn.click(
                    lambda idx, fmax: min(idx + 1, fmax),
                    inputs=[ses_frame_idx, ses_frame_max],
                    outputs=ses_frame_idx,
                ).then(
                    on_session_select,
                    inputs=[ses_split, ses_session, ses_frame_idx],
                    outputs=_ses_outputs,
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
        allowed_paths=[str(_EXTRACT_ROOT)],
    )


if __name__ == "__main__":
    '''
    uv run python src/vis/viewer.py --port 7860 --share

    # 新しい tmux セッションを作成して起動
    tmux new-session -d -s gradio_viewer -c /home/team-005/work/tanaka_y/10_dataset_viewer \
    'uv run python src/vis/viewer.py --port 7860 --share'

    # 状態確認
    tmux ls

    # セッションにアタッチ（ログを見たいとき）
    tmux attach -t gradio_viewer

    # アタッチ中に「デタッチ」して抜ける
    Ctrl+b → d

    # セッションを停止したいとき
    tmux kill-session -t gradio_viewer

    '''    
    main()
