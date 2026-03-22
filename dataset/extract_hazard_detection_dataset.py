"""Extract anomaly sample frames from training_mixed_set.zip.

Selects 10 anomaly classes, picks the first contiguous run of each,
and extracts all frames to:
  /home/team-005/data/hazard-detection/dataset/anomaly_samples/<class_name>/
"""

import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET_DIR = Path("/home/team-005/data/hazard-detection/dataset")
METADATA_DIR = DATASET_DIR / "metadata"
ZIP_PATH = DATASET_DIR / "training_mixed_set.zip"
OUTPUT_DIR = DATASET_DIR / "anomaly_samples"

LABELS_FILE = METADATA_DIR / "labels_mapping.txt"
FRAMES_LABELS_CSV = METADATA_DIR / "training_mixed_frames_labels.csv"

# ── All 20 anomaly classes to extract (label_id: class_name) ──────────────
SELECTED_CLASSES = {
    1:  "box",           # first run: frames 0–2117      (2118 frames)
    2:  "cable",         # first run: frames 8201–10637  (2437 frames)
    3:  "cones",         # first run: frames 32800–34521 (1722 frames)
    4:  "debris",        # first run: frames 5278–7022   (1745 frames)
    5:  "defects",       # first run: frames 42389–49488 (7100 frames)
    6:  "door",          # only run:  frames 7494–8200   ( 707 frames)
    7:  "floor",         # first run: frames 36062–37895 (1834 frames)
    8:  "human",         # first run: frames 49489–54549 (5061 frames)
    9:  "misc",          # only run:  frames 10638–11234 ( 597 frames)
    10: "tape",          # first run: frames 11235–13723 (2489 frames)
    11: "trolley",       # first run: frames 13724–15566 (1843 frames)
    12: "clutter",       # only run:  frames 18576–19234 ( 659 frames)
    13: "foam",          # only run:  frames 23749–23922 ( 174 frames)
    14: "sawdust",       # only run:  frames 24077–24329 ( 253 frames)
    15: "shard",         # only run:  frames 24330–24368 (  39 frames)
    16: "cellophane",    # only run:  frames 32428–32799 ( 372 frames)
    17: "screws",        # only run:  frames 38118–38286 ( 169 frames)
    18: "water",         # only run:  frames 41089–41299 ( 211 frames)
    19: "obj_on_robot",  # first run: frames 85315–90565 (5251 frames)
    20: "obj_on_robot2", # first run: frames 58791–64161 (5371 frames)
}


def find_first_run(
    frames_labels_csv: Path, target_label: int
) -> tuple[int, int]:
    """Return (start_frame_id, end_frame_id) of the first contiguous run
    for target_label in the CSV."""
    in_run = False
    run_start = -1
    prev_fid = -1

    with open(frames_labels_csv) as f:
        for row in csv.DictReader(f):
            fid = int(row["frame_id"])
            lbl = int(row["label"])
            if lbl == target_label:
                if not in_run:
                    run_start = fid
                    in_run = True
            else:
                if in_run:
                    return run_start, prev_fid
            prev_fid = fid

    if in_run:
        return run_start, prev_fid
    raise ValueError(f"Label {target_label} not found in {frames_labels_csv}")


def frame_id_to_zip_name(frame_id: int) -> str:
    """Convert frame_id integer to filename inside the zip."""
    return f"unlabeled_set/{frame_id:06d}_512_512.jpg"


def extract_samples(dry_run: bool = False) -> None:
    # Validate inputs
    assert ZIP_PATH.exists(), f"Zip not found: {ZIP_PATH}"
    assert FRAMES_LABELS_CSV.exists(), f"CSV not found: {FRAMES_LABELS_CSV}"

    label_map = json.loads(LABELS_FILE.read_text())
    id_to_label = {v: k for k, v in label_map.items()}

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Source zip: {ZIP_PATH}\n")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zip_names = set(zf.namelist())

        for label_id, class_name in sorted(SELECTED_CLASSES.items()):
            start, end = find_first_run(FRAMES_LABELS_CSV, label_id)
            frame_ids = range(start, end + 1)
            out_dir = OUTPUT_DIR / class_name
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"[{label_id:2d}] {class_name:15s}  frames {start:6d}–{end:6d}"
                f"  ({len(frame_ids):4d} frames)  -> {out_dir}"
            )

            missing = 0
            for fid in frame_ids:
                zip_name = frame_id_to_zip_name(fid)
                if zip_name not in zip_names:
                    missing += 1
                    continue
                if dry_run:
                    continue
                dest = out_dir / f"{fid:06d}.jpg"
                dest.write_bytes(zf.read(zip_name))

            if missing:
                print(f"    WARNING: {missing} frames missing in zip")

    print("\nDone.")


if __name__ == "__main__":
    extract_samples(dry_run=False)
