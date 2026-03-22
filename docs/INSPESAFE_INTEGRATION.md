# InspecSafe-V1 Integration

Safety View Agent は `data.mode: "inspesafe"` で InspecSafe-V1 のセッションを直接処理できます。
このモードでは、RGB / 赤外線 / 音声を `run.py` が自動的に展開して `Observation` を構築します。

## 前提

- InspecSafe-V1 がローカルに展開済み
- `configs/default.yaml` の `data.inspesafe.dataset_path` が実在パス
- 対象セッションに `*_visible_*.mp4` がある

## 設定

```yaml
data:
  mode: "inspesafe"
  inspesafe:
    dataset_path: "/path/to/InspecSafe-V1"
    session: "train/Other_modalities/your_session_dir"
```

### `session` について

`session` は **`DATA_PATH/` 以降の相対パス** を指定します。

例:

```text
InspecSafe-V1/
└── DATA_PATH/
    └── train/
        └── Other_modalities/
            └── 58132919535743_20251118_session_1400_2#bowenguanshang-you/
```

このときの `session` は:

```text
train/Other_modalities/58132919535743_20251118_session_1400_2#bowenguanshang-you
```

## `run.py` が行うこと

`prepare_observations_inspesafe()` は対象セッションを解決して、次を自動実行します。

1. `*_infrared_*.mp4` を探す
   - 見つかれば `data/infrared_frames/` へ展開
2. `*_visible_*.mp4` を探す
   - `data/frames/` へ展開
3. `*_audio_*.wav` を探す
   - `data/audio/audio.wav` へコピー
4. Demo UI 用に `data/videos/video.mp4` を生成
   - 音声があれば ffmpeg で mux
   - 音声が無ければ RGB 動画をそのままコピー
5. 各フレームから `Observation` を作る

## 期待するファイル構成

最低限、セッションディレクトリには次のいずれかが必要です。

- `*_visible_*.mp4` 必須
- `*_infrared_*.mp4` 任意
- `*_audio_*.wav` 任意

## 実行

```bash
uv run python src/run.py
```

## 生成される主な出力

```text
data/
├── frames/
├── infrared_frames/
├── audio/audio.wav
├── videos/video.mp4
├── perception_results/
├── depth/            # enable_depth=true の場合
├── sam3_masks/       # agent.enable_sam3=true の場合
└── voice/            # tts.enabled=true の場合
```

## セッション確認のヒント

```bash
ls /path/to/InspecSafe-V1/DATA_PATH/train/Other_modalities
```

もしくは Linux / macOS なら:

```bash
find /path/to/InspecSafe-V1/DATA_PATH -type d -path "*/Other_modalities/*"
```

## よくある問題

### `セッションが見つかりません`

- `dataset_path` が `InspecSafe-V1` ルートを指しているか確認
- `session` が `DATA_PATH/` 以降の相対パスになっているか確認

### RGB 動画が見つからない

- `*_visible_*.mp4` が無いと処理できません

### 音声付き `video.mp4` ができない

- `ffmpeg` が無い可能性があります
- その場合でも RGB 動画の直接コピーにフォールバックします

### 赤外線が表示されない

- `*_infrared_*.mp4` が無いか、展開結果が `data/infrared_frames/` に無い可能性があります

## manual モードに戻す

InspecSafe を使わないときは戻し忘れに注意してください。

```yaml
data:
  mode: "manual"
```

`data.mode: "inspesafe"` のままだと、ローカル動画を置いてもまず dataset 側を探しに行きます。
