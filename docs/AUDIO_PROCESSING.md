# Audio Processing

このドキュメントは、InspecSafe-V1 データセットの音声に背景音・異常音を合成するスクリプトの使用方法を説明します。

## 概要

`audio_processing/audio_processing.py` は、InspecSafe-V1 の各セッションディレクトリをコピーし、元の音声ファイルに工場背景音と異常音（衝突音など）を重ね合わせて合成します。

### 目的

- 実データに近い音響環境を再現したトレーニングデータを生成する
- 背景ノイズ・異常音の重みを設定で調整可能
- バッチスクリプトとの整合性を保ちつつ、元のディレクトリ構造を維持したまま出力

### 出力ディレクトリ構造

元の `Other_modalities/` 内に `_audio` サフィックスを付けたセッションディレクトリを生成します：

```
test/Other_modalities/
├── 58132919535743_20251118_session_1400_16#refengweiguan-you/        ← 元のまま維持
└── 58132919535743_20251118_session_1400_16#refengweiguan-you_audio/  ← 合成済み音声
    ├── 16#refengweiguan-you_audio_2025111814.wav   ← 上書き（背景音+異常音を混合）
    ├── 16#refengweiguan-you_infrared_2025111814.mp4
    ├── 16#refengweiguan-you_visible_2025111814.mp4
    └── ...
```

## セットアップ

### 1. 依存関係をインストール

```bash
uv sync --extra dev
```

関連パッケージ（`dev` extra に含まれる）：

- `librosa` - 音声ファイルの読み込み・リサンプリング
- `soundfile` - 音声ファイルの書き出し
- `numpy` - 波形の数値計算
- `pyyaml` - 設定ファイル読み込み

### 2. 素材ファイルを配置

背景音・異常音の素材ファイルを `data/audio_materials/` に配置します：

```
data/audio_materials/
├── factory.wav   ← 工場背景音
└── crash.wav     ← 異常音（衝突音など）
```

### 3. InspecSafe-V1 データセット

InspecSafe-V1 データセットは `../InspecSafe-V1/` に配置されている必要があります（プロジェクトルートから見て1つ上の `work/` ディレクトリ）：

```
/home/tetsutani/work/
├── InspecSafe-V1/
│   └── DATA_PATH/test/Other_modalities/
└── kanden-koi-voltmind-hackathon-2026/   ← プロジェクトルート
```

## 設定

すべての設定は `audio_processing/audio_processing.yaml` で管理します。

### 入力設定

```yaml
original:
  dir: "../InspecSafe-V1/DATA_PATH/test/Other_modalities"  # 入力ディレクトリ（プロジェクトルート基準）
  session_names: [                                           # 処理対象セッション名
      "58132919535743_20251118_session_1400_16#refengweiguan-you",
      ...
  ]
  weight: 0.5   # 元の音声の音量倍率
```

### 素材設定

```yaml
material:
  dir: "data/audio_materials"  # 素材ディレクトリ（プロジェクトルート基準）
  background:
    file_name: "factory.wav"   # 背景音ファイル名
    weight: 0.7                # 背景音の音量倍率
  anomaly:
    file_name: "crash.wav"     # 異常音ファイル名
    weight: 1.0                # 異常音の音量倍率
```

### 音量バランスの調整

3 つの `weight` を調整することで合成音声のバランスを変えられます：

| 設定キー | 初期値 | 効果 |
|----------|--------|------|
| `original.weight` | 0.5 | 元の音声を下げて素材音を際立たせる |
| `background.weight` | 0.7 | 工場の常時ノイズ感を調整 |
| `anomaly.weight` | 1.0 | 異常音を最も強調 |

## 使用方法

### 基本的な実行

**プロジェクトルートから実行すること**（config の相対パスがプロジェクトルート基準のため）

```bash
python audio_processing/audio_processing.py
```

または uv 経由:

```bash
uv run python audio_processing/audio_processing.py
```

### カスタム設定ファイルを指定

```bash
uv run python audio_processing/audio_processing.py --config path/to/custom.yaml
```

※ `--config` 引数は `main()` の `config_path` パラメータに対応します。

### Python から呼び出す

```python
from audio_processing.audio_processing import main

main(config_path="audio_processing/audio_processing.yaml")
```

## 出力

処理が完了すると、指定した各セッションに対して `{session_name}_audio/` ディレクトリが作成されます。

- 元のセッションディレクトリは **上書きされません**（`shutil.copytree` でコピー後に wav のみ上書き）
- 動画・センサーデータ等のファイルはコピーされますが変更はありません
- `.wav` ファイルのみ合成済みデータで上書きされます

## 合成ロジック

異常音（crash.wav）は末尾 6 秒を切り出し、元の音声長に合わせて前後にゼロパディングしてから重ね合わせます：

```
元の音声:    [========================] 全長
異常音:      [000000][crash_6sec][0000] ← センタリング配置
背景音:      [=====================...] 先頭から元の音声長ぶん使用
```

最終的な合成式：

```
合成音 = 元の音声 × 0.5 + 背景音 × 0.7 + 異常音 × 1.0
```

## トラブルシューティング

### エラー: "Config not found"

原因: スクリプトをプロジェクトルート以外から実行している

解決策: プロジェクトルートから実行する

```bash
cd /home/tetsutani/work/kanden-koi-voltmind-hackathon-2026
python audio_processing/audio_processing.py
```

### エラー: "list index out of range" (wav ファイル取得時)

原因: セッションディレクトリ内に `.wav` ファイルが存在しない

確認: セッションディレクトリの内容を確認

```bash
ls ../InspecSafe-V1/DATA_PATH/test/Other_modalities/{session_name}/
```

### エラー: "No such file or directory" (素材ファイル)

原因: `data/audio_materials/` に素材ファイルが未配置

解決策: `factory.wav` と `crash.wav` を `data/audio_materials/` に配置してから再実行

### エラー: "ModuleNotFoundError: librosa"

解決策:

```bash
uv sync --extra dev
```

または個別インストール:

```bash
uv pip install librosa soundfile
```
