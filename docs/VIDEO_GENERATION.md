# Video Generation with LTX-2

このドキュメントは、LTX-2 を用いた動画生成スクリプトの使用方法を説明します。

## 概要

`video_generation/generate.py` は、diffusers ライブラリの `LTX2Pipeline` を使用して、テキストプロンプトから高品質な動画を生成します。

### 環境要件

- **GPU**: NVIDIA H100 以上推奨（VRAM 80GB+）
- **CUDA**: 12.4 以上（Driver 550.163.01 で確認済）
- **Python**: 3.12
- **パッケージマネージャ**: uv

### 特徴

- **CUDA 12.4 対応**: 公式 LTX-2 の `ltx-pipelines` が CUDA > 12.7 を要求するため、diffusers 経由の `LTX2Pipeline` を採用
- **設定ベース**: `configs/gen_prompt.yaml` で全パラメータを指定
- **バッチ処理**: 複数のプロンプトを順序実行可能
- **フレーム保存**: オプションで個別フレームを PNG として出力可能

## セットアップ

### 1. 依存関係をインストール

```bash
uv sync
```

新しく追加された依存パッケージ:
- `diffusers>=0.33.0` - LTX2Pipeline
- `transformers>=4.51.0` - テキスト処理
- `accelerate>=1.6.0` - GPU メモリ管理
- `imageio>=2.36.0`, `imageio-ffmpeg>=0.6.0` - MP4 書き出し
- `sentencepiece>=0.2.0` - トークナイザー

### 2. HuggingFace モデルのダウンロード（初回のみ）

```bash
# HuggingFace CLI を使用（H100には/home/team-005/data/ltx-2-modelsに配置済）
huggingface-cli download Lightricks/LTX-2 --local-dir ./ltx-2-models

# または Python から
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Lightricks/LTX-2', local_dir='./ltx-2-models', repo_type='model')
"
```

ダウンロードサイズ: 約 50GB

オフライン実行する場合は、`configs/gen_prompt.yaml` の `model.local_path` をダウンロード先に設定:

```yaml
model:
  repo_id: "Lightricks/LTX-2"
  local_path: "/path/to/ltx-2-models"  # ← ここに指定
```

## 設定

すべての設定は `configs/gen_prompt.yaml` で管理します。

### モデル設定

```yaml
model:
  repo_id: "Lightricks/LTX-2"        # HuggingFace リポジトリ ID
  local_path: null                    # ローカルキャッシュパス（優先度高）
  dtype: "bfloat16"                  # torch dtype (H100 推奨)
  device: "cuda"                      # デバイス
  enable_model_cpu_offload: false     # CPU オフロード（遅い）
  enable_vae_slicing: true            # VAE スライシング（VRAM削減）
  enable_vae_tiling: false            # VAE タイリング
```

### 生成パラメータ

```yaml
generation:
  width: 768                          # 画像幅（32の倍数）
  height: 512                         # 画像高さ（32の倍数）
  num_frames: 121                     # フレーム数 (8k+1形式: 9,17,25,33,...)
  num_inference_steps: 40             # 拡散ステップ (20-50推奨)
  guidance_scale: 3.0                 # 文本ガイダンス強度
  seed: 42                            # 乱数シード (null=ランダム)
  num_videos_per_prompt: 1            # プロンプト当たりの動画数
  generate_audio: false               # 音声生成（実験的）
```

### プロンプト設定

```yaml
prompts:
  negative_prompt: "worst quality, ..."  # 負のプロンプト
  items:
    - id: "factory_floor_normal"
      prompt: "..."                      # 生成内容の説明
      overrides:                         # generation設定の上書き
        guidance_scale: 4.0
        seed: 100
```

### 出力設定

```yaml
output:
  dir: "data/videos"                      # 出力ディレクトリ
  filename_template: "{id}_{seed}_{timestamp}.mp4"  # ファイル名テンプレート
  fps: 24                                 # フレームレート
  codec: "libx264"                        # ビデオコーデック
  quality: 8                              # 品質 (0-10)
  save_frames: false                      # フレーム個別保存
```

## 使用方法

### 基本的な実行

```bash
# すべてのプロンプトで動画生成
uv run python video_generation/generate.py
```

### ドライラン（設定検証のみ）

```bash
# GPU 計算なしで設定をパース・表示
uv run python video_generation/generate.py --dry-run
```

出力例:
```
2026-03-20 15:38:47 - video_generation - INFO - Video generation plan: 3 prompt(s)
2026-03-20 15:38:47 - video_generation - INFO -   - [factory_floor_normal] A factory floor with workers wearing...
2026-03-20 15:38:47 - video_generation - INFO -   - [hazard_detection_scenario] Industrial plant corridor with...
2026-03-20 15:38:47 - video_generation - INFO -   - [overhead_crane_operation] Overhead crane moving a heavy...
2026-03-20 15:38:47 - video_generation - INFO - Dry run complete. Exiting without inference.
```

### 特定のプロンプトのみ生成

```bash
# ID で指定したプロンプトのみ実行
uv run python video_generation/generate.py --prompt-id factory_floor_normal
```

### カスタム設定ファイル

```bash
# 異なる設定ファイルを使用
uv run python video_generation/generate.py --config configs/gen_prompt_custom.yaml
```

## 出力

### ファイル構造

生成されたファイルは `data/videos/` に保存されます:

```
data/videos/
├── factory_floor_normal_42_20260320_150000.mp4
├── hazard_detection_scenario_42_20260320_150005.mp4
└── overhead_crane_operation_100_20260320_150010.mp4
```

### ファイル仕様

- **形式**: MP4 (H.264 コーデック)
- **フレームレート**: 24 FPS
- **解像度**: 768×512 px
- **フレーム数**: 121 フレーム
- **ファイルサイズ**: 約 300-500 MB (品質設定に依存)

## パフォーマンス

### 推奨設定

H100 での実行時間目安 (単一プロンプト):

| パラメータ | 実行時間 | VRAM使用量 |
|---|---|---|
| 40 steps, 768x512, 121 frames | ~3-5 分 | ~50GB |
| 30 steps, 768x512, 81 frames | ~2-3 分 | ~40GB |
| 20 steps, 512x384, 121 frames | ~1-2 分 | ~30GB |

### メモリ削減

VRAM が不足する場合:

```yaml
generation:
  num_inference_steps: 30  # ステップ数を減らす
  num_frames: 81           # フレーム数を減らす

model:
  enable_model_cpu_offload: true  # CPU オフロード (速度低下)
  enable_vae_slicing: true        # VAE スライシング
  enable_vae_tiling: true         # VAE タイリング
```

## トラブルシューティング

### エラー: "CUDA out of memory"

解決策:
1. `num_frames` を 81 に減らす
2. `num_inference_steps` を 30 に減らす
3. 解像度を 512×384 に下げる
4. `enable_model_cpu_offload: true` を設定
5. `enable_vae_tiling: true` を設定

### エラー: "num_frames is invalid"

原因: `(num_frames - 1) % 8 != 0`

LTX-2 は 8k+1 フレーム形式を要求します。有効な値:
- 9, 17, 25, 33, 49, 65, 81, 97, 121, 161, 201

### エラー: "HuggingFace model not found"

原因: モデルファイルがダウンロードされていない

解決策:
```bash
huggingface-cli download Lightricks/LTX-2
```

または `configs/gen_prompt.yaml` の `model.local_path` を確認。

## API リファレンス

### `load_gen_config(config_path)`

YAML 設定ファイルをロードして検証

```python
from video_generation.generate import load_gen_config
cfg = load_gen_config("configs/gen_prompt.yaml")
```

### `build_pipeline(cfg)`

LTX2Pipeline を初期化

```python
from video_generation.generate import build_pipeline
pipe = build_pipeline(cfg)
```

### `generate_video(pipe, cfg, prompt_item)`

単一プロンプトから動画を生成

```python
from video_generation.generate import generate_video
paths = generate_video(pipe, cfg, {"id": "test", "prompt": "..."})
```

## 参考資料

- LTX-2 GitHub: https://github.com/Lightricks/LTX-2
- diffusers 文書: https://huggingface.co/docs/diffusers/
- HuggingFace Hub: https://huggingface.co/Lightricks/LTX-2
