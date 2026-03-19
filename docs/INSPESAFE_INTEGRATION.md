# InspecSafe-V1 データセット統合ガイド

このガイドでは、**InspecSafe-V1 産業用マルチモーダルデータセット**を Safety View Agent に統合する方法を説明します。

## 概要

Safety View Agent は 2 つのデータ入力モードをサポートしています：

| モード | 用途 | 入力ソース |
|--------|------|-----------|
| **manual** | カスタム動画による評価 | `data/videos/` |
| **inspesafe** | InspecSafe-V1 産業用データセット | `../InspecSafe-V1/DATA_PATH/` |

`data.mode: inspesafe` に設定すると、セッションパスを指定するだけで、動画自動展開と音声ファイルの自動コピーが行われます。

---

## インストール手順

### ステップ 1: ディスク容量確認

InspecSafe-V1 は合計 **23GB（圧縮）→ 29GB（展開済み）** のデータセットです。

```bash
# ディスク空き容量確認（50GB 以上推奨）
df -h /home/{username}/work/

# 出力例:
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/sda1       500G  150G  350G  30% /
```

### ステップ 2: InspecSafe-V1 をダウンロード

```bash
# HuggingFace Hub からデータセットをダウンロード
# （認証不要 - 公開データセット）
cd /home/{username}/work

hf download Tetrabot2026/InspecSafe-V1 \
  --repo-type dataset \
  --local-dir ../../InspecSafe-V1
```

**ダウンロード完了後の構造:**

```
/home/{username}/work/
├── kanden-koi-voltmind-hackathon-2026/  ← Safety View Agent
└── InspecSafe-V1/
    ├── test.tar          (5.4 GB)
    ├── train.tar         (17 GB)
    └── README.md
```

### ステップ 3: tar ファイルを展開

```bash
cd ../InspecSafe-V1

# 方法 A: バックグラウンド並列実行（推奨・高速）
tar -xf test.tar > /tmp/test_extract.log 2>&1 &
tar -xf train.tar > /tmp/train_extract.log 2>&1 &
wait
echo "✅ 展開完了"

# または、方法 B: シンプル版（直列実行）
# tar -xf test.tar && tar -xf train.tar
```

**展開状況をリアルタイム監視:**

```bash
# 別ターミナルで実行（5秒ごとに更新）
watch -n 5 'du -sh /home/{username}/work/InspecSafe-V1/DATA_PATH/'
```

### ステップ 4: 展開完了の確認

```bash
# ファイル構造確認
ls -lh ../InspecSafe-V1/DATA_PATH/

# 出力例:
# drwxrwxr-x  test/
# drwxrwxr-x  train/

# データセット内容確認
find ../InspecSafe-V1/DATA_PATH -type f | wc -l
# → 約 50,000+ ファイル
```

---

## 使用方法

### 設定ファイルで inspesafe モードを有効化

**`configs/default.yaml`:**

```yaml
# データ入力モード
data:
  mode: "inspesafe"  # ← "manual" から "inspesafe" に変更

  # inspesafe モード: InspecSafe-V1 のセッションを指定
  inspesafe:
    dataset_path: "../InspecSafe-V1"
    session: "train/Other_modalities/58132919535743_20251118_session_1400_2#bowenguanshang-you"
    # 注: session は DATA_PATH/ 以降の相対パスを指定
```

### セッション ID の見つけ方

**InspecSafe-V1 のセッション構造:**

```
DATA_PATH/
├── train/
│   └── Other_modalities/
│       ├── 58132919535743_20251118_session_1400_2#bowenguanshang-you/
│       │   ├── *_visible_*.mp4      ← RGB 動画（フレーム展開）
│       │   ├── *_infrared_*.mp4     ← 赤外線動画（フレーム展開・新規）
│       │   ├── *_audio_*.wav        ← 音声ファイル（自動コピー）
│       │   └── ...
│       └── （その他のセッション）
│
└── test/
    └── Other_modalities/
        └── （テストセッション）
```

**セッション ID の確認:**

```bash
# 利用可能なセッション一覧
ls -lh ../InspecSafe-V1/DATA_PATH/train/Other_modalities/

# セッションをフィルタ（"bowenguanshang-you" を含む）
ls -d ../InspecSafe-V1/DATA_PATH/train/Other_modalities/*bowenguanshang-you*
```

### エージェント実行

```bash
# inspesafe モードで実行
python src/run.py
```

**実行フロー:**

```
configs/default.yaml で data.mode: "inspesafe" を読み込み
    ↓
prepare_observations_inspesafe() が自動実行
    ├─ 赤外線動画検出: *_infrared_*.mp4 を検索
    │   ├─ _process_infrared_inspesafe() で フレーム抽出
    │   └─ data/infrared_frames/ に自動展開（新規 Phase 12）
    │
    ├─ RGB 動画検出: *_visible_*.mp4 を検索
    │   └─ data/frames/ に自動展開
    │
    └─ 音声ファイル検出: *_audio_*.wav を検索
        └─ data/audio/ に自動コピー
    ↓
マルチモーダル統合（RGB + 赤外線 + 音声）
    ↓
通常の Safety View Agent パイプライン実行
```

**実行ログの例:**

```
[safety_view_agent] INFO: [inspesafe] 赤外線動画を検出（フレーム抽出開始）
[safety_view_agent] INFO: Extracted 8 infrared frames to data/infrared_frames/
[safety_view_agent] INFO: [inspesafe] 動画: ../InspecSafe-V1/DATA_PATH/train/Other_modalities/.../58132919535743_20251118_visible_1.mp4
[safety_view_agent] INFO: [inspesafe] 音声コピー: ../InspecSafe-V1/DATA_PATH/train/Other_modalities/.../58132919535743_20251118_audio_1.wav → data/audio/audio.wav
[safety_view_agent] INFO: [inspesafe] 30 フレーム展開完了
[safety_view_agent] INFO: Processing 30 frame image(s)
```

**出力ディレクトリ構造:**

```
data/
├── frames/                      # RGB フレーム（*_visible_*.mp4 から）
│   ├── frame_0.000s.jpg
│   ├── frame_1.005s.jpg
│   └── ...
├── infrared_frames/             # 赤外線フレーム（*_infrared_*.mp4 から・新規）
│   ├── frame_0.000s.jpg
│   ├── frame_1.000s.jpg
│   └── ...
├── audio/
│   └── audio.wav                # 音声ファイル（*_audio_*.wav から）
└── perception_results.json      # 分析結果（frames 配列）
```

---

## データセット内容

### モダリティ別ファイル形式

| モダリティ | ファイル形式 | 説明 |
|-----------|-----------|------|
| **RGB（可視光）** | `.mp4` / `.jpg` | RGB ビデオ・フレーム画像 |
| **赤外線** | `.mp4` | サーマル画像ビデオ |
| **オーディオ** | `.wav` | 音声記録（16 kHz, mono） |
| **点群** | `.bag` | LiDAR / 深度データ（ROS bag） |
| **センサー** | `.txt` | ガス濃度、温度、湿度ログ |
| **セグメンテーション** | `.json` / `.png` | ピクセルレベル物体マスク |

### ファイル命名規則

Safety View Agent の inspesafe モードは以下のパターンで自動検索します：

```
*_visible_*.mp4    → RGB 動画（フレーム展開対象）
*_infrared_*.mp4   → 赤外線動画（フレーム展開対象・新規 Phase 12）
*_audio_*.wav      → 音声ファイル（自動コピー対象）
```

例：
```
58132919535743_20251118_visible_1.mp4
58132919535743_20251118_visible_2.mp4
58132919535743_20251118_infrared_1.mp4     ← 新規対応
58132919535743_20251118_audio_1.wav
58132919535743_20251118_audio_2.wav
```

### セッション統計

| カテゴリ | 数量 |
|---------|-----|
| **訓練セッション** | 3,763 |
| **テストセッション** | 1,250 |
| **総ファイル数** | 50,000+ |
| **合計ディスク容量** | 29 GB（展開済み） |

---

## トラブルシューティング

### ❌ エラー: "セッションが見つかりません"

**原因:** セッションパスが誤っているか、展開が未完了

```bash
# セッションパスを確認
echo "../InspecSafe-V1/DATA_PATH/train/Other_modalities/58132919535743_20251118_session_1400_2#bowenguanshang-you"

# ディレクトリが存在するか確認
test -d "../InspecSafe-V1/DATA_PATH/train/Other_modalities/58132919535743_20251118_session_1400_2#bowenguanshang-you" && echo "✅ OK" || echo "❌ NG"

# 実際の名前をリストしてコピー
ls ../InspecSafe-V1/DATA_PATH/train/Other_modalities/ | head -5
```

### ❌ エラー: "RGB 動画が見つかりません"

**原因:** セッションに `*_visible_*.mp4` ファイルが存在しない

```bash
# セッション内のファイルを確認
ls -lh "../InspecSafe-V1/DATA_PATH/train/Other_modalities/{セッション名}/" | grep visible

# 出力例:
# -rw-rw-r-- 58132919535743_20251118_visible_1.mp4
# -rw-rw-r-- 58132919535743_20251118_visible_2.mp4
```

### ⚠️ 警告: "音声ファイルなし"

**原因:** セッションに `*_audio_*.wav` がない（処理は継続）

```bash
# 音声ファイルを確認
ls -lh "../InspecSafe-V1/DATA_PATH/train/Other_modalities/{セッション名}/" | grep audio

# 出力がなければ、このセッションは音声なしセッション
```

### ❌ ダウンロードが失敗する

```bash
# HuggingFace CLI がインストール済みか確認
hf --version

# キャッシュをクリアして再ダウンロード
rm -rf ~/.cache/huggingface/hub/Tetrabot2026*
hf download Tetrabot2026/InspecSafe-V1 --repo-type dataset --local-dir ../InspecSafe-V1
```

### ❌ 展開中にディスクフルエラー

**原因:** ディスク容量が不足

```bash
# 利用可能な容量を確認
df -h /home/{username}/work/

# 不要なファイルを削除（キャッシュなど）
rm -rf ~/.cache/
du -sh /home/{username}/work/* | sort -h
```

### ❌ ファイルアクセス権限エラー

```bash
# 所有権を確認
ls -ld ../InspecSafe-V1/

# 必要に応じて権限を修正
chmod -R u+rw ../InspecSafe-V1/
```

---

## 設定のベストプラクティス

### マルチセッション評価

異なるセッションを順序に評価する場合：

```yaml
# 設定 A: セッション 1
data:
  mode: "inspesafe"
  inspesafe:
    dataset_path: "../InspecSafe-V1"
    session: "train/Other_modalities/session_id_1"

# → python src/run.py
# → 結果は data/ に追記保存
```

```yaml
# 設定 B: セッション 2（同じコマンドで実行）
data:
  mode: "inspesafe"
  inspesafe:
    dataset_path: "../InspecSafe-V1"
    session: "train/Other_modalities/session_id_2"

# → python src/run.py
# → 結果は data/ に追記保存
```

**結果の確認:**

```bash
# 複数セッションの分析結果を確認（frames 配列形式）
python -c "
import json
with open('data/perception_results.json') as f:
    data = json.load(f)
    print(f'Total frames analyzed: {len(data.get(\"frames\", []))}')
    print(f'Sample frame keys: {list(data[\"frames\"][0].keys()) if data[\"frames\"] else []}')
"
```

### パフォーマンス最適化

```yaml
agent:
  max_steps: 10  # セッションの最初の 10 フレームのみ分析

video:
  fps: 0.5       # フレームレートを低下（0.5 fps = 2秒ごと）
  max_frames: 20 # 最大 20 フレーム
```

---

## 赤外線フレーム処理の詳細（Phase 12）

### 自動検出と処理フロー

InspecSafe-V1 セッションに `*_infrared_*.mp4` ファイルが存在する場合、以下の処理が自動実行されます：

1. **赤外線動画検出**: `prepare_observations_inspesafe()` が glob パターンで検索
2. **フレーム抽出**: `_process_infrared_inspesafe()` でフレーム展開
3. **タイムスタンプマップ生成**: RGB フレームと対応付け
4. **Observation に統合**: `infrared_image_path` フィールドに赤外線フレームパスを格納

### 出力形式

```python
# src/safety_agent/schema.py の Observation クラス
class Observation:
    obs_id: str                      # フレーム識別子
    image_path: str                  # RGB フレームパス
    infrared_image_path: Optional[str]  # 赤外線フレームパス（新規）
    prev_image_path: Optional[str]   # 前フレーム（光学フロー用）
    audio_path: Optional[str]        # 音声ファイルパス
    video_timestamp: Optional[float] # ビデオ内タイムスタンプ
```

### 赤外線フレーム活用例

```python
# depth_node などで赤外線フレームを活用可能
observation = agentState.observation  # Observation オブジェクト
if observation.infrared_image_path:
    # 赤外線フレーム分析
    ir_analyzer = InfraredImageAnalyzer(...)
    ir_result = ir_analyzer.analyze(observation.infrared_image_path)
```

## 次のステップ

- **設定チューニング**: `view_planning` パラメータを調整して視点選択の戦略を変更
- **赤外線分析統合**: `InfraredImageAnalyzer` を使用した温度異常検出
- **バッチ処理**: 複数セッションの自動評価スクリプトを作成

詳細は [CLAUDE.md](../CLAUDE.md) および [EXTENDING.md](EXTENDING.md) を参照してください。
