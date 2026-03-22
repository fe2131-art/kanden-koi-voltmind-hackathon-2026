# React デモアプリ - VLM Overlay Demo

Safety View Agent のリアルタイム UI デモアプリケーション。
WebSocket を使用して、バックエンド推論結果をブラウザにストリーミング表示します。

## 概要

**VLM Overlay Demo** は、動画再生中にリアルタイムで BBox（バウンディングボックス）、
安全性 Assessment、深度マップを表示するデモアプリです。

### 機能

- 動画再生（HTML5 video コントロール）
- BBox オーバーレイ描画（canvas、severity カラーコーディング）
- Assessment パネル（risk_level / action_type / temporal_status / hazards）
- 深度マップパネル（Depth Map）
- シーン説明パネル（collapsible）
- WebSocket 自動接続・自動再接続（3 秒インターバル）
- 動画タイムスタンプ同期（Sync / Latest モード）
- ログパネル（受信フレーム表示）

### 技術スタック

| コンポーネント | 技術 | ポート |
|--------------|------|--------|
| フロントエンド | React 18 + Vite 5 | 5173 |
| WebSocket サーバー | Python asyncio + websockets | 8001 |
| 静的アセット配信 | Vite `publicDir: data/` | 5173 |

---

## アーキテクチャ

```
┌─────────────────────────────────────┐
│  ブラウザ  http://localhost:5173      │
│                                     │
│  App.jsx                            │
│  ├─ <video src="/videos/video.mp4"> │  ← data/videos/video.mp4 をライブ配信
│  ├─ <canvas>  BBox オーバーレイ      │
│  ├─ Assessment / Depth / Scene パネル│
│  └─ WebSocket クライアント           │
└─────────────┬───────────────────────┘
              │ ws://127.0.0.1:8001 (直接接続)
              ▼
┌─────────────────────────────────────┐
│  server.py  (Port 8001)             │
│                                     │
│  manifest.json を 0.1 秒ごとにポーリング
│  frame_count が増加したら              │
│  → 新フレームの JSON を読み込み        │
│  → normalize_critical_point() 変換   │
│  → depth/voice ファイルの存在確認     │
│  → WebSocket で送信                  │
└─────────────┬───────────────────────┘
              │ ファイル読み込み
              ▼
┌─────────────────────────────────────┐
│  data/perception_results/           │
│  ├─ manifest.json  ← frame_count    │
│  └─ frames/                         │
│     ├─ 000000_img_0.json            │
│     ├─ 000001_img_1.json            │
│     └─ ...                          │
│                                     │
│  data/frames/       ← RGB 静止画    │
│  data/depth/        ← 深度マップ画像 │
│  data/voice/        ← 音声ファイル   │
└─────────────┬───────────────────────┘
              ↑ アトミック書き込み
┌─────────────────────────────────────┐
│  python src/run.py                  │
│  Safety View Agent                  │
│  (LangGraph パイプライン)             │
└─────────────────────────────────────┘
```

---

## ファイル構成

```
src/apps/
├── App.jsx          # React メインコンポーネント
├── main.jsx         # React エントリーポイント
├── index.html       # HTML テンプレート
├── vite.config.js   # Vite 設定（publicDir: data/）
├── package.json     # npm 依存
├── server.py        # WebSocket サーバー（manifest 監視・ストリーミング）
└── dist/            # npm run build で生成（gitignore）
```

---

## 前提条件

```bash
# Python バージョン確認
python3 --version   # 3.12 推奨

# Node.js バージョン確認
node --version      # v18.0.0 以上

# 必須ポート
# 5173  Vite dev server
# 8001  WebSocket server
```

---

## セットアップ（初回のみ）

### Python 環境

```bash
cd /path/to/kanden-koi-voltmind-hackathon-2026

uv sync --extra dev
```

### npm 依存関係

```bash
cd src/apps
npm install
```

### データディレクトリ確認

```bash
# プロジェクトルートで確認
ls data/videos/              # video.mp4 が必要
ls data/perception_results/  # manifest.json + frames/ が必要
```

---

## デモ起動手順

3 つのターミナルを使って起動します。

### ターミナル A ─ WebSocket サーバー

```bash
cd /path/to/kanden-koi-voltmind-hackathon-2026
python src/apps/server.py
```

期待される出力：
```
2026-03-20 10:00:00 apps.server INFO ws server: ws://localhost:8001
2026-03-20 10:00:00 apps.server INFO monitoring: data/perception_results/manifest.json
```

### ターミナル B ─ Vite dev server（フロントエンド）

```bash
cd src/apps
npm run dev
```

期待される出力：
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### ターミナル C ─ Safety View Agent（推論）

```bash
cd /path/to/kanden-koi-voltmind-hackathon-2026
python src/run.py
```

`data/perception_results/frames/` に結果が逐次書き込まれ、
ブラウザがリアルタイムで受信します。

### ブラウザで確認

```
http://localhost:5173
```

ページを開くと：
1. ヘッダーの **● LIVE** バッジが緑色になれば WebSocket 接続完了
2. 動画エリアで再生ボタンをクリック
3. Agent が推論するたびに Assessment・BBox が更新される

---

## 表示モード

### Sync モード（推奨）

動画の現在時刻に合わせて BBox を同期表示します。

```
表示対象 = video.currentTime - D(秒) 以前の最新フレーム
```

`D`（Delay）は Status パネルで調整可能：

| 処理構成 | 推奨 D 値 |
|---------|---------|
| VLM のみ | 0.3 〜 0.7 s |
| VLM + 深度 | 0.5 〜 1.0 s |
| 全モダリティ | 1.0 〜 2.0 s |

### Latest モード

常に最新受信フレームの結果を表示します。同期遅延の調整中に便利です。

---

## WebSocket メッセージ形式

server.py が送信する JSON の構造です。

```json
{
  "t": 2.0,
  "video_timestamp": 2.0,
  "text": "配管とバルブが密集した機械設備の視野。",
  "frame_id": "img_2",
  "assessment": {
    "risk_level": "high",
    "safety_status": "現場には高温の領域と腐食の兆候が確認されます。",
    "detected_hazards": ["縦型パイプの腐食兆候", "高温発熱領域", "死角の存在"],
    "action_type": "inspect_region",
    "target_region": "critical_point_0",
    "reason": "vision で腐食兆候を検出、infrared で高温領域を検出。",
    "priority": 0.8,
    "temporal_status": "new",
    "confidence_score": 0.75
  },
  "critical_points": [
    {
      "region_id": "critical_point_0",
      "description": "縦型パイプ表面に腐食の兆候および接続部周辺の劣化の可能性",
      "severity": "unknown",
      "bbox": [0.4, 0.25, 0.62, 0.75]
    }
  ],
  "scene_description": "配管とバルブが密集した機械設備の視野。",
  "depth_image_path": "/depth/frame_2.0s.jpg",
  "voice_path": null
}
```

### フィールド説明

| フィールド | 型 | 説明 |
|-----------|---|------|
| `t` | number | フレームタイムスタンプ（秒）。`video_timestamp` がある場合はその値 |
| `video_timestamp` | number \| null | 動画内タイムスタンプ（秒）。`frame_{t:.1f}s.jpg` と対応 |
| `text` | string | シーン説明テキスト |
| `frame_id` | string | フレーム識別子（例: `img_0`, `img_3`） |
| `assessment` | object \| null | 安全性評価（下表参照） |
| `critical_points` | array | 注意点リスト（正規化 bbox 付き） |
| `scene_description` | string | シーン詳細説明 |
| `depth_image_path` | string \| null | 深度マップ画像パス（`/depth/frame_{t:.1f}s.jpg`） |
| `voice_path` | string \| null | 対応音声ファイルパス（`/voice/frame_{t:.1f}s.wav`） |

### assessment フィールド

| フィールド | 値の例 |
|-----------|-------|
| `risk_level` | `"low"` / `"medium"` / `"high"` / `"critical"` |
| `action_type` | `"monitor"` / `"inspect_region"` / `"mitigate"` / `"emergency_stop"` |
| `temporal_status` | `"new"` / `"persistent"` / `"worsening"` / `"improving"` / `"resolved"` / `"unknown"` |
| `priority` | 0.0 〜 1.0 |
| `confidence_score` | 0.0 〜 1.0 |

#### action_type の意味

LLMが判断した「次に取るべき行動」です。`configs/prompt.yaml` で4択に制約することで出力の一貫性を保っています。

| 値 | 意味 | 想定リスクレベル |
|---|------|----------------|
| `emergency_stop` | **緊急停止** — 即座に作業・機械を止める必要がある重大な危険を検出 | critical / high |
| `inspect_region` | **現場確認** — 特定の領域を目視・物理的に詳しく確認すべき状況 | high / medium |
| `mitigate` | **リスク軽減** — 停止はしないが、対策（立入禁止・警告など）が必要 | medium |
| `monitor` | **継続監視** — 現時点で行動は不要だが、引き続き注意して観察する | low / medium |

#### temporal_status の意味

ハザードの**時間的な変化状態**を表します。LLM は `belief_state`（前フレームまでの世界モデル）を参照してこの値を決定します。`persistent` は「危険が長時間続いている」ことを示し、`new` より優先度を上げる判断根拠になります。

| 値 | 意味 |
|---|------|
| `new` | **新規検出** — 今回初めて現れたハザード |
| `persistent` | **継続中** — 前回からずっと存在し続けているハザード |
| `worsening` | **悪化中** — 前回より状況が悪くなっている |
| `improving` | **改善中** — 前回より状況が良くなっている |
| `resolved` | **解消** — 以前あったハザードが消えた |
| `unknown` | **不明** — 判断できない（初回フレームや情報不足など） |

### critical_points の bbox

```
bbox: [x_min, y_min, x_max, y_max]  (正規化座標 0.0 〜 1.0)

例: [0.4, 0.25, 0.62, 0.75]
  → 動画幅の 40〜62%、高さの 25〜75% の領域
```

---

## 静的アセット配信の仕組み

`vite.config.js` の `publicDir` を `data/` に設定しているため、
`data/` 配下のファイルが URL にマッピングされます。

```
data/videos/video.mp4       → http://localhost:5173/videos/video.mp4
data/depth/frame_0.0s.jpg   → http://localhost:5173/depth/frame_0.0s.jpg
data/voice/frame_0.0s.wav   → http://localhost:5173/voice/frame_0.0s.wav
```

> **注意**: `npm run preview`（`dist/` ベース）では、ビルド時点のデータの
> スナップショットが配信されます。ライブデモには `npm run dev` を使用してください。

---

## トラブルシューティング

### WebSocket が接続できない（● OFFLINE 表示）

```bash
# サーバーが起動しているか確認
ps aux | grep "server.py"

# ポート 8001 がリッスン中か確認
lsof -i :8001

# サーバーを起動
python src/apps/server.py
```

### 動画が再生されない（黒画面）

```bash
# 動画ファイルの存在確認
ls -lh data/videos/video.mp4

# Vite publicDir 設定の確認
grep publicDir src/apps/vite.config.js
```

App.jsx の `src="/videos/video.mp4"` は `data/videos/video.mp4` にマッピングされます。
ファイル名が異なる場合は `App.jsx:484` の `src` を変更してください。

### BBox が表示されない

```bash
# manifest.json のフレーム数を確認
cat data/perception_results/manifest.json

# frames/ ディレクトリを確認
ls data/perception_results/frames/ | head

# python src/run.py が実行されているか確認
# → ターミナル C を確認
```

Sync モードで出ない場合は Mode を **Latest** に切り替えて確認してください。
出れば Delay（D）値の調整で解消します。

### 深度マップが表示されない

```bash
# depth/ ディレクトリにファイルがあるか確認
ls data/depth/

# ファイル名フォーマット確認（.1f 形式が正しい）
# 例: frame_0.0s.jpg, frame_1.0s.jpg
```

server.py は `frame_{video_ts:.1f}s.jpg` でパスを生成します。
`data/depth/` のファイル名と一致している必要があります。

### ポート競合

```bash
# ポート 5173 が使用中
lsof -i :5173
kill -9 <PID>
# または別ポートで起動
npm run dev -- --port 5174

# ポート 8001 が使用中
lsof -i :8001
kill -9 <PID>
```

---

## ビルド・本番配信

```bash
cd src/apps

# ビルド（dist/ に出力、data/ のスナップショットも同梱）
npm run build

# ビルド結果のプレビュー（静的サーバー）
npm run preview
# → http://localhost:4173
```

> `npm run preview` はデータをビルド時のスナップショットから配信するため、
> `python src/run.py` で生成した新フレームはリアルタイム反映されません。
> 実際のデモでは `npm run dev` を使用してください。

---

## コマンドリファレンス

```bash
# ── セットアップ（初回のみ） ──
uv sync --extra dev           # Python 依存
cd src/apps && npm install    # npm 依存

# ── 起動（3 ターミナル） ──
python src/apps/server.py     # ターミナル A: WS サーバー
cd src/apps && npm run dev    # ターミナル B: Vite dev server
python src/run.py             # ターミナル C: 推論エージェント

# ── 動作確認 ──
cat data/perception_results/manifest.json          # フレーム数確認
ls data/perception_results/frames/ | wc -l        # フレームファイル数
python -c "import json; d=json.load(open('data/perception_results/frames/000000_img_0.json')); print(d['assessment']['risk_level'])"

# ── 本番ビルド ──
cd src/apps && npm run build   # dist/ に出力
cd src/apps && npm run preview # ビルド結果プレビュー

# ── デバッグ ──
PYTHONUNBUFFERED=1 python src/apps/server.py  # サーバー詳細ログ
```

---

## 関連ドキュメント

- [QUICK_START.md](QUICK_START.md) - エージェント単体の起動手順
- [ARCHITECTURE.md](ARCHITECTURE.md) - システム設計・LangGraph グラフ構造
- [INSPESAFE_INTEGRATION.md](INSPESAFE_INTEGRATION.md) - InspecSafe-V1 データセット連携
- [CLAUDE.md](../CLAUDE.md) - プロジェクト全体の仕様書

---

**最終更新:** 2026-03-20
**対応バージョン:** フレームスキップ機能 + 1フレーム1ファイル保存（commit 1e90be8 以降）
