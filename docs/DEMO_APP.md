# Safety View Agent Demo UI

React + Vite ベースの Demo UI です。
`src/apps/server.py` が `data/perception_results/` を監視し、WebSocket 経由で `src/apps/App.jsx` に結果を流します。

## できること

- 動画再生
- `vision_analysis.critical_points` の BBox オーバーレイ
- `assessment` の表示
- `audio` cue の表示
- Depth 画像の表示
- Infrared 画像の表示
- TTS 生成済み音声の自動再生
- Sync / Latest の切り替え

## 必要なもの

- Python 側
  - `uv sync --extra dev --extra demo`
- フロントエンド側
  - Node.js 18+
  - `cd src/apps && npm install`

## データ前提

Demo UI は以下を参照します。

- `data/perception_results/manifest.json`
- `data/perception_results/frames/*.json`
- `data/frames/`
- `data/depth/`（存在すれば表示）
- `data/infrared_frames/`（存在すれば表示）
- `data/voice/`（存在すれば再生）
- `data/videos/*.mp4`（ブラウザ動画用）

### 動画ファイルについて

- `inspesafe` モードでは `run.py` が自動生成します
- `manual` モードでは、`data/videos/` 内の動画が使われます
- `src/apps/server.py` が接続直後の `init` メッセージで実ファイル名を UI に通知するため、必ずしも `video.mp4` 固定である必要はありません
- ただし `data/videos/` を空にするとブラウザ側に動画 URL を渡せないため、動画表示を使う場合は少なくとも 1 本の MP4 を置いてください

## 起動手順

3 つのターミナルを使います。

### ターミナル 1: WebSocket サーバー

```bash
uv run python src/apps/server.py
```

既定ポート:

- WebSocket: `ws://127.0.0.1:8010`

### ターミナル 2: Vite

```bash
cd src/apps
npm install
npm run dev
```

既定ポート:

- フロントエンド: `http://localhost:5173`

### ターミナル 3: 推論

```bash
uv run python src/run.py
```

## データフロー

```text
run.py
  └─ data/perception_results/manifest.json + frames/*.json を更新
      └─ src/apps/server.py がポーリング
          └─ WebSocket 送信
              └─ App.jsx が動画時間に合わせて表示
```

## WebSocket メッセージで主に使うフィールド

- `frame_id`
- `video_timestamp`
- `assessment`
- `critical_points`
- `scene_description`
- `audio_cues`
- `depth_image_path`
- `infrared_image_path`
- `voice_path`

## 表示の仕様

### BBox

- `vision_analysis.critical_points` のうち `normalized_bbox` を持つものだけ描画されます
- `assessment.target_region` と `critical_points[].region_id` が一致すると強調表示されます

### 音声

- `tts.enabled=true` の場合、`assessment.safety_status` から `data/voice/*.wav` が生成されます
- App は `voice_path` があるフレームで自動再生します

### Sync / Latest

- `Sync`
  - 動画時刻に最も近いフレームを使います
- `Latest`
  - 最新到着フレームを表示します

## よくある確認ポイント

- WebSocket が未接続
  - `uv run python src/apps/server.py` が起動しているか
- 画面に何も出ない
  - `data/perception_results/manifest.json` が更新されているか
- 動画は出るが BBox が出ない
  - `critical_points` に `normalized_bbox` が無い可能性があります
- 音声が出ない
  - `tts.enabled` と `data/voice/` を確認してください
- Depth / Infrared が出ない
  - それぞれの出力ディレクトリが生成されているか確認してください

## 実務上の注意

- `server.py` は `manifest.json` をポーリングしているだけなので、古い出力が残っているとそのまま流れます
- `run.py` は新規実行時に `data/perception_results/` を `data/results_archive/` に退避します
- `server.py` は接続時に `data/videos/` から最初に見つかった MP4 を選び、UI へ `video_url` として通知します
- UI 側は `grounded_critical_points` ではなく、bbox を持つ `critical_points` をオーバーレイに使います

## 関連ドキュメント

- [QUICK_START.md](./QUICK_START.md)
- [SETUP.md](./SETUP.md)
- [ARCHITECTURE.md](./ARCHITECTURE.md)
