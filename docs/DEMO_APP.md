# React デモアプリ - VLM Overlay Demo

Safety View Agent のリアルタイム UI デモアプリケーション。
WebSocket を使用して、バックエンド処理結果をブラウザにストリーミング表示します。

## 概要

**VLM Overlay Demo** は、動画再生中にリアルタイムで BBox（バウンディングボックス）を描画するデモアプリです。

### 機能
- 動画再生（HTML5 video コントロール）
- リアルタイム BBox 描画（canvas オーバーレイ）
- WebSocket でサーバー接続（自動接続・再接続）
- 同期制御（Sync/Latest モード）
- ログパネル（受信データ表示）

### 技術スタック
- **フロントエンド**: React 18 + Vite
- **バックエンド**: Python WebSocket サーバー
- **プロトコル**: JSON over WebSocket

## ファイル構成

```
src/apps/
├── package.json               # npm 設定
├── vite.config.js             # Vite 設定（WebSocket プロキシ）
├── index.html                 # エントリーHTML（<div id="root">）
├── main.jsx                   # React エントリー
├── App.jsx                    # メインコンポーネント
└── server.py                  # WebSocket サーバー
```

## アーキテクチャ

```
┌──────────────────────────────┐
│  ブラウザ (http://localhost:5173)
│  ┌────────────────────────┐
│  │  React App (App.jsx)   │
│  ├────────────────────────┤
│  │ Video Player (canvas)  │
│  │ - BBox 描画            │
│  │ - sync/latest 切り替え │
│  ├────────────────────────┤
│  │ Status/HUD/Log Panel   │
│  └────────────────────────┘
└──────────────────────────────┘
           │
           │ ws://localhost:5173/ws
           │  (Vite proxy)
           │
           ▼
┌──────────────────────────────┐
│  WebSocket Server (Port 8001)
│  (python src/apps/server.py)
│  ┌────────────────────────┐
│  │ data/ 監視              │
│  │ (perception_results.json) │
│  └────────────────────────┘
└──────────────────────────────┘
```

## セットアップ

### 1. バックエンド（WebSocket サーバー）

```bash
cd /home/tetsutani/work/kanden-koi-voltmind-hackathon-2026

# 依存関係インストール（demo オプション）
uv sync --extra demo

# WebSocket サーバー起動
python src/apps/server.py
# → ws://localhost:8001 で待機
```

### 2. フロントエンド（React アプリ）

```bash
cd src/apps

# npm 依存関係インストール
npm install

# Vite dev server 起動
npm run dev
# → http://localhost:5173 で起動（Vite proxy: /ws → 8001）
```

### 3. ブラウザで確認

```
http://localhost:5173
```

## 使用方法

### 基本操作

#### 動画再生
1. ブラウザで `http://localhost:5173` を開く
2. Video Player エリアで「再生」をクリック
3. `/videos/video.mp4` が再生される

#### BBox 同期制御

**Status パネルの設定値：**

| 項目 | 説明 | 範囲 |
|------|------|------|
| **表示モード** | Sync = 動画時刻同期 / Latest = 最新のみ | - |
| **D(s)** | 処理遅延補正（秒） | 0.0 ～ 2.0 |

#### モード詳細

**Sync モード（推奨）:**
```
targetT = video.currentTime - D
→ 動画時刻より D 秒前の結果を表示
→ 実際の processing latency に合わせ調整
```

**Latest モード:**
```
最新の受信結果を常に表示
→ 同期不要（リアルタイム推し出し）
```

## WebSocket メッセージ形式

### サーバー → クライアント

#### Detection メッセージ

```json
{
  "t": 2.5,
  "t_sent": 2.95,
  "text": "人物を検出 (0.85)",
  "detections": [
    {
      "label": "person",
      "score": 0.85,
      "bbox": [0.3, 0.2, 0.7, 0.9]
    }
  ]
}
```

**フィールド:**
- `t`: サーバー側の時刻（秒）
- `t_sent`: メッセージ送信時刻（秒）
- `text`: 人間向けテキスト
- `detections`: BBox 配列
  - `label`: 物体名
  - `score`: 信度（0～1）
  - `bbox`: 正規化座標 `[x1, y1, x2, y2]`

## トラブルシューティング

### WebSocket が接続できない

**症状:** Status パネルが `CLOSED` のままで、データが流れない

**原因:**
1. WebSocket サーバーが起動していない
2. Vite proxy 設定が誤っている
3. ファイアウォール/ポートブロック

**対策:**
```bash
# サーバーが起動しているか確認
curl http://localhost:8001
# Expected: Connection refused (サーバーが起動していないはず)

# サーバー起動
python src/apps/server.py

# ブラウザで ws:// 接続を確認
# → F12 (DevTools) > Console で WS エラー確認
```

### 動画が再生されない

**症状:** ビデオプレイヤーが真っ黒で何も表示されない

**原因:**
1. `data/videos/` に動画ファイルが存在しない
2. Vite dev server が publicDir (data/) を配信していない

**対策:**
```bash
# data/videos/ に動画ファイルがあるか確認
ls -lh data/videos/

# 存在しない場合は、動画ファイルを配置してください
cp /path/to/video.mp4 data/videos/
```

### BBox が描画されない

**症状:** 動画は再生されるが、緑枠が出ない

**原因:**
1. WebSocket 接続が確立していない
2. Mode が Latest なのに message がまだ届いていない
3. canvas サイズが正しく計算されていない
4. data/perception_results.json がない

**対策:**
```
1. Status パネル確認: WS 接続状態が `1=OPEN` か？
   → NO: WebSocket サーバーを起動
2. Mode を Latest に変更 → BBox が出るか？
   → YES: video_timestamp 同期に時間がかかっているだけ
3. ブラウザ DevTools > Elements で <canvas> サイズ確認
4. data/perception_results.json が存在し、video_timestamp が記録されているか確認
```

## 開発ガイド

### App.jsx の状態管理

```jsx
// WebSocket 接続
wsRef.current = new WebSocket(wsUrl)

// 受信メッセージ処理
wsRef.current.onmessage = (ev) => {
  const msg = JSON.parse(ev.data)
  // msg.t: サーバー時刻
  // msg.detections: BBox 配列
}

// BBox 描画
drawDetections(cur.detections, w, h)
```

### Canvas 描画カスタマイズ

色やサイズを変更したい場合は、`App.jsx` の `drawDetections()` 関数を編集：

```jsx
const drawDetections = (dets, w, h) => {
  ctx.lineWidth = 3           // ← 枠線の太さ
  ctx.strokeStyle = 'lime'    // ← 枠線の色
  ctx.fillStyle = 'rgba(0,0,0,0.6)'  // ← ラベルの背景色
}
```

### Server.py の拡張

`server.py` の `monitor_and_stream()` 関数に新しい logic を追加できます：

```python
async def monitor_and_stream(websocket):
    # data/perception_results.json を監視
    # 変更検出時に WebSocket で送信（video_timestamp 含む）
```

## パフォーマンス

| 項目 | 値 | 説明 |
|------|-----|------|
| Vite dev server 起動 | 1-2秒 | HMR 対応 |
| Canvas 更新 | 60 FPS | requestAnimationFrame で毎フレーム描画 |
| ファイル監視 | 0.5秒 | server.py のポーリング間隔 |
| 推奨 D(s) 値 | 0.5-1.0 | 処理 latency による |

## 将来の拡張

### 1. run.py との統合

現在、server.py が `data/perception_results.json` を監視してストリーミングしています。
video_timestamp フィールドにより、動画フレームと検出結果の完全自動同期を実現。

将来は以下が可能：
- フレームごとのリアルタイム処理表示
- 処理状態表示（Running/Done/Error）
- 複数フレームの同時表示

### 2. 自動再接続

WebSocket 接続が切れた場合、自動的に再接続する機能

### 3. 処理状態表示

"Running", "Done", "Error" などのステータスを HUD で表示

## 参考資料

- [Vite Documentation](https://vitejs.dev/)
- [React Documentation](https://react.dev/)
- [WebSocket MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)

---

**最終更新:** 2026-03-04
**バージョン:** v1.0 (Beta)
**対象:** Safety View Agent React Demo App
