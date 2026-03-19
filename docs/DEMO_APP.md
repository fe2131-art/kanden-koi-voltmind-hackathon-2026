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
- 深度画像・音声ファイルのブラウザ表示対応

### 技術スタック
- **フロントエンド**: React 18 + Vite
- **バックエンド**: Python WebSocket サーバー（asyncio）
- **プロトコル**: JSON over WebSocket
- **通信ポート**: 8001（WebSocket）→ 5173（Vite proxy）

## ファイル構成

```
src/apps/
├── __init__.py                # Python パッケージマーク
├── package.json               # npm 設定（React, Vite 依存）
├── package-lock.json          # npm ロックファイル
├── vite.config.js             # Vite 設定（/ws → localhost:8001 プロキシ）
├── index.html                 # エントリーHTML（<div id="root">）
├── main.jsx                   # React エントリーポイント
├── App.jsx                    # メインコンポーネント（ビデオプレイヤー + HUD）
├── server.py                  # WebSocket サーバー（データ監視・ストリーミング）
└── dist/                      # ビルド出力ディレクトリ（npm run build 後）
```

## 前提条件

### システム要件
- **Python**: 3.12.x（プロジェクト構成で指定）
- **Node.js**: 18.0.0 以上（npm はバンドル）
- **OS**: Linux / macOS / Windows（WSL2）
- **ポート**: 5173（Vite dev server）、8001（WebSocket server） 開放確認

### 環境準備の確認

```bash
# Python バージョン確認（3.12 推奨）
python3 --version
# → Python 3.12.x

# Node.js バージョン確認
node --version
# → v18.0.0 以上

# npm バージョン確認
npm --version
# → 9.0.0 以上
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

デモアプリの起動には **Python バックエンド** と **React フロントエンド** の 2 つのプロセスが必要です。
別々のターミナルウィンドウで実行することを推奨します。

### 【ステップ 1】 Python 環境準備（初回のみ）

プロジェクトのルートディレクトリで実行：

```bash
cd /home/tetsutani/work/kanden-koi-voltmind-hackathon-2026

# uv を用いた依存関係インストール（demo オプション含める）
uv sync --extra demo

# 確認: 仮想環境が作成されたか
ls -la .venv/
# → .venv/ ディレクトリが存在すれば OK
```

**何をインストールされるか:**
- `websockets>=11.0` (WebSocket サーバー用)
- その他 Python 依存（pydantic, langgraph など）

### 【ステップ 2】 npm 依存関係準備（初回のみ）

```bash
# React アプリディレクトリへ移動
cd src/apps

# npm 依存関係をインストール
npm install

# 確認: node_modules が作成されたか
ls -d node_modules/
# → node_modules/ ディレクトリが存在すれば OK
```

**何をインストールされるか:**
- React 18.2.0
- react-dom 18.2.0
- Vite 5.0.0 (および関連パッケージ)

### 【ステップ 3】 データディレクトリ準備

デモアプリはエージェント実行結果を `data/` 配下から読み込みます。

```bash
# プロジェクトルートに戻る
cd /home/tetsutani/work/kanden-koi-voltmind-hackathon-2026

# 必須ディレクトリの確認・作成
mkdir -p data/{frames,depth,voice,infrared_frames}

# 動画サンプルの確認
ls -lh data/videos/
# → 少なくとも 1 つの .mp4 ファイルが必要

# JSON 初期化（初回のみ）
echo '{"frames": []}' > data/perception_results.json
```

### 【ステップ 4】 WebSocket サーバー起動（ターミナル A）

```bash
# プロジェクトルートで実行
cd /home/tetsutani/work/kanden-koi-voltmind-hackathon-2026

# WebSocket サーバーを起動
python src/apps/server.py

# 期待される出力:
# → "[WebSocket] Server listening on ws://0.0.0.0:8001"
```

**サーバーが起動しない場合:**
- ポート 8001 が既に使用されていないか確認（`lsof -i :8001`）
- Python 環境が有効か確認（`.venv/bin/python` で実行）

### 【ステップ 5】 Vite dev server 起動（ターミナル B）

```bash
# React アプリディレクトリで実行
cd src/apps

# Vite 開発サーバーを起動
npm run dev

# 期待される出力:
# → "VITE v5.x.x ready in xxx ms"
# → "➜  Local:   http://localhost:5173/"
# → "➜  press h to show help"
```

**サーバーが起動しない場合:**
- ポート 5173 が既に使用されていないか確認（`lsof -i :5173`）
- Node.js が PATH に含まれているか確認（`which node`）

### 【ステップ 6】 ブラウザで確認

```
http://localhost:5173
```

ブラウザで以下が表示されれば成功：
- **Video Player**: 空の黒いエリア（動画なし時）
- **Status Panel**: `WS State: OPEN` と表示
- **Log Panel**: WebSocket 接続ログ表示

### 【ステップ 7】 エージェント実行でデータ生成（ターミナル C）

デモアプリにデータを送信するには、Safety View Agent を実行して `data/perception_results.json` を生成する必要があります：

```bash
# プロジェクトルートで実行
cd /home/tetsutani/work/kanden-koi-voltmind-hackathon-2026

# 設定ファイルの data.mode に応じて自動実行
python src/run.py

# または、設定ファイルを指定
python src/run.py --config configs/default.yaml
```

**対応モード** (configs/default.yaml の `data.mode` で設定):
- `manual`: `data/videos/` のビデオを自動展開
- `inspesafe`: InspecSafe-V1 データセットのセッションを指定（後述）
- `demo`: デモ観測モード（サンプルデータで実行）

実行中：
- `data/frames/` に RGB フレーム画像が生成
- `data/infrared_frames/` に赤外線フレームが生成（inspesafe モードのみ）
- `data/perception_results.json` が更新される（frames 配列が増加）
- React UI が自動的にデータをストリーミング表示

### トラブルシューティング：セットアップ時のエラー

**「Module not found: websockets」**
```bash
# 再度インストール
uv sync --extra demo --force

# 環境確認
python -c "import websockets; print(websockets.__version__)"
```

**「npm: command not found」**
```bash
# Node.js を再インストール（Homebrew / apt など）
which npm
node --version
```

**「VITE proxy エラー: Cannot connect to ws://localhost:8001」**
```bash
# WebSocket サーバーが起動しているか確認
curl -v http://localhost:8001
# 期待される応答: Connection refused or WebSocket upgrade response

# サーバーログを確認
# ターミナル A で実行中のサーバーログを確認
```

## 使用方法

### 基本操作フロー

#### 1. ブラウザを開く
```
http://localhost:5173
```

#### 2. 接続状態確認
Status Panel で以下を確認：
```
WS State: OPEN          ✅ WebSocket 接続確立
Connected: yyyy-mm-dd hh:mm:ss
FPS: 0 (receiving...)
```

#### 3. エージェント実行でデータ生成
別ターミナル（C）で：
```bash
python src/run.py --mode demo
```

実行中、UI は以下を表示します：
- **Log Panel**: 受信したフレーム数
- **Canvas**: BBox が描画される（data/perception_results.json の frames 配列が増加）

#### 4. 動画再生
Video Player エリアで「再生」をクリック。対応する `.mp4` ファイルが再生されます。

### 表示モード・パラメータ制御

#### Status Panel の設定値

| パラメータ | 説明 | 範囲 | 推奨値 |
|-----------|------|------|--------|
| **Mode** | 同期モード | Sync / Latest | Sync |
| **D (Delay, 秒)** | 処理遅延補正（動画フレームと検出結果のズレを吸収） | 0.0 ～ 2.0 | 0.5-1.0 |
| **FPS** | フレームレート（受信更新速度） | 読み取り専用 | - |

#### Sync モード（推奨）
```
targetTime = video.currentTime - D(秒)
→ 動画フレーム時刻より D 秒前の検出結果を表示
→ 例: video.currentTime = 2.5s, D = 1.0s → 1.5s 時刻の結果を表示
→ 実際の processing latency に合わせて D を調整
```

**推奨設定：**
- RGB フレーム分析のみ: D = 0.3 ～ 0.7 秒
- VLM + 深度分析: D = 0.5 ～ 1.0 秒
- 複数モダリティ（深度 + 音声 含む）: D = 1.0 ～ 2.0 秒

#### Latest モード
```
最新の受信結果を常に表示
→ 動画フレーム時刻との同期なし（リアルタイム推し出し）
→ 同期遅延をより容易に可視化する際に使用
```

### 実践例

**シナリオ: InspecSafe-V1 データセットでデモを実行**

```bash
# 【事前準備】configs/default.yaml を編集
# data:
#   mode: "inspesafe"
#   inspesafe:
#     dataset_path: "../InspecSafe-V1"
#     session: "train/Other_modalities/58132919535743_20251118_session_1400_2#bowenguanshang-you"

# ターミナル A: WebSocket サーバー起動
python src/apps/server.py

# ターミナル B: Vite dev server 起動（src/apps で実行）
npm run dev

# ターミナル C: エージェント実行（config 自動読み込み）
python src/run.py

# → ブラウザで http://localhost:5173 を開く
# → Status: OPEN を確認
# → Video Player で再生ボタンをクリック
# → Canvas に BBox がリアルタイム描画される
```

## WebSocket メッセージ形式

### サーバー → クライアント

#### フレーム分析結果メッセージ

```json
{
  "t": 2.5,
  "video_timestamp": 2.5,
  "text": "Robot in motion - Normal operation",
  "frame_id": "frame_2.500s.jpg",
  "assessment": {
    "risk_level": "low",
    "action_type": "monitor",
    "reason": "Normal operation detected"
  },
  "critical_points": [
    {
      "description": "Robot moving",
      "severity": "low",
      "bbox": [0.3, 0.2, 0.7, 0.9]
    }
  ],
  "scene_description": "Industrial workspace with robot moving at normal speed",
  "depth_image_path": "/depth/frame_2.500s.jpg",
  "voice_path": "/voice/frame_2.500s.wav"
}
```

**フィールド:**
- `t`: フレームタイムスタンプ（秒）
- `video_timestamp`: 動画内での時刻（秒、指定時は video_timestamp を優先）
- `text`: 人間向けシーン説明テキスト
- `frame_id`: フレーム識別子（ファイル名）
- `assessment`: 安全性評価（risk_level, action_type, reason）
- `critical_points`: 注意が必要なポイント配列
  - `description`: ポイント説明
  - `severity`: 重要度（low, medium, high）
  - `bbox`: 正規化座標 `[x_min, y_min, x_max, y_max]`
- `scene_description`: シーン詳細説明
- `depth_image_path`: 深度画像へのパス（オプション）
- `voice_path`: 対応音声ファイルへのパス（オプション）

## トラブルシューティング

### ケース 1: WebSocket が接続できない

**症状:**
- Status Panel: `WS State: CLOSED`
- Console: `WebSocket is closed`
- データが受信されない

**診断:**

```bash
# 1. サーバーが起動しているか確認
ps aux | grep "python.*server.py"

# 2. ポート 8001 がリッスンしているか確認
lsof -i :8001
# → 出力がない場合、サーバーが起動していない

# 3. ファイアウォール確認（Linux）
sudo ufw status
# → 8001 が許可されているか確認
```

**対策:**

```bash
# ターミナ A でサーバーを起動
cd /home/tetsutani/work/kanden-koi-voltmind-hackathon-2026
python src/apps/server.py

# 期待される出力:
# → "[WebSocket] Server listening on ws://0.0.0.0:8001"

# ブラウザを再読み込み（F5 または Ctrl+R）
```

**詳細デバッグ:**

```bash
# curl で WebSocket エンドポイントをテスト
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
  http://localhost:8001

# → HTTP/1.1 101 Switching Protocols が返されれば OK
```

### ケース 2: 動画が再生されない

**症状:**
- Video Player が真っ黒
- 再生ボタンをクリックしても何も起きない
- DevTools Console にエラーがない

**診断:**

```bash
# 1. data/videos/ に動画ファイルがあるか確認
ls -lh data/videos/
# → 最低 1 つ以上の .mp4 ファイルが必要

# 2. 動画ファイルが有効か確認（ffprobe で検査）
ffprobe -v error data/videos/sample.mp4
# → codec, fps, duration などが表示されれば OK

# 3. Vite 設定で publicDir が正しいか確認
grep -A 5 "publicDir" src/apps/vite.config.js
```

**対策:**

```bash
# 1. 動画ファイルがない場合は配置
cp /path/to/sample.mp4 data/videos/

# 2. 動画ファイルのパーミッション確認
chmod 644 data/videos/*.mp4

# 3. Vite dev server を再起動（src/apps で）
npm run dev

# 4. ブラウザを再読み込み
```

**期待される URL:**
```
<video> タグが読み込む URL:
http://localhost:5173/videos/sample.mp4
```

### ケース 3: BBox が描画されない

**症状:**
- 動画は再生される
- Canvas は表示される（灰色）
- 緑枠（BBox）が出ない
- Console にエラーはない

**診断:**

```bash
# 1. WebSocket 接続状態を確認
# → Status Panel: WS State が OPEN か？

# 2. data/perception_results.json が存在し、データが入っているか確認
cat data/perception_results.json | jq '.'
# → {"frames": [...]} 形式で、frames 配列が空でないか確認

# 3. フレームのタイムスタンプが正しいか確認
cat data/perception_results.json | jq '.frames[0]'
# → video_timestamp フィールドが存在するか確認

# 4. ブラウザ DevTools > Console で Canvas エラーを確認
# → F12 > Console タブ
```

**対策:**

```bash
# 1. data/perception_results.json がない場合は初期化
echo '{"frames": []}' > data/perception_results.json

# 2. エージェント実行でデータを生成
python src/run.py --mode demo

# 実行中、以下を確認:
# → data/perception_results.json が更新される
# → data/frames/ に画像ファイルが生成される
# → ブラウザ Log Panel にフレーム受信が表示される

# 3. Mode を "Latest" に変更して、BBox が出るか確認
# → Sync モードで出ない場合、遅延設定の問題の可能性
```

**Sync モード時に BBox が出ない場合:**

```bash
# D(遅延補正) を調整
# 1. Mode を "Sync" に設定
# 2. D を 0.0 から始めて、0.1 ずつ増加させてテスト
# 3. BBox が出始める D 値を記録（推奨: 0.5-1.5 秒）

# JSON 確認コマンド
cat data/perception_results.json | jq '.frames | length'
# → フレーム数が増加していることを確認
```

### ケース 4: npm install でエラー

**症状:**
```
npm ERR! Unable to resolve dependency tree
```

**対策:**

```bash
# 1. 既存の node_modules とロックファイルを削除
rm -rf node_modules package-lock.json

# 2. 再度インストール
npm install

# 3. Node.js が新しいか確認
node --version   # 18.0.0 以上
npm --version    # 9.0.0 以上
```

### ケース 5: Vite dev server の起動エラー

**症状:**
```
error: port 5173 is already in use by another process
```

**対策:**

```bash
# 1. ポート 5173 を使用しているプロセスを確認
lsof -i :5173

# 2. プロセスを終了
kill -9 <PID>

# 3. 別のポートで起動
npm run dev -- --port 5174
```

### ケース 6: サーバーログが見えない（デバッグしたい）

**対策:**

```bash
# Python サーバーの詳細ログを有効化
PYTHONUNBUFFERED=1 python src/apps/server.py

# または、テンポラリにログレベルを変更（server.py 編集）
# logging.basicConfig(level=logging.DEBUG)
```

## 開発・カスタマイズガイド

### App.jsx の構造

メインファイルの重要な関数：

```jsx
// 1. WebSocket 接続初期化
useEffect(() => {
  wsRef.current = new WebSocket(wsUrl)
  wsRef.current.onopen = () => setWsState(1)    // OPEN
  wsRef.current.onmessage = handleMessage
}, [])

// 2. メッセージ処理
const handleMessage = (ev) => {
  const msg = JSON.parse(ev.data)
  setFrames(prev => [...prev, msg])  // フレームバッファに追加
}

// 3. Canvas 描画
useEffect(() => {
  drawFrame()  // requestAnimationFrame で毎フレーム実行
}, [currentFrame, mode, delaySeconds])

// 4. フレーム抽出（Sync / Latest）
const getCurrentFrame = () => {
  if (mode === 'Latest') {
    return frames[frames.length - 1]  // 最新フレーム
  } else {
    // Sync: targetTime = video.currentTime - D
    const targetTime = video.currentTime - delaySeconds
    return frames.find(f => f.video_timestamp <= targetTime)
  }
}
```

### Canvas 描画のカスタマイズ

BBox の色、枠線の太さ、ラベルフォントなどを変更：

```jsx
// src/apps/App.jsx の drawDetections() 関数

const drawDetections = (detections, videoWidth, videoHeight) => {
  const ctx = canvasRef.current?.getContext('2d')
  if (!ctx) return

  // 枠線設定
  ctx.lineWidth = 4              // ← 太さ: 2-6 推奨
  ctx.strokeStyle = '#00FF00'    // ← 色: '#00FF00'(緑) / '#FF0000'(赤) など

  // テキストラベル設定
  ctx.font = 'bold 16px Arial'   // ← フォント: Arial, Helvetica, monospace など
  ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'  // ← 背景: 透明度変更は最後の値
  ctx.textBaseline = 'top'

  detections.forEach(det => {
    const [x1, y1, x2, y2] = det.bbox.map(v => v * (det.isNormalized ? 1 : videoWidth))

    // BBox を描画
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

    // ラベル描画
    const label = `${det.label} ${det.score.toFixed(2)}`
    ctx.fillRect(x1, y1 - 25, label.length * 8, 25)
    ctx.fillStyle = '#FFFFFF'
    ctx.fillText(label, x1 + 5, y1 - 20)
  })
}
```

### server.py の拡張例

新しい検出タイプを追加する場合：

```python
# src/apps/server.py の monitor_and_stream() 関数内

async def monitor_and_stream(websocket):
    """Extend to support custom data types."""

    while True:
        if PERCEPTION_RESULTS.exists():
            data = json.load(open(PERCEPTION_RESULTS))
            frames = data.get('frames', [])

            for frame in frames[last_count:]:
                # カスタム変換ロジック
                custom_msg = {
                    'frame_id': frame.get('frame_id'),
                    'video_timestamp': frame.get('video_timestamp'),
                    'detections': extract_detections(frame),
                    'custom_field': frame.get('custom_data'),  # ← 新フィールド
                }

                await websocket.send(json.dumps(custom_msg))
```

### ログレベルの変更

デバッグを容易にするため、ログレベルを変更：

```python
# src/apps/server.py の冒頭に追加

import logging
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG / INFO / WARNING / ERROR
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 利用
logger.debug(f"Processing frame: {frame_id}")
logger.info(f"Client connected: {websocket.remote_address}")
```

### Vite 設定のカスタマイズ

デバッグポート、プロキシ設定を変更：

```js
// src/apps/vite.config.js

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,           // ← 変更可能
    host: '0.0.0.0',      // リモートアクセス許可
    proxy: {
      '/ws': {
        target: 'ws://localhost:8001',
        ws: true,
        rewrite: path => path.replace(/^\/ws/, '')
      }
    }
  }
})
```

## パフォーマンス

### 動作指標

| メトリクス | 値 | 説明 |
|-----------|-----|------|
| Vite dev server 起動時間 | 1-2秒 | HMR（Hot Module Replacement）対応 |
| Canvas 描画フレームレート | 60 FPS | requestAnimationFrame で毎フレーム実行 |
| WebSocket メッセージ遅延 | <100ms | ローカルホスト通信（localhost:8001） |
| ファイル監視ポーリング間隔 | 0.5秒 | server.py が data/ をスキャン |
| JSON パース時間 | <10ms | 小規模データセット（frames < 1000） |
| 推奨 Canvas サイズ | 1280x720 | アスペクト比 16:9 |
| 推奨 D(delay) 値 | 0.5-1.5秒 | RGB + VLM 並列処理時の処理 latency |

### メモリ使用量

```
フレームバッファ（in-memory）:
- 100 フレーム × 1KB/frame ≈ 100KB
- 1000 フレーム × 1KB/frame ≈ 1MB

推奨: 最新 100 フレームのみメモリ保持
（古いフレームは自動削除）

実装例:
frames = frames.slice(-100)  // 最新 100 フレームのみ
```

### 最適化のコツ

```jsx
// 1. 不要な re-render を避ける
const memoFrame = useMemo(() => getCurrentFrame(), [frames, mode, delay])

// 2. Canvas 操作を最適化
const drawFrame = useCallback(() => {
  // 差分描画: 前フレームと異なる BBox のみ再描画
}, [])

// 3. WebSocket メッセージバッチング
// サーバー側で複数フレームをまとめて送信
```

## 実装のポイント

### 1. ファイル同期の仕組み

```
data/perception_results.json
  ├─ frames[] 配列（フラット構造）
  │
  ↓
server.py (監視・増分ストリーミング)
  ├─ 前回の frames 数をトラック
  ├─ vision_analysis.critical_points → normalized_bbox 変換
  │
  ↓
WebSocket メッセージ送信
  │
  ↓
App.jsx (受信・バッファに追加)
  ↓
Canvas 描画
```

**重要:** `perception_results.json` の `frames` 配列が増加すると、新しいフレームが自動的に WebSocket で送信されます。

### 2. タイムスタンプ同期

```javascript
// フレームオブジェクトのフィールド（フラット構造）:
{
  "frame_id": "frame_0.000s.jpg",
  "video_timestamp": 0.0,           // ← 動画内での時刻（秒）
  "timestamp": 1710349200.123,      // ← Unix timestamp
  "vision_summary": "...",           // VLM 分析結果（scene_description, critical_points, blind_spots）
  "objects": [...],                 // 検出オブジェクト
  "hazards": [...],                 // 推定ハザード
  "audio": [...],                   // 音声キュー
  "unobserved": [...],              // 未観測領域
  "assessment": {...},              // LLM / 固定値による安全判断
  "world_state": {...},             // 世界モデル（known_hazards, blind_spots）
  "errors": []                       // エラー情報
}

// App.jsx での同期:
const targetTime = video.currentTime - delaySeconds
const matchingFrame = frames.find(f => f.video_timestamp <= targetTime)
```

### 3. エラーハンドリング

```python
# server.py でのエラーハンドリング例
try:
    data = json.load(open(PERCEPTION_RESULTS))
except json.JSONDecodeError as e:
    logger.error(f"JSON parse error: {e}")
    # 前回の有効なデータを使用（フォールバック）

except asyncio.CancelledError:
    logger.info("WebSocket closed by client")
```

## 将来の拡張アイデア

### 1. リアルタイム処理ステータス表示

```jsx
// Status Panel に処理状態を追加
<div className="processing-status">
  Running: {runningFrames}
  Completed: {completedFrames}
  Error: {errorCount}
</div>
```

### 2. マルチフレームコンパレーション

```jsx
// 複数フレーム（t-1, t, t+1）を同時表示
<div className="multi-frame-view">
  <Canvas key="prev" frame={frames[i-1]} />
  <Canvas key="curr" frame={frames[i]} />
  <Canvas key="next" frame={frames[i+1]} />
</div>
```

### 3. 深度画像・音声ファイルの統合表示

```jsx
// 既に対応済み:
// - depth_image_path: /depth/frame_0.000s.jpg
// - voice_path: /voice/frame_0.000s.wav
// UI にオーディオプレイヤーと深度ビューアを追加可能
```

### 4. 自動再接続 + 再構成キュー

```javascript
// WebSocket 切断時の自動復旧
const reconnectWebSocket = async () => {
  while (!isConnected) {
    try {
      wsRef.current = new WebSocket(wsUrl)
      await delay(500)
    } catch (e) {
      // 指数バックオフで再試行
      await delay(Math.min(1000 * Math.pow(2, retryCount), 30000))
    }
  }
}
```

### 5. フレーム録画 / エクスポート

```javascript
// Canvas 描画結果を MP4 に変換（MediaRecorder API）
const recorder = new MediaRecorder(canvas.captureStream())
recorder.start()
// ... 再生 ...
recorder.stop()
```

## コマンドリファレンス

### セットアップ

```bash
# Python 環境準備
cd /home/tetsutani/work/kanden-koi-voltmind-hackathon-2026
uv sync --extra demo

# npm 準備
cd src/apps
npm install
```

### 実行

```bash
# ターミナル A: WebSocket サーバー
python src/apps/server.py

# ターミナル B: Vite dev server
cd src/apps && npm run dev

# ターミナル C: エージェント実行
python src/run.py --mode demo
```

### ビルド・デプロイ

```bash
# 本番ビルド
cd src/apps
npm run build
# → dist/ フォルダに静的ファイルが生成される

# ビルト結果をプレビュー
npm run preview
```

### デバッグ

```bash
# Python server デバッグ
PYTHONUNBUFFERED=1 python src/apps/server.py

# npm デバッグ（verbose）
npm run dev -- --debug

# WebSocket 接続確認
curl -v http://localhost:8001

# JSON データ確認
jq '.frames | length' data/perception_results.json
```

## 参考資料

- [Safety View Agent CLAUDE.md](/CLAUDE.md) - プロジェクト全体設定
- [Vite 公式ドキュメント](https://vitejs.dev/)
- [React 公式ドキュメント](https://react.dev/)
- [WebSocket MDN リファレンス](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)
- [WebSocket サーバー (websockets ライブラリ)](https://websockets.readthedocs.io/)
- [npm - package.json スクリプト](https://docs.npmjs.com/cli/v10/using-npm/scripts)

---

**最終更新:** 2026-03-19
**バージョン:** v1.2 (Frames API Integration)
**対象:** Safety View Agent - React Demo App
**セットアップ難易度:** ⭐⭐☆☆☆ (簡単)
