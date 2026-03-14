import React, { useEffect, useRef, useState } from 'react'

// ★ Insight ─────────────────────────────────────
// カラー定義をモジュールレベルに配置することで、
// useEffect 内の drawOverlay() 関数（クロージャ）から
// 直接参照できます。コンポーネント再レンダリング時に
// 定数が再定義されずメモリ効率向上。
// ─────────────────────────────────────────────────

const RISK_COLORS = {
  high:   { bg: '#5c0000', fg: '#ff6b6b', border: '#ff4444' },
  medium: { bg: '#5c3a00', fg: '#ffb347', border: '#ff8800' },
  low:    { bg: '#1a4a1a', fg: '#7fff7f', border: '#44ff44' },
}

const ACTION_COLORS = {
  emergency_stop: { bg: '#5c0000', fg: '#ff6b6b' },
  inspect_region: { bg: '#003080', fg: '#6699ff' },
  mitigate:       { bg: '#4a3a00', fg: '#ffcc44' },
  monitor:        { bg: '#2a2a2a', fg: '#999999' },
}

const TEMPORAL_COLORS = {
  new:        { bg: '#004444', fg: '#00ffff' },
  persistent: { bg: '#4a4a00', fg: '#ffff44' },
  worsening:  { bg: '#5c0000', fg: '#ff4444' },
  improving:  { bg: '#004400', fg: '#44ff44' },
  resolved:   { bg: '#002244', fg: '#4488ff' },
  unknown:    { bg: '#222222', fg: '#888888' },
}

const SEVERITY_COLORS = {
  critical: '#ff0000',
  high:     '#ff4444',
  medium:   '#ff8844',
  low:      '#ffdd44',
  unknown:  '#aaaaaa',
}

const styles = {
  body: {
    fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
    margin: '14px',
  },
  layout: {
    display: 'flex',
    gap: '14px',
    alignItems: 'flex-start',
  },
  left: {
    flex: '1 1 auto',
    minWidth: '320px',
  },
  right: {
    width: '420px',
    flex: '0 0 420px',
  },
  videoWrap: {
    position: 'relative',
    width: '960px',
    maxWidth: '100%',
  },
  video: {
    width: '100%',
    height: 'auto',
    display: 'block',
    background: '#000',
  },
  canvas: {
    position: 'absolute',
    left: '0',
    top: '0',
    pointerEvents: 'none',
  },
  panel: {
    background: '#111',
    color: '#ddd',
    borderRadius: '12px',
    padding: '10px 12px',
    marginBottom: '12px',
  },
  panelH3: {
    margin: '0 0 8px 0',
    fontSize: '13px',
    color: '#bbb',
    fontWeight: '600',
  },
  kv: {
    font: '12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace',
    whiteSpace: 'pre-wrap',
  },
  controls: {
    display: 'flex',
    gap: '10px',
    alignItems: 'center',
    flexWrap: 'wrap',
  },
  controlsLabel: {
    fontSize: '12px',
    color: '#bbb',
  },
  input: {
    background: '#222',
    color: '#ddd',
    border: '1px solid #333',
    borderRadius: '8px',
    padding: '6px 8px',
    fontSize: '12px',
  },
  button: {
    background: '#333',
    color: '#ddd',
    border: '1px solid #444',
    borderRadius: '8px',
    padding: '6px 12px',
    fontSize: '12px',
    cursor: 'pointer',
  },
  log: {
    height: '200px',
    overflow: 'auto',
    font: '12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace',
    whiteSpace: 'pre-wrap',
    border: '1px solid #333',
    borderRadius: '4px',
    padding: '8px',
    boxSizing: 'border-box',
  },
  hint: {
    fontSize: '12px',
    color: '#aaa',
    marginTop: '6px',
  },
  riskBadge: {
    display: 'inline-block', fontWeight: '700', fontSize: '14px',
    padding: '5px 12px', borderRadius: '6px', letterSpacing: '0.05em',
  },
  smallBadge: {
    display: 'inline-block', fontSize: '11px', fontWeight: '600',
    padding: '3px 8px', borderRadius: '4px',
  },
  hazardList: {
    margin: '0', paddingLeft: '16px', fontSize: '12px', color: '#ffb347',
  },
  progressBar: {
    height: '6px', background: '#333', borderRadius: '3px', overflow: 'hidden',
  },
  badge: {
    display: 'inline-block',
    fontSize: '11px',
    fontWeight: '600',
    padding: '4px 8px',
    borderRadius: '4px',
    marginBottom: '10px',
  },
  badgeConnected: {
    background: '#1a6f1a',
    color: '#7fff7f',
  },
  badgeDisconnected: {
    background: '#6f1a1a',
    color: '#ff7f7f',
  },
  badgeReconnecting: {
    background: '#6f6f1a',
    color: '#ffff7f',
  },
}

function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const audioRef = useRef(null)

  const [wsUrl, setWsUrl] = useState('ws://127.0.0.1:8001')
  const [mode, setMode] = useState('sync')
  const [delay, setDelay] = useState(0.7)
  const [status, setStatus] = useState('initializing...')
  const [connectionState, setConnectionState] = useState('initializing')
  const [log, setLog] = useState('')
  const [curAssessment, setCurAssessment] = useState(null)
  const [curSceneDesc, setCurSceneDesc] = useState('')
  const [sceneDescOpen, setSceneDescOpen] = useState(false)
  const [curDepthImagePath, setCurDepthImagePath] = useState(null)
  const [statusOpen, setStatusOpen] = useState(false)
  const [curVoicePath, setCurVoicePath] = useState(null)

  const wsRef = useRef(null)
  const resultsRef = useRef([])
  const recvCountRef = useRef(0)
  const tOffsetRef = useRef(null)
  const haveVideoPlayedRef = useRef(false)
  const animIdRef = useRef(null)
  const lastVideoTimeRef = useRef(0)
  const lastAssessmentFrameRef = useRef(null)

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx) return

    const updateStatus = (extra = '') => {
      const video = videoRef.current
      const vT = video?.currentTime || 0
      const off = tOffsetRef.current === null ? 'null' : tOffsetRef.current.toFixed(3)
      const wsState = wsRef.current?.readyState ?? -1
      setStatus(
        `ws: ${wsState} (0=CONNECTING,1=OPEN,2=CLOSING,3=CLOSED)\nrecv: ${recvCountRef.current}\nvideo.t: ${vT.toFixed(2)}\ntOffset: ${off}\n${extra}`
      )
    }

    const resizeCanvasToVideo = () => {
      const video = videoRef.current
      if (!video) return
      const rect = video.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      const canvas = canvasRef.current
      canvas.style.width = rect.width + 'px'
      canvas.style.height = rect.height + 'px'
      canvas.width = Math.round(rect.width * dpr)
      canvas.height = Math.round(rect.height * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    const appendLogLine = (msg) => {
      const lat = typeof msg.t_sent === 'number' ? (msg.t_sent - msg.t) : 0
      // frame_id を表示（存在しない場合はタイムスタンプを表示）
      const timeDisplay = msg.frame_id ? `[${msg.frame_id}]` : `${msg.t.toFixed(2)}s`
      const line = `${timeDisplay} (+${lat.toFixed(2)}s): ${msg.text ?? '(no text)'}\n`
      setLog((prev) => {
        const combined = prev + line
        if (combined.length > 25000) {
          return combined.slice(-18000)
        }
        return combined
      })
    }

    const drawOverlay = (criticalPoints, w, h, targetRegion) => {
      ctx.clearRect(0, 0, w, h)
      if (!criticalPoints?.length) return

      ctx.font = '12px system-ui, -apple-system, Segoe UI, sans-serif'

      const targetIdx = (() => {
        if (!targetRegion) return -1
        const m = targetRegion.match(/^critical_point_(\d+)$/)
        return m ? parseInt(m[1]) : -1
      })()

      for (let i = 0; i < criticalPoints.length; i++) {
        const cp = criticalPoints[i]
        if (!cp?.bbox || cp.bbox.length !== 4) continue
        const [bx1, by1, bx2, by2] = cp.bbox
        const px1 = bx1*w, py1 = by1*h, px2 = bx2*w, py2 = by2*h
        const isTarget = (i === targetIdx)

        if (isTarget) {
          // target_region: 白色・5px・点線 + 白背景ラベル
          ctx.strokeStyle = '#ffffff'
          ctx.lineWidth = 5
          ctx.setLineDash([8, 4])
          ctx.strokeRect(px1-2, py1-2, px2-px1+4, py2-py1+4)
          ctx.setLineDash([])
          const label = `→ TARGET CP${i} ${cp.severity ?? ''}`
          const tw = ctx.measureText(label).width
          ctx.fillStyle = 'rgba(255,255,255,0.85)'
          ctx.fillRect(px1, Math.max(0, py1-18), tw+8, 18)
          ctx.fillStyle = '#111'
          ctx.font = 'bold 12px system-ui'
          ctx.fillText(label, px1+4, Math.max(13, py1-4))
          ctx.font = '12px system-ui'
        } else {
          // 通常 CP: severity カラー・3px
          const color = SEVERITY_COLORS[cp.severity] ?? SEVERITY_COLORS.unknown
          ctx.strokeStyle = color
          ctx.lineWidth = 3
          ctx.strokeRect(px1, py1, px2-px1, py2-py1)
          const label = `CP${i} ${cp.severity ?? ''}`
          const tw = ctx.measureText(label).width
          ctx.fillStyle = 'rgba(0,0,0,0.7)'
          ctx.fillRect(px1, Math.max(0, py1-16), tw+8, 16)
          ctx.fillStyle = color
          ctx.fillText(label, px1+4, Math.max(12, py1-3))
        }
      }
      ctx.setLineDash([])
    }

    const pickLatestAtOrBefore = (targetT) => {
      for (let i = resultsRef.current.length - 1; i >= 0; i--) {
        if (resultsRef.current[i].t_adj <= targetT) return resultsRef.current[i]
      }
      return null
    }

    const onVideoPlayOnce = () => {
      haveVideoPlayedRef.current = true
      updateStatus('video started')
    }

    const connectWS = () => {
      if (wsRef.current) try { wsRef.current.close() } catch (e) { }
      recvCountRef.current = 0
      tOffsetRef.current = null
      resultsRef.current = []
      setLog('')
      setCurAssessment(null)
      setCurSceneDesc('')
      setCurDepthImagePath(null)
      setCurVoicePath(null)
      setSceneDescOpen(false)
      setStatusOpen(false)
      lastAssessmentFrameRef.current = null

      const url = wsUrl.trim()
      setConnectionState('connecting')
      wsRef.current = new WebSocket(url)

      wsRef.current.onopen = () => {
        setConnectionState('connected')
        updateStatus('✓ connected')
      }
      wsRef.current.onerror = () => {
        wsRef.current.close()
      }
      wsRef.current.onclose = () => {
        setConnectionState('reconnecting')
        updateStatus('✗ disconnected — reconnecting in 3s...')
        setTimeout(() => connectWS(), 3000)
      }

      wsRef.current.onmessage = (ev) => {
        const msg = JSON.parse(ev.data)
        recvCountRef.current++

        // 完全自動同期: video_timestamp がある場合は直接マッピング
        let t_adj
        if (typeof msg.video_timestamp === 'number') {
          // video_timestamp がある → 直接動画タイムラインにマッピング（tOffset 不要）
          t_adj = msg.video_timestamp
        } else {
          // 後方互換: 従来の tOffset ロジック
          // 絶対時刻ベースの同期：tOffset = videoTime - frameTimestamp
          // msg.t は Unix timestamp（秒単位）
          if (haveVideoPlayedRef.current && tOffsetRef.current === null && typeof msg.t === 'number') {
            tOffsetRef.current = videoRef.current.currentTime - msg.t
          }
          // msg.t が絶対時刻の場合、tOffset で補正して動画タイムラインにマッピング
          t_adj = tOffsetRef.current === null ? msg.t : msg.t + tOffsetRef.current
        }
        resultsRef.current.push({ ...msg, t_adj })

        const vNow = videoRef.current?.currentTime || 0
        const cut = vNow - 10
        while (resultsRef.current.length && resultsRef.current[0].t_adj < cut) resultsRef.current.shift()

        appendLogLine(msg)
        updateStatus('')
      }
      updateStatus('CONNECTING...')
    }

    const tick = () => {
      const video = videoRef.current
      if (video?.readyState >= 2) {
        const rect = video.getBoundingClientRect()
        const w = rect.width
        const h = rect.height

        // 動画が巻き戻されたら tOffset をリセット（自動同期リセット）
        if (video.currentTime < lastVideoTimeRef.current - 0.5) {
          tOffsetRef.current = null
          setStatus('⚠️ 動画が巻き戻されたため、同期をリセットしました')
        }
        lastVideoTimeRef.current = video.currentTime

        const D = Math.max(0, Number(delay || 0))
        let cur = null

        if (mode === 'latest') {
          cur = resultsRef.current.length ? resultsRef.current[resultsRef.current.length - 1] : null
        } else {
          const targetT = Math.max(0, video.currentTime - D)
          cur = pickLatestAtOrBefore(targetT)
        }

        if (cur) {
          // フレーム切り替わり時のみ state 更新（60fps で毎回 setState しない）
          if (cur.frame_id !== lastAssessmentFrameRef.current) {
            lastAssessmentFrameRef.current = cur.frame_id
            setCurAssessment(cur.assessment ?? null)
            setCurSceneDesc(cur.scene_description ?? '')
            setCurDepthImagePath(cur.depth_image_path ?? null)
            setCurVoicePath(cur.voice_path ?? null)

            // 音声再生（voice_path がある場合）
            if (cur.voice_path && audioRef.current) {
              audioRef.current.src = cur.voice_path
              audioRef.current.play().catch(() => {})
            }
          }
          drawOverlay(cur.critical_points, w, h, cur.assessment?.target_region)
          const vT = video.currentTime.toFixed(2)
          const off = tOffsetRef.current === null ? 'null' : tOffsetRef.current.toFixed(3)
          setStatus(`ws:${wsRef.current?.readyState ?? -1} recv:${recvCountRef.current} | video:${vT}s tOffset:${off}`)
        } else {
          if (lastAssessmentFrameRef.current !== null) {
            lastAssessmentFrameRef.current = null
            setCurAssessment(null)
            setCurSceneDesc('')
            setCurDepthImagePath(null)
            setCurVoicePath(null)
          }
          ctx.clearRect(0, 0, w, h)
        }
      }
      animIdRef.current = requestAnimationFrame(tick)
    }

    resizeCanvasToVideo()
    connectWS()

    window.addEventListener('resize', resizeCanvasToVideo)
    videoRef.current?.addEventListener('loadedmetadata', resizeCanvasToVideo)
    videoRef.current?.addEventListener('play', resizeCanvasToVideo)
    videoRef.current?.addEventListener('play', onVideoPlayOnce, { once: true })

    animIdRef.current = requestAnimationFrame(tick)

    return () => {
      if (animIdRef.current) cancelAnimationFrame(animIdRef.current)
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      window.removeEventListener('resize', resizeCanvasToVideo)
      videoRef.current?.removeEventListener('loadedmetadata', resizeCanvasToVideo)
      videoRef.current?.removeEventListener('play', resizeCanvasToVideo)
      videoRef.current?.removeEventListener('play', onVideoPlayOnce)
    }
  }, [mode, delay, wsUrl])

  return (
    <div style={styles.body}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '12px',
        marginBottom: '14px', paddingBottom: '10px',
        borderBottom: '1px solid #222',
      }}>
        <span style={{ fontSize: '18px', fontWeight: '700', letterSpacing: '0.05em' }}>
          ⚡ Safety View Agent
        </span>
        {/* 接続バッジ */}
        <div style={{
          ...styles.badge, marginBottom: 0,
          ...(connectionState === 'connected' ? styles.badgeConnected :
              connectionState === 'reconnecting' ? styles.badgeReconnecting :
              { background: '#6f1a1a', color: '#ff7f7f' }),
        }}>
          {connectionState === 'connected' ? '● LIVE' :
           connectionState === 'reconnecting' ? '⟳ RECONNECTING' : '● OFFLINE'}
        </div>
      </div>

      <div style={styles.layout}>
        <div style={styles.left}>
          <div style={styles.videoWrap}>
            <video
              ref={videoRef}
              controls
              src="/videos/video.mp4"
              style={styles.video}
            ></video>
            <canvas ref={canvasRef} style={styles.canvas}></canvas>
          </div>
          <audio ref={audioRef} style={{ display: 'none' }} />
          <div style={styles.hint}>
            ※ data/perception_results.json から推論結果を読み込みます。
          </div>
        </div>

        <div style={styles.right}>
          {/* Assessment Panel */}
          <div style={styles.panel}>
            <h3 style={styles.panelH3}>Assessment</h3>
            {curAssessment ? (
              <div>
                {/* Risk level + Priority */}
                <div style={{ display:'flex', alignItems:'center', gap:'8px', marginBottom:'10px' }}>
                  <span style={{
                    ...styles.riskBadge,
                    background: RISK_COLORS[curAssessment.risk_level]?.bg ?? '#222',
                    color:      RISK_COLORS[curAssessment.risk_level]?.fg ?? '#aaa',
                    border: `1px solid ${RISK_COLORS[curAssessment.risk_level]?.border ?? '#555'}`,
                  }}>
                    RISK: {(curAssessment.risk_level ?? '').toUpperCase()}
                  </span>
                  <div style={{ flex:1 }}>
                    <div style={{ fontSize:'10px', color:'#888', marginBottom:'2px' }}>
                      Priority {((curAssessment.priority ?? 0)*100).toFixed(0)}%
                    </div>
                    <div style={styles.progressBar}>
                      <div style={{
                        height:'100%',
                        width:`${(curAssessment.priority ?? 0)*100}%`,
                        background: RISK_COLORS[curAssessment.risk_level]?.border ?? '#666',
                        transition:'width 0.3s ease',
                      }} />
                    </div>
                  </div>
                </div>

                {/* action_type + temporal_status */}
                <div style={{ marginBottom:'8px', display:'flex', gap:'6px', flexWrap:'wrap' }}>
                  <span style={{
                    ...styles.smallBadge,
                    background: ACTION_COLORS[curAssessment.action_type]?.bg ?? '#222',
                    color:      ACTION_COLORS[curAssessment.action_type]?.fg ?? '#aaa',
                  }}>
                    {curAssessment.action_type ?? '-'}
                  </span>
                  <span style={{
                    ...styles.smallBadge,
                    background: TEMPORAL_COLORS[curAssessment.temporal_status]?.bg ?? '#222',
                    color:      TEMPORAL_COLORS[curAssessment.temporal_status]?.fg ?? '#aaa',
                  }}>
                    {curAssessment.temporal_status ?? 'unknown'}
                  </span>
                </div>

                {/* safety_status */}
                <div style={{ fontSize:'12px', color:'#ccc', marginBottom:'8px', lineHeight:'1.5' }}>
                  {curAssessment.safety_status}
                </div>

                {/* detected_hazards */}
                {curAssessment.detected_hazards?.length > 0 && (
                  <div style={{ marginBottom:'8px' }}>
                    <div style={{ fontSize:'11px', color:'#888', marginBottom:'4px' }}>Detected Hazards</div>
                    <ul style={styles.hazardList}>
                      {curAssessment.detected_hazards.map((h, i) => <li key={i}>{h}</li>)}
                    </ul>
                  </div>
                )}

                {/* target_region */}
                {curAssessment.target_region && (
                  <div style={{ fontSize:'11px', color:'#6699ff', marginBottom:'6px' }}>
                    Target: {curAssessment.target_region}
                  </div>
                )}

                {/* reason */}
                <div style={{ fontSize:'11px', color:'#777', borderTop:'1px solid #333', paddingTop:'6px', lineHeight:'1.5' }}>
                  {curAssessment.reason}
                </div>
              </div>
            ) : (
              <div style={{ fontSize:'12px', color:'#555' }}>(no assessment)</div>
            )}
          </div>

          {/* Depth Image Panel (右側だけ表示) */}
          {curDepthImagePath && (
            <div style={styles.panel}>
              <h3 style={styles.panelH3}>Depth Map</h3>
              <div style={{
                width: '100%',
                height: '150px',
                overflow: 'hidden',
                borderRadius: '4px',
                background: '#000',
              }}>
                <img
                  src={curDepthImagePath}
                  alt="depth map"
                  style={{
                    width: '200%',           /* side-by-side は 2倍の幅 */
                    height: '100%',
                    marginLeft: '-100%',     /* 右側だけを見えるようにシフト */
                    objectFit: 'cover',
                  }}
                />
              </div>
            </div>
          )}

          {/* Scene Description Panel */}
          <div style={styles.panel}>
            <h3
              style={{ ...styles.panelH3, cursor:'pointer', userSelect:'none' }}
              onClick={() => setSceneDescOpen(o => !o)}
            >
              Scene {sceneDescOpen ? '▾' : '▸'}
            </h3>
            {sceneDescOpen && (
              <div style={{ fontSize:'12px', color:'#bbb', lineHeight:'1.6', maxHeight:'120px', overflowY:'auto' }}>
                {curSceneDesc || '(no scene description)'}
              </div>
            )}
          </div>

          {/* Status Panel (Collapsible) */}
          <div style={styles.panel}>
            <h3
              style={{ ...styles.panelH3, cursor: 'pointer', userSelect: 'none' }}
              onClick={() => setStatusOpen(o => !o)}
            >
              Status {statusOpen ? '▾' : '▸'}
              <span style={{ fontSize: '11px', color: '#666', marginLeft: '8px' }}>
                {connectionState === 'connected' ? '● LIVE' :
                 connectionState === 'reconnecting' ? '⟳ RECONNECTING' : '● OFFLINE'}
              </span>
            </h3>
            {statusOpen && (
              <div>
                <div style={{ ...styles.controls, marginTop: '8px', flexDirection: 'column', alignItems: 'flex-start', gap: '12px' }}>
                  <label style={styles.controlsLabel}>
                    バックエンド URL
                    <input
                      type="text"
                      value={wsUrl}
                      onChange={(e) => setWsUrl(e.target.value)}
                      placeholder="ws://127.0.0.1:8001"
                      style={{ ...styles.input, width: '100%', marginTop: '4px' }}
                    />
                  </label>
                  <label style={styles.controlsLabel}>
                    表示モード
                    <select
                      value={mode}
                      onChange={(e) => setMode(e.target.value)}
                      style={styles.input}
                    >
                      <option value="sync">Sync (video time)</option>
                      <option value="latest">Latest (no sync)</option>
                    </select>
                  </label>
                  <label style={styles.controlsLabel}>
                    D(s){' '}
                    <input
                      type="number"
                      min="0"
                      step="0.1"
                      value={delay}
                      onChange={(e) => setDelay(parseFloat(e.target.value))}
                      style={styles.input}
                    />
                  </label>
                  <button
                    onClick={() => {
                      if (resultsRef.current.length > 0) {
                        const lastMsg = resultsRef.current[resultsRef.current.length - 1]
                        tOffsetRef.current = videoRef.current.currentTime - lastMsg.t
                        setStatus(`✓ 同期完了: offset=${tOffsetRef.current.toFixed(3)}s`)
                      } else {
                        setStatus('⚠ メッセージを受け取ってから実行してください')
                      }
                    }}
                    style={styles.button}
                  >
                    🔄 動画とBBoxを同期
                  </button>
                </div>
                <div id="status" style={{ ...styles.kv, marginTop: '8px', fontSize: '11px' }}>
                  {status}
                </div>
              </div>
            )}
          </div>

          {/* Log Panel */}
          <div style={styles.panel}>
            <h3 style={styles.panelH3}>Log</h3>
            <div style={styles.log}>{log}</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
