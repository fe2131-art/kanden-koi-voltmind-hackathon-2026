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

const MODALITY_COLORS = {
  vision:   { bg: '#002255', fg: '#66aaff' },
  depth:    { bg: '#220055', fg: '#cc88ff' },
  audio:    { bg: '#003322', fg: '#44ddaa' },
  infrared: { bg: '#441a00', fg: '#ff9944' },
  temporal: { bg: '#3a3a00', fg: '#eeee44' },
  sam3:     { bg: '#3a0055', fg: '#ee44ff' },
}

const DEPTH_ZONE_COLORS = {
  near: { bg: '#4a0000', fg: '#ff8888' },
  mid:  { bg: '#2a2a00', fg: '#ffee88' },
  far:  { bg: '#002244', fg: '#88aaff' },
}

function ModalitySection({ title, defaultOpen = true, count, badge, children }) {
  const [open, setOpen] = React.useState(defaultOpen)
  return (
    <div style={{ borderTop: '1px solid #2e2e2e', paddingTop: '8px' }}>
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer',
          fontSize: '12px', fontWeight: '600', color: '#aaa',
          marginBottom: open ? '8px' : 0, userSelect: 'none',
        }}
        onClick={() => setOpen(o => !o)}
      >
        <span>{title} {open ? '▾' : '▸'}</span>
        {count != null && count > 0 && (
          <span style={{
            fontSize: '10px', background: '#333', color: '#888',
            padding: '1px 5px', borderRadius: '3px',
          }}>{count}</span>
        )}
        {badge && (
          <span style={{
            display: 'inline-block', fontSize: '10px', fontWeight: '600',
            padding: '2px 6px', borderRadius: '4px', marginLeft: 'auto',
            background: RISK_COLORS[badge]?.bg ?? '#222',
            color: RISK_COLORS[badge]?.fg ?? '#aaa',
          }}>{badge}</span>
        )}
      </div>
      {open && children}
    </div>
  )
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
    width: '460px',
    flex: '0 0 460px',
    maxHeight: 'calc(100vh - 28px)',
    overflowY: 'auto',
    position: 'sticky',
    top: '14px',
  },
  videoWrap: {
    position: 'relative',
    width: '720px',
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
    height: '160px',
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

  const params = new URLSearchParams(window.location.search)
  const [wsUrl, setWsUrl] = useState(params.get('ws') ?? 'ws://127.0.0.1:8001')
  // URLSearchParams は + をスペースに変換する（unquote_plus 相当）ため
  // data パラムはファイルパスの + を保持するため手動パース（decodeURIComponent）する
  const rawDataParam = window.location.search.slice(1).split('&').find(p => p.startsWith('data='))
  const [dataDir, setDataDir] = useState(rawDataParam ? decodeURIComponent(rawDataParam.slice(5)) : '')
  const [mode, setMode] = useState(params.get('mode') ?? 'sync')
  const [delay, setDelay] = useState(Number(params.get('delay') ?? 0))
  const [status, setStatus] = useState('initializing...')
  const [connectionState, setConnectionState] = useState('initializing')
  const [log, setLog] = useState('')
  const [curAssessment, setCurAssessment] = useState(null)
  const [curSceneDesc, setCurSceneDesc] = useState('')
  const [sceneDescOpen, setSceneDescOpen] = useState(false)
  const [curDepthImagePath, setCurDepthImagePath] = useState(null)
  const [statusOpen, setStatusOpen] = useState(false)
  const [curVoicePath, setCurVoicePath] = useState(null)
  const [curAudioCues, setCurAudioCues] = useState([])
  const [curInfraredImagePath, setCurInfraredImagePath] = useState(null)
  const [isVoicePlaying, setIsVoicePlaying] = useState(false)
  const [curBeliefState, setCurBeliefState] = useState(null)
  const [curBlindSpots, setCurBlindSpots] = useState([])
  const [curInfraredAnalysis, setCurInfraredAnalysis] = useState(null)
  const [curDepthAnalysis, setCurDepthAnalysis] = useState(null)
  const [curTemporalAnalysis, setCurTemporalAnalysis] = useState(null)
  const [curErrors, setCurErrors] = useState([])
  const [modalityOpen, setModalityOpen] = useState(false)

  const wsRef = useRef(null)
  const logRef = useRef(null)
  const resultsRef = useRef([])
  const recvCountRef = useRef(0)
  const tOffsetRef = useRef(null)
  const haveVideoPlayedRef = useRef(false)
  const animIdRef = useRef(null)
  const lastVideoTimeRef = useRef(0)
  const lastAssessmentFrameRef = useRef(null)
  const lastDepthImagePathRef = useRef(null)

  useEffect(() => {
    let cancelled = false
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
      const timeDisplay = msg.frame_id ? `[${msg.frame_id}]` : `${msg.t.toFixed(2)}s`
      const risk = msg.assessment?.risk_level ?? '-'
      const action = msg.assessment?.action_type ?? '-'
      const line = `${timeDisplay} (+${lat.toFixed(2)}s) risk=${risk} action=${action}\n`
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

      for (let i = 0; i < criticalPoints.length; i++) {
        const cp = criticalPoints[i]
        if (!cp?.bbox || cp.bbox.length !== 4) continue
        const [bx1, by1, bx2, by2] = cp.bbox
        const px1 = bx1*w, py1 = by1*h, px2 = bx2*w, py2 = by2*h
        // region_id で突き合わせ（配列インデックスは filter で変わるため不使用）
        const isTarget = !!targetRegion && !!cp.region_id && cp.region_id === targetRegion
        const cpLabel = cp.region_id || `CP${i}`

        if (isTarget) {
          // target_region: 白色・5px・点線 + 白背景ラベル
          ctx.strokeStyle = '#ffffff'
          ctx.lineWidth = 5
          ctx.setLineDash([8, 4])
          ctx.strokeRect(px1-2, py1-2, px2-px1+4, py2-py1+4)
          ctx.setLineDash([])
          const label = `→ TARGET ${cpLabel} ${cp.severity ?? ''}`
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
          const label = `${cpLabel} ${cp.severity ?? ''}`
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
      setCurAudioCues([])
      setCurInfraredImagePath(null)
      setCurBeliefState(null)
      setCurBlindSpots([])
      setCurInfraredAnalysis(null)
      setCurDepthAnalysis(null)
      setCurTemporalAnalysis(null)
      setCurErrors([])
      setSceneDescOpen(false)
      setStatusOpen(false)
      lastAssessmentFrameRef.current = null

      const url = wsUrl.trim() + (dataDir.trim() ? `?data=${encodeURIComponent(dataDir.trim())}` : '')
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
        // error 状態（ディレクトリ不存在など）では再接続しない
        setConnectionState((prev) => {
          if (prev === 'error') return 'error'
          updateStatus('✗ disconnected — reconnecting in 3s...')
          setTimeout(() => {
            if (cancelled) return
            connectWS()
          }, 3000)
          return 'reconnecting'
        })
      }

      wsRef.current.onmessage = (ev) => {
        let msg
        try {
          msg = JSON.parse(ev.data)
        } catch (e) {
          console.warn('[WS] invalid JSON received:', e)
          return
        }

        // サーバーからのエラー通知（ディレクトリ不存在など）
        if (msg.error) {
          setConnectionState('error')
          updateStatus(`⚠ ${msg.error}`)
          wsRef.current?.close()
          return
        }

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
            // depth は null のフレームでも前回画像を維持する
            if (cur.depth_image_path) {
              lastDepthImagePathRef.current = cur.depth_image_path
            }
            setCurDepthImagePath(lastDepthImagePathRef.current)
            setCurVoicePath(cur.voice_path ?? null)
            setCurAudioCues(cur.audio_cues ?? [])
            setCurInfraredImagePath(cur.infrared_image_path ?? null)
            setCurBeliefState(cur.belief_state ?? null)
            setCurBlindSpots(cur.blind_spots ?? [])
            setCurInfraredAnalysis(cur.infrared_analysis ?? null)
            setCurDepthAnalysis(cur.depth_analysis ?? null)
            setCurTemporalAnalysis(cur.temporal_analysis ?? null)
            setCurErrors(cur.errors ?? [])

            // 音声再生（voice_path がある場合）
            if (cur.voice_path && audioRef.current) {
              audioRef.current.src = cur.voice_path
              audioRef.current.play().catch(() => {})
              setIsVoicePlaying(true)
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
            setCurAudioCues([])
            setCurInfraredImagePath(null)
            setCurBeliefState(null)
            setCurBlindSpots([])
            setCurInfraredAnalysis(null)
            setCurDepthAnalysis(null)
            setCurTemporalAnalysis(null)
            setCurErrors([])
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
      cancelled = true
      if (animIdRef.current) cancelAnimationFrame(animIdRef.current)
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      const videoEl = videoRef.current
      window.removeEventListener('resize', resizeCanvasToVideo)
      videoEl?.removeEventListener('loadedmetadata', resizeCanvasToVideo)
      videoEl?.removeEventListener('play', resizeCanvasToVideo)
      videoEl?.removeEventListener('play', onVideoPlayOnce)
    }
  }, [mode, delay, wsUrl, dataDir])

  // Log 自動スクロール: 新しいエントリが追加されるたびに最下部へ追随
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [log])

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
           connectionState === 'reconnecting' ? '⟳ RECONNECTING' :
           connectionState === 'error' ? '⚠ ERROR' : '● OFFLINE'}
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
          <audio
            ref={audioRef}
            style={{ display: 'none' }}
            onEnded={() => setIsVoicePlaying(false)}
          />
          {isVoicePlaying && (
            <button
              onClick={() => {
                audioRef.current?.pause()
                audioRef.current && (audioRef.current.currentTime = 0)
                setIsVoicePlaying(false)
              }}
              style={{
                ...styles.button,
                marginTop: '6px',
                background: '#5c0000',
                color: '#ff6b6b',
                border: '1px solid #ff4444',
              }}
            >
              ⏹ 音声停止
            </button>
          )}

          {/* Depth Map + Infrared 横並び（動画直下） */}
          {(curDepthImagePath || curInfraredImagePath) && (
            <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
              {curDepthImagePath && (
                <div style={{ flex: 1, background: '#111', borderRadius: '8px', overflow: 'hidden' }}>
                  <div style={{ fontSize: '11px', color: '#888', padding: '4px 8px' }}>Depth Map</div>
                  <div style={{ aspectRatio: '16/9', overflow: 'hidden' }}>
                    <img
                      src={curDepthImagePath}
                      alt="depth map"
                      style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'right center', display: 'block' }}
                    />
                  </div>
                </div>
              )}
              {!curDepthImagePath && <div style={{ flex: 1 }} />}
              {curInfraredImagePath && (
                <div style={{ flex: 1, background: '#111', borderRadius: '8px', overflow: 'hidden' }}>
                  <div style={{ fontSize: '11px', color: '#888', padding: '4px 8px' }}>Infrared</div>
                  <div style={{ aspectRatio: '16/9', overflow: 'hidden' }}>
                    <img
                      src={curInfraredImagePath}
                      alt="infrared"
                      style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          <div style={styles.hint}>
            ※ data/perception_results/frames/ から推論結果を読み込みます。
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
                <div style={{ fontSize:'13px', color:'#fff', borderTop:'1px solid #333', paddingTop:'6px', lineHeight:'1.5', maxHeight:'120px', overflowY:'auto' }}>
                  {curAssessment.reason}
                </div>
              </div>
            ) : (
              <div style={{ fontSize:'12px', color:'#555' }}>(no assessment)</div>
            )}
          </div>



          {/* Belief State Panel */}
          {curBeliefState?.hazard_tracks?.length > 0 && (
            <div style={styles.panel}>
              <h3 style={styles.panelH3}>
                🧠 Belief State
                {curBeliefState.overall_risk && (
                  <span style={{
                    marginLeft: '8px',
                    fontSize: '11px', fontWeight: '600',
                    padding: '2px 7px', borderRadius: '4px',
                    background: RISK_COLORS[curBeliefState.overall_risk]?.bg ?? '#222',
                    color: RISK_COLORS[curBeliefState.overall_risk]?.fg ?? '#aaa',
                  }}>
                    overall: {curBeliefState.overall_risk}
                  </span>
                )}
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {curBeliefState.hazard_tracks.map((track, i) => (
                  <div key={i} style={{
                    background: '#1a1a1a',
                    border: `1px solid ${SEVERITY_COLORS[track.severity] ?? '#333'}33`,
                    borderLeft: `3px solid ${SEVERITY_COLORS[track.severity] ?? '#555'}`,
                    borderRadius: '6px',
                    padding: '8px 10px',
                  }}>
                    {/* Track header: type + status + confidence */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px', flexWrap: 'wrap' }}>
                      <span style={{
                        fontSize: '11px', fontWeight: '700', color: '#ddd',
                        letterSpacing: '0.03em',
                      }}>
                        {track.hazard_type ?? track.hazard_id}
                      </span>
                      <span style={{
                        ...styles.smallBadge,
                        background: TEMPORAL_COLORS[track.status]?.bg ?? '#222',
                        color: TEMPORAL_COLORS[track.status]?.fg ?? '#888',
                      }}>
                        {track.status ?? 'unknown'}
                      </span>
                      <span style={{
                        ...styles.smallBadge,
                        background: SEVERITY_COLORS[track.severity] ? `${SEVERITY_COLORS[track.severity]}22` : '#222',
                        color: SEVERITY_COLORS[track.severity] ?? '#aaa',
                        border: `1px solid ${SEVERITY_COLORS[track.severity] ?? '#444'}55`,
                      }}>
                        {track.severity}
                      </span>
                      <span style={{ marginLeft: 'auto', fontSize: '10px', color: '#666' }}>
                        conf: {((track.confidence_score ?? 0) * 100).toFixed(0)}%
                      </span>
                    </div>

                    {/* Supporting modalities */}
                    {track.supporting_modalities?.length > 0 && (
                      <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', marginBottom: '6px' }}>
                        {track.supporting_modalities.map((mod, j) => (
                          <span key={j} style={{
                            fontSize: '10px', fontWeight: '600',
                            padding: '2px 6px', borderRadius: '3px',
                            background: MODALITY_COLORS[mod]?.bg ?? '#1a1a2a',
                            color: MODALITY_COLORS[mod]?.fg ?? '#aaaacc',
                          }}>
                            {mod}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Evidence */}
                    {track.evidence?.length > 0 && (
                      <ul style={{
                        margin: '0', paddingLeft: '14px',
                        fontSize: '11px', color: '#999', lineHeight: '1.5',
                        maxHeight: '60px', overflowY: 'auto',
                      }}>
                        {track.evidence.map((ev, j) => (
                          <li key={j}>{ev}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Errors Banner */}
          {curErrors.length > 0 && (
            <div style={{
              background: '#3a1a00', border: '1px solid #ff8800',
              borderRadius: '8px', padding: '8px 12px', marginBottom: '12px',
              fontSize: '11px', color: '#ffbb66', lineHeight: '1.6',
            }}>
              {curErrors.map((e, i) => <div key={i}>⚠️ {e}</div>)}
            </div>
          )}

          {/* Modality Details Panel */}
          <div style={styles.panel}>
            <h3
              style={{ ...styles.panelH3, cursor: 'pointer', userSelect: 'none' }}
              onClick={() => setModalityOpen(o => !o)}
            >
              🔍 Modality Details {modalityOpen ? '▾' : '▸'}
              {(curBlindSpots.length > 0 || curInfraredAnalysis?.hot_spots?.length > 0) && (
                <span style={{ marginLeft: '6px', fontSize: '10px', color: '#ff8844' }}>●</span>
              )}
            </h3>

            {modalityOpen && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
                {/* 🚫 Blind Spots */}
                <ModalitySection title="🚫 Blind Spots" defaultOpen={false} count={curBlindSpots.length}>
                  {curBlindSpots.length > 0 ? curBlindSpots.map((bs, i) => (
                    <div key={i} style={{
                      borderLeft: `3px solid ${SEVERITY_COLORS[bs.severity] ?? '#555'}`,
                      paddingLeft: '8px',
                      marginBottom: i < curBlindSpots.length - 1 ? '6px' : 0,
                    }}>
                      <div style={{ display: 'flex', gap: '6px', alignItems: 'center', marginBottom: '3px' }}>
                        <span style={{
                          ...styles.smallBadge,
                          background: (SEVERITY_COLORS[bs.severity] ?? '#aaa') + '22',
                          color: SEVERITY_COLORS[bs.severity] ?? '#aaa',
                          border: `1px solid ${SEVERITY_COLORS[bs.severity] ?? '#444'}55`,
                        }}>{bs.severity}</span>
                        <span style={{ fontSize: '11px', color: '#ccc', fontWeight: '600' }}>{bs.region_id}</span>
                      </div>
                      {bs.position && (
                        <div style={{ fontSize: '10px', color: '#666', marginBottom: '2px' }}>{bs.position}</div>
                      )}
                      <div style={{ fontSize: '11px', color: '#999', lineHeight: '1.4' }}>{bs.description}</div>
                    </div>
                  )) : <div style={{ fontSize: '11px', color: '#444' }}>(none)</div>}
                </ModalitySection>

                {/* 🔥 Infrared Hot Spots */}
                <ModalitySection
                  title="🔥 Infrared Hot Spots"
                  defaultOpen={false}
                  badge={curInfraredAnalysis?.overall_risk}
                  count={curInfraredAnalysis?.hot_spots?.length ?? 0}
                >
                  {curInfraredAnalysis?.hot_spots?.length > 0
                    ? curInfraredAnalysis.hot_spots.map((hs, i) => (
                        <div key={i} style={{
                          borderLeft: `3px solid ${SEVERITY_COLORS[hs.severity] ?? '#555'}`,
                          paddingLeft: '8px',
                          marginBottom: i < curInfraredAnalysis.hot_spots.length - 1 ? '6px' : 0,
                        }}>
                          <div style={{ display: 'flex', gap: '6px', alignItems: 'center', marginBottom: '3px' }}>
                            <span style={{
                              ...styles.smallBadge,
                              background: (SEVERITY_COLORS[hs.severity] ?? '#aaa') + '22',
                              color: SEVERITY_COLORS[hs.severity] ?? '#aaa',
                              border: `1px solid ${SEVERITY_COLORS[hs.severity] ?? '#444'}55`,
                            }}>{hs.severity}</span>
                            <span style={{ fontSize: '11px', color: '#ccc', fontWeight: '600' }}>{hs.region_id}</span>
                          </div>
                          <div style={{ fontSize: '11px', color: '#999', lineHeight: '1.4' }}>{hs.description}</div>
                        </div>
                      ))
                    : <div style={{ fontSize: '11px', color: '#444' }}>(none)</div>
                  }
                </ModalitySection>

                {/* ⏱️ Temporal */}
                <ModalitySection title="⏱️ Temporal" defaultOpen={false}>
                  {curTemporalAnalysis ? (
                    <div>
                      <div style={{ display: 'flex', gap: '6px', alignItems: 'center', marginBottom: '4px' }}>
                        <span style={{
                          ...styles.smallBadge,
                          background: curTemporalAnalysis.change_detected ? '#5c2200' : '#1a3300',
                          color: curTemporalAnalysis.change_detected ? '#ff8844' : '#66ee66',
                        }}>
                          {curTemporalAnalysis.change_detected ? '⚠ CHANGE' : '✓ NO CHANGE'}
                        </span>
                        <span style={{
                          ...styles.smallBadge,
                          background: RISK_COLORS[curTemporalAnalysis.overall_risk]?.bg ?? '#222',
                          color: RISK_COLORS[curTemporalAnalysis.overall_risk]?.fg ?? '#aaa',
                        }}>{curTemporalAnalysis.overall_risk}</span>
                        <span style={{ fontSize: '10px', color: '#666', marginLeft: 'auto' }}>
                          conf: {((curTemporalAnalysis.confidence_score ?? 0) * 100).toFixed(0)}%
                        </span>
                      </div>
                      {curTemporalAnalysis.changes?.length > 0 && (
                        <ul style={{ margin: '4px 0 0', paddingLeft: '14px', fontSize: '11px', color: '#999', lineHeight: '1.5' }}>
                          {curTemporalAnalysis.changes.map((c, i) => (
                            <li key={i}>{c.region_id}: {c.description}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  ) : <div style={{ fontSize: '11px', color: '#444' }}>(no data)</div>}
                </ModalitySection>

                {/* 📐 Depth Layers */}
                <ModalitySection
                  title="📐 Depth Layers"
                  defaultOpen={false}
                  badge={curDepthAnalysis?.overall_risk}
                >
                  {curDepthAnalysis?.depth_layers?.length > 0
                    ? curDepthAnalysis.depth_layers.map((layer, i) => (
                        <div key={i} style={{
                          display: 'grid', gridTemplateColumns: '40px 1fr', gap: '6px',
                          alignItems: 'start',
                          marginBottom: i < curDepthAnalysis.depth_layers.length - 1 ? '5px' : 0,
                        }}>
                          <span style={{
                            ...styles.smallBadge, textAlign: 'center',
                            background: DEPTH_ZONE_COLORS[layer.zone]?.bg ?? '#222',
                            color: DEPTH_ZONE_COLORS[layer.zone]?.fg ?? '#aaa',
                          }}>{layer.zone}</span>
                          <span style={{ fontSize: '11px', color: '#aaa', lineHeight: '1.4' }}>{layer.description}</span>
                        </div>
                      ))
                    : <div style={{ fontSize: '11px', color: '#444' }}>(no data)</div>
                  }
                </ModalitySection>
              </div>
            )}
          </div>

          {/* Audio Cues Panel */}
          {curAudioCues.length > 0 && (
            <div style={styles.panel}>
              <h3 style={styles.panelH3}>🔊 Audio Cues</h3>
              {curAudioCues.map((cue, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'flex-start', gap: '8px',
                  marginBottom: i < curAudioCues.length - 1 ? '6px' : 0,
                }}>
                  <span style={{
                    ...styles.smallBadge,
                    flexShrink: 0,
                    background: SEVERITY_COLORS[cue.severity] ? `${SEVERITY_COLORS[cue.severity]}22` : '#222',
                    color: SEVERITY_COLORS[cue.severity] ?? '#aaa',
                    border: `1px solid ${SEVERITY_COLORS[cue.severity] ?? '#444'}`,
                  }}>
                    {cue.severity ?? 'unknown'}
                  </span>
                  <div>
                    <div style={{ fontSize: '12px', color: '#eee', fontWeight: '600' }}>{cue.cue}</div>
                    {cue.evidence && (
                      <div style={{ fontSize: '11px', color: '#777', marginTop: '2px' }}>{cue.evidence}</div>
                    )}
                  </div>
                </div>
              ))}
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
              <div style={{ fontSize:'12px', color:'#bbb', lineHeight:'1.6', maxHeight:'160px', overflowY:'auto' }}>
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
                 connectionState === 'reconnecting' ? '⟳ RECONNECTING' :
           connectionState === 'error' ? '⚠ ERROR' : '● OFFLINE'}
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
                    データディレクトリ（省略時は data/）
                    <input
                      type="text"
                      value={dataDir}
                      onChange={(e) => setDataDir(e.target.value)}
                      placeholder="例: ~/data/result_1"
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
            <div ref={logRef} style={styles.log}>{log}</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
