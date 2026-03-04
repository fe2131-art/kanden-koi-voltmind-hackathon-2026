import React, { useEffect, useRef, useState } from 'react'

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
    width: '340px',
    flex: '0 0 340px',
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
    height: '360px',
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
}

function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  const [wsUrl, setWsUrl] = useState('ws://127.0.0.1:8001')
  const [mode, setMode] = useState('sync')
  const [delay, setDelay] = useState(0.7)
  const [status, setStatus] = useState('initializing...')
  const [hud, setHud] = useState('(waiting for data/perception_results.json)')
  const [log, setLog] = useState('')

  const wsRef = useRef(null)
  const resultsRef = useRef([])
  const recvCountRef = useRef(0)
  const tOffsetRef = useRef(null)
  const haveVideoPlayedRef = useRef(false)
  const animIdRef = useRef(null)
  const lastVideoTimeRef = useRef(0)

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
      // msg.t が Unix timestamp（絶対値）なので、表示用に相対時刻に変換
      // または obs_id を表示
      const timeDisplay = msg.obs_id ? `[${msg.obs_id}]` : `${msg.t.toFixed(2)}s`
      const line = `${timeDisplay} (+${lat.toFixed(2)}s): ${msg.text ?? '(no text)'}\n`
      setLog((prev) => {
        const combined = prev + line
        if (combined.length > 25000) {
          return combined.slice(-18000)
        }
        return combined
      })
    }

    const drawDetections = (dets, w, h) => {
      ctx.clearRect(0, 0, w, h)
      if (!dets?.length) return

      ctx.lineWidth = 3
      ctx.font = '12px system-ui, -apple-system, Segoe UI, sans-serif'

      for (const d of dets) {
        const b = d.bbox
        if (!b || b.length !== 4) continue

        const x1 = b[0] * w
        const y1 = b[1] * h
        const x2 = b[2] * w
        const y2 = b[3] * h

        ctx.strokeStyle = 'lime'
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

        const label = `${d.label ?? 'obj'} ${(d.score ?? 0).toFixed(2)}`
        const pad = 4
        const th = 16
        const tw = ctx.measureText(label).width

        ctx.fillStyle = 'rgba(0,0,0,0.6)'
        ctx.fillRect(x1, Math.max(0, y1 - th), tw + pad * 2, th)
        ctx.fillStyle = 'white'
        ctx.fillText(label, x1 + pad, Math.max(12, y1 - 4))
      }
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

      const url = wsUrl.trim()
      wsRef.current = new WebSocket(url)

      wsRef.current.onopen = () => updateStatus('OPEN')
      wsRef.current.onerror = () => updateStatus('ERROR (check console)')
      wsRef.current.onclose = () => updateStatus('CLOSED')

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
          const lat = typeof cur.t_sent === 'number' ? cur.t_sent - cur.t : 0
          setHud(
            `text: ${cur.text ?? ''}\nframe: ${cur.obs_id ?? 'unknown'}\nt(abs): ${cur.t?.toFixed?.(2) ?? cur.t}\nt(video): ${cur.t_adj?.toFixed?.(2) ?? cur.t_adj}\nlatency: ${lat.toFixed(2)}s\nmode: ${mode}  D=${D.toFixed(2)}s`
          )
          drawDetections(cur.detections, w, h)
        } else {
          setHud('(no data yet)')
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
      <div style={styles.layout}>
        <div style={styles.left}>
          <div style={styles.videoWrap}>
            <video
              ref={videoRef}
              controls
              src="/videos/free-video7-rice-cafinet.mp4"
              style={styles.video}
            ></video>
            <canvas ref={canvasRef} style={styles.canvas}></canvas>
          </div>
          <div style={styles.hint}>
            ※ data/perception_results.json から推論結果を読み込みます。
          </div>
        </div>

        <div style={styles.right}>
          {/* Status Panel */}
          <div style={styles.panel}>
            <h3 style={styles.panelH3}>Status</h3>
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

          {/* HUD Panel */}
          <div style={styles.panel}>
            <h3 style={styles.panelH3}>HUD</h3>
            <div id="hud" style={styles.kv}>
              {hud}
            </div>
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
