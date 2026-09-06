import React, { useEffect, useMemo, useRef, useState } from 'react'
import { formatLapTime } from './CircuitMap'
import { fetchJson, lapKey } from './LapExplorer'
import { TrackRaceMap, timeAtDistance, distanceAtTime } from './TrackRaceMap'

const WIDTH = 960
const HEIGHT = 150
const PAD = { x: 50, y: 12, r: 14, b: 20 }

const fmtClock = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, '0')}`

function TraceChart({ label, traces, yMin, yMax, yFmt, markers, cursor, step, xMax }) {
  const px = (d) => PAD.x + (d / xMax) * (WIDTH - PAD.x - PAD.r)
  const clamp = (v) => Math.min(Math.max(v, yMin), yMax)
  const py = (v) => PAD.y + (1 - (clamp(v) - yMin) / (yMax - yMin)) * (HEIGHT - PAD.y - PAD.b)

  const pathFor = (trace) => {
    if (step) {
      let d = ''
      trace.values.forEach((v, i) => {
        const x = px(trace.distances[i]).toFixed(1)
        const y = py(v).toFixed(1)
        d += i === 0 ? `M${x} ${y}` : `H${x} V${y}`
      })
      return d
    }
    return trace.values.map((v, i) => `${i === 0 ? 'M' : 'L'}${px(trace.distances[i]).toFixed(1)} ${py(v).toFixed(1)}`).join(' ')
  }

  return <div className="tc-chart">
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={label}>
      <g className="lx-grid">
        <line x1={PAD.x} y1={PAD.y} x2={WIDTH - PAD.r} y2={PAD.y} />
        <line x1={PAD.x} y1={(HEIGHT - PAD.b + PAD.y) / 2} x2={WIDTH - PAD.r} y2={(HEIGHT - PAD.b + PAD.y) / 2} />
        <line x1={PAD.x} y1={HEIGHT - PAD.b} x2={WIDTH - PAD.r} y2={HEIGHT - PAD.b} />
      </g>
      {(markers ?? []).map((m) => <g key={m.label}>
        <line className="tc-sector" x1={px(m.d)} y1={PAD.y} x2={px(m.d)} y2={HEIGHT - PAD.b} />
        <text className="tc-sector-label" x={px(m.d) + 4} y={PAD.y + 10}>{m.label}</text>
      </g>)}
      {traces.map((trace, i) => step
        ? <path key={i} d={pathFor(trace)} fill="none" stroke={trace.color} strokeWidth="1.8" />
        : <path key={i} d={pathFor(trace)} fill="none" stroke={trace.color} strokeWidth="1.8" />)}
      {cursor != null && <line className="tc-cursor" x1={px(cursor)} y1={PAD.y} x2={px(cursor)} y2={HEIGHT - PAD.b} />}
      <text className="lx-tick" x={PAD.x - 6} y={PAD.y + 4} textAnchor="end">{yFmt(yMax)}</text>
      <text className="lx-tick" x={PAD.x - 6} y={HEIGHT - PAD.b} textAnchor="end">{yFmt(yMin)}</text>
      <text className="tc-chart-label" x={PAD.x + 6} y={HEIGHT - 6}>{label}</text>
    </svg>
  </div>
}

export function TelemetryCompare({ selectedLaps, onLapsChange }) {
  const [telemetry, setTelemetry] = useState({})
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [cursorT, setCursorT] = useState(0)
  const [trackmap, setTrackmap] = useState(null)
  const requested = useRef(new Set())
  const trackmapRequested = useRef(new Set())

  const keysKey = selectedLaps.map(lapKey).join('|')
  const sessionKey = selectedLaps.length
    ? `${selectedLaps[0].year}:${selectedLaps[0].round}:${selectedLaps[0].session}`
    : ''

  const loadLap = (sel) => {
    const key = lapKey(sel)
    requested.current.add(key)
    setTelemetry((prev) => ({ ...prev, [key]: { loading: true } }))
    fetchJson(`/api/f1/telemetry?year=${sel.year}&round=${sel.round}&session=${encodeURIComponent(sel.session)}&driver=${sel.driver}&lap=${sel.lap}`)
      .then((data) => setTelemetry((prev) => ({ ...prev, [key]: data })))
      .catch((err) => setTelemetry((prev) => ({ ...prev, [key]: { error: err.message } })))
  }

  useEffect(() => {
    selectedLaps.forEach((sel) => {
      const key = lapKey(sel)
      if (requested.current.has(key)) return
      loadLap(sel)
    })
  }, [keysKey])

  useEffect(() => {
    if (!sessionKey || trackmapRequested.current.has(sessionKey)) return
    trackmapRequested.current.add(sessionKey)
    const first = selectedLaps[0]
    setTrackmap({ loading: true })
    fetchJson(`/api/f1/trackmap?year=${first.year}&round=${first.round}&session=${encodeURIComponent(first.session)}`)
      .then((data) => setTrackmap(data.points?.length ? data : { error: data.error || 'no position data' }))
      .catch((err) => setTrackmap({ error: err.message }))
  }, [sessionKey])

  useEffect(() => {
    setCursorT(0)
    setPlaying(false)
  }, [keysKey])

  const loaded = selectedLaps
    .map((sel) => ({ sel, data: telemetry[lapKey(sel)] }))
    .filter((entry) => entry.data?.trace)

  const reference = useMemo(() => loaded.length
    ? loaded.reduce((best, entry) => (entry.data.lapTimeS < best.data.lapTimeS ? entry : best))
    : null, [keysKey, telemetry])

  const total = useMemo(() => (loaded.length
    ? Math.max(...loaded.map((entry) => entry.data.lapTimeS))
    : 0), [keysKey, telemetry])

  useEffect(() => {
    if (total && cursorT >= total) setPlaying(false)
  }, [cursorT, total])

  useEffect(() => {
    if (!playing || !total) return undefined
    let raf
    let last = performance.now()
    const tick = (now) => {
      const dt = (now - last) / 1000
      last = now
      setCursorT((t) => Math.min(total, t + dt * speed))
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, speed, total])

  if (!selectedLaps.length) {
    return <section className="lx-root tc-root">
      <div className="lx-panel-head"><span>TELEMETRY COMPARISON</span><em>0 LAPS</em></div>
      <p className="lx-empty">Select laps in the TRACK tab lap table, or press COMPARE FASTEST LAPS there.</p>
    </section>
  }

  const refTrace = reference?.data.trace ?? null
  const grid = refTrace?.distance ?? []
  const xMax = Math.max(1, ...loaded.map((entry) => entry.data.trace.distance[entry.data.trace.distance.length - 1]))
  const markers = refTrace ? [
    { d: reference.data.sectorMarkers?.s2Distance, label: 'S2' },
    { d: reference.data.sectorMarkers?.s3Distance, label: 'S3' },
  ].filter((m) => m.d != null) : []
  const cursor = refTrace && (playing || cursorT > 0) ? distanceAtTime(refTrace, cursorT) : null

  const maxRpm = Math.max(1, ...loaded.flatMap((entry) => entry.data.trace.rpm))

  const charts = refTrace ? [
    {
      label: 'SPEED (KM/H)', yMin: 0, yMax: 360, yFmt: (v) => `${Math.round(v)}`,
      traces: loaded.map((entry) => ({ color: entry.sel.color, distances: entry.data.trace.distance, values: entry.data.trace.speed })),
    },
    {
      label: 'DELTA (S)', yFmt: (v) => `${v > 0 ? '+' : ''}${v.toFixed(1)}`,
      yMin: Math.min(-0.5, ...loaded.map((entry) => Math.min(...grid.map((d) => timeAtDistance(entry.data.trace, d) - timeAtDistance(refTrace, d))))),
      yMax: Math.max(0.5, ...loaded.map((entry) => Math.max(...grid.map((d) => timeAtDistance(entry.data.trace, d) - timeAtDistance(refTrace, d))))),
      traces: loaded.map((entry) => ({
        color: entry.sel.color,
        distances: grid,
        values: grid.map((d) => timeAtDistance(entry.data.trace, d) - timeAtDistance(refTrace, d)),
      })),
    },
    {
      label: 'THROTTLE (%)', yMin: 0, yMax: 100, yFmt: (v) => `${Math.round(v)}`,
      traces: loaded.map((entry) => ({ color: entry.sel.color, distances: entry.data.trace.distance, values: entry.data.trace.throttle })),
    },
    {
      label: 'BRAKE', yMin: 0, yMax: 1, yFmt: (v) => (v ? 'ON' : 'OFF'), step: true,
      traces: loaded.map((entry) => ({ color: entry.sel.color, distances: entry.data.trace.distance, values: entry.data.trace.brake.map((b) => (b ? 1 : 0)) })),
    },
    {
      label: 'RPM (%)', yMin: 0, yMax: 100, yFmt: (v) => `${Math.round(v)}`,
      traces: loaded.map((entry) => ({ color: entry.sel.color, distances: entry.data.trace.distance, values: entry.data.trace.rpm.map((r) => (r / maxRpm) * 100) })),
    },
    {
      label: 'DRS', yMin: 0, yMax: 1, yFmt: (v) => (v ? 'OPEN' : 'SHUT'), step: true,
      traces: loaded.map((entry) => ({ color: entry.sel.color, distances: entry.data.trace.distance, values: entry.data.trace.drs.map((v) => (v >= 8 ? 1 : 0)) })),
    },
    {
      label: 'GEAR', yMin: 0, yMax: 8, yFmt: (v) => `${Math.round(v)}`, step: true,
      traces: loaded.map((entry) => ({ color: entry.sel.color, distances: entry.data.trace.distance, values: entry.data.trace.gear })),
    },
  ] : []

  return <section className="lx-root tc-root">
    <div className="lx-panel-head"><span>TELEMETRY COMPARISON</span><em>{loaded.length} LAP{loaded.length === 1 ? '' : 'S'}</em></div>

    <div className="tc-laps">
      {selectedLaps.map((sel) => {
        const data = telemetry[lapKey(sel)]
        const delta = data?.lapTimeS != null && reference ? data.lapTimeS - reference.data.lapTimeS : null
        return <div key={lapKey(sel)} className="tc-lap-card" style={{ borderColor: sel.color }}>
          <i style={{ background: sel.color }} />
          <b>{sel.driver}</b>
          <span>LAP {sel.lap}</span>
          {data?.loading && <em>loading…</em>}
          {data?.error && <em className="tc-err">{data.error}</em>}
          {data?.error && <button className="tc-retry" onClick={() => loadLap(sel)}>RETRY</button>}
          {data?.trace && <strong>{formatLapTime(data.lapTimeS)}</strong>}
          {delta != null && delta > 0.0005 && <em className="tc-delta">+{delta.toFixed(3)}s</em>}
          <button onClick={() => onLapsChange(selectedLaps.filter((other) => lapKey(other) !== lapKey(sel)))} title="remove lap">✕</button>
        </div>
      })}
    </div>

    {trackmap?.loading && <div className="lx-loading">
      <span className="lx-spinner" />
      LOADING CIRCUIT MAP
    </div>}
    {trackmap?.points?.length > 0 && <TrackRaceMap trackmap={trackmap} loadedLaps={loaded} cursorT={cursorT} />}

    <div className="tm-transport">
      <button onClick={() => setPlaying((p) => !p)} disabled={!reference}>{playing ? 'PAUSE' : 'PLAY'}</button>
      <button onClick={() => { setPlaying(false); setCursorT(0) }} disabled={!reference}>RESTART</button>
      <input
        className="tm-scrubber"
        type="range"
        min={0}
        max={total || 1}
        step={0.1}
        value={Math.min(cursorT, total || 1)}
        onChange={(e) => setCursorT(Number(e.target.value))}
        disabled={!reference}
      />
      <span className="tm-readout">{fmtClock(cursorT)} / {fmtClock(total)}</span>
      <div className="tm-speeds">
        {[0.5, 1, 2, 4].map((v) => <button
          key={v}
          className={speed === v ? 'tm-active' : ''}
          onClick={() => setSpeed(v)}
        >{v}x</button>)}
      </div>
    </div>

    {selectedLaps.some((sel) => telemetry[lapKey(sel)]?.loading) && <div className="lx-loading">
      <span className="lx-spinner" />
      FETCHING LAP TELEMETRY — first load of a lap can take a moment
    </div>}
    {selectedLaps.some((sel) => telemetry[lapKey(sel)]?.error) && <div className="lx-error">
      {selectedLaps.map((sel) => telemetry[lapKey(sel)]?.error).filter(Boolean).join(' · ')}
    </div>}

    <div className="tc-charts">
      {charts.map((chart) => <TraceChart
        key={chart.label}
        label={chart.label}
        traces={chart.traces}
        yMin={chart.yMin}
        yMax={chart.yMax}
        yFmt={chart.yFmt}
        step={chart.step}
        markers={markers}
        cursor={cursor}
        xMax={xMax}
      />)}
    </div>
  </section>
}
