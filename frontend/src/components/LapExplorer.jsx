import React, { useEffect, useMemo, useState } from 'react'
import { formatLapTime } from './CircuitMap'
import '../lapexplorer.css'

const YEARS = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018]
const FALLBACK_COLOR = '#71867e'
const JSON_CACHE_LIMIT = 180
const jsonRequestCache = new Map()

export async function fetchJson(url) {
  // The dashboard mounts adjacent pages independently. Keep both pending and
  // completed GETs in one short-lived browser cache so Team Lap, Optimal Lap
  // and Telemetry do not repeatedly parse the same FastF1 response.
  const cached = jsonRequestCache.get(url)
  if (cached) return cached

  const request = (async () => {
    const response = await fetch(url)
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) {
      const error = new Error(payload.error || `request failed (${response.status})`)
      error.status = response.status
      throw error
    }
    return payload
  })()

  jsonRequestCache.set(url, request)
  if (jsonRequestCache.size > JSON_CACHE_LIMIT) {
    jsonRequestCache.delete(jsonRequestCache.keys().next().value)
  }
  try {
    return await request
  } catch (error) {
    // Errors must never become sticky: retries should always reach the API.
    if (jsonRequestCache.get(url) === request) jsonRequestCache.delete(url)
    throw error
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

// Transient-failure resilience: the Vite proxy answers 500/502 while the backend
// restarts or a FastF1 spawn times out, so retry those but never client errors.
export async function fetchWithRetry(url, attempts = 3) {
  let lastError = null
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      return await fetchJson(url)
    } catch (error) {
      lastError = error
      if (error.status != null && error.status < 500) throw error
      if (attempt < attempts - 1) await sleep(1000 * 2 ** attempt)
    }
  }
  throw lastError
}

function driverColor(session, abbr) {
  const driver = (session?.drivers ?? []).find((entry) => entry.abbr === abbr)
  return driver?.teamColorHex || FALLBACK_COLOR
}

export function lapKey(entry) {
  return `${entry.year}:${entry.round}:${entry.session}:${entry.driver}:${entry.lap}`
}

function LapTimeChart({ session, selectedDrivers, showOutliers, xMode, onToggleXMode }) {
  const width = 960
  const height = 300
  const pad = { x: 62, y: 24, r: 16, b: 44 }

  const series = useMemo(() => selectedDrivers.map((abbr) => {
    const rows = (session.laps ?? [])
      .filter((lap) => lap.driver === abbr && lap.lapTimeS != null && (showOutliers || !lap.isOutlier))
      .sort((a, b) => a.lapNumber - b.lapNumber)
    let cumulative = 0
    const points = rows.map((lap) => {
      cumulative += lap.lapTimeS
      return { lap, x: xMode === 'lap' ? lap.lapNumber : cumulative }
    })
    return { abbr, color: driverColor(session, abbr), points }
  }).filter((entry) => entry.points.length), [session, selectedDrivers, showOutliers, xMode])

  const all = series.flatMap((entry) => entry.points)
  if (!all.length) return <p className="lx-empty">No timed laps for the current selection.</p>

  const xMin = Math.min(...all.map((p) => p.x))
  const xMax = Math.max(...all.map((p) => p.x))
  const yMin = Math.min(...all.map((p) => p.lap.lapTimeS)) - 0.4
  const yMax = Math.max(...all.map((p) => p.lap.lapTimeS)) + 0.4
  const px = (x) => pad.x + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * (width - pad.x - pad.r)
  const py = (v) => pad.y + (1 - (v - yMin) / (yMax - yMin)) * (height - pad.y - pad.b)
  const ticks = [0, 1, 2, 3, 4, 5, 6].map((i) => xMin + (i / 6) * (xMax - xMin))

  return <div className="lx-chart">
    <div className="lx-legend">
      {series.map((entry) => <span key={entry.abbr} style={{ color: entry.color }}>─●─ {entry.abbr}</span>)}
    </div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="lap time comparison">
      <g className="lx-grid">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const y = pad.y + f * (height - pad.y - pad.b)
          return <line key={f} x1={pad.x} y1={y} x2={width - pad.r} y2={y} />
        })}
        {ticks.map((t, i) => <line key={i} x1={px(t)} y1={pad.y} x2={px(t)} y2={height - pad.b} />)}
      </g>
      {series.map((entry) => <g key={entry.abbr}>
        <polyline
          fill="none"
          stroke={entry.color}
          strokeWidth="2"
          points={entry.points.map((p) => `${px(p.x)},${py(p.lap.lapTimeS)}`).join(' ')}
        />
        {entry.points.map((p) => <circle
          key={p.lap.lapNumber}
          cx={px(p.x)}
          cy={py(p.lap.lapTimeS)}
          r="3.4"
          fill={entry.color}
          stroke="#07100e"
          strokeWidth="1"
        >
          <title>{`${entry.abbr} · lap ${p.lap.lapNumber} · ${formatLapTime(p.lap.lapTimeS)}${p.lap.isPitLap ? ' · pit' : ''}${p.lap.compound ? ` · ${p.lap.compound.toLowerCase()}` : ''}`}</title>
        </circle>)}
      </g>)}
      <text className="lx-axis-label" transform={`translate(14 ${height / 2}) rotate(-90)`} textAnchor="middle">LAP TIME</text>
      {ticks.map((t, i) => <text key={i} className="lx-tick" x={px(t)} y={height - pad.b + 16} textAnchor="middle">{xMode === 'lap' ? Math.round(t) : Math.round(t / 60)}{xMode === 'time' ? 'm' : ''}</text>)}
    </svg>
    <button className="lx-axis-btn" onClick={onToggleXMode}>TIME / LAP NUMBER</button>
  </div>
}

const TYRE_COLORS = {
  SOFT: '#e10600',
  MEDIUM: '#ffd12e',
  HARD: '#e8e8e8',
  INTERMEDIATE: '#43b02a',
  WET: '#0067ad',
}

function TyreBadge({ compound }) {
  if (!compound) return null
  const color = TYRE_COLORS[compound] ?? '#71867e'
  return <i className="lx-tyre" style={{ borderColor: color, color }} title={compound.toLowerCase()}>{compound[0]}</i>
}

function LapTable({ session, selectedDrivers, selectedLaps, onToggleLap }) {
  const byKey = useMemo(() => {
    const map = new Map()
    for (const lap of session.laps ?? []) map.set(`${lap.driver}:${lap.lapNumber}`, lap)
    return map
  }, [session])

  const maxLap = useMemo(() => Math.max(0, ...selectedDrivers.flatMap((abbr) =>
    (session.laps ?? []).filter((lap) => lap.driver === abbr).map((lap) => lap.lapNumber))), [session, selectedDrivers])

  if (!maxLap) return null
  const isSelected = (abbr, lapNumber) => selectedLaps.some((entry) =>
    entry.driver === abbr && entry.lap === lapNumber && entry.round === session.event.round && entry.year === session.event.year && entry.session === session.session)

  return <div className="lx-table-wrap">
    <table className="lx-table">
      <thead>
        <tr>
          <th className="lx-table-head">Lap:</th>
          {Array.from({ length: maxLap }, (_, i) => <th key={i}>{i + 1}</th>)}
        </tr>
      </thead>
      <tbody>
        {selectedDrivers.map((abbr) => <tr key={abbr}>
          <td className="lx-table-head" style={{ color: driverColor(session, abbr) }}>{abbr}</td>
          {Array.from({ length: maxLap }, (_, i) => {
            const lap = byKey.get(`${abbr}:${i + 1}`)
            if (!lap || lap.lapTimeS == null || !lap.isAccurate) {
              return <td key={i}><button className="lx-cell na" disabled>N/A<TyreBadge compound={lap?.compound} /></button></td>
            }
            const classes = ['lx-cell']
            if (lap.isPitLap) classes.push('pit')
            if (lap.isOutlier) classes.push('outlier')
            if (isSelected(abbr, lap.lapNumber)) classes.push('sel')
            return <td key={i}>
              <button className={classes.join(' ')} onClick={() => onToggleLap(abbr, lap)} title={`${abbr} lap ${lap.lapNumber}${lap.isPitLap ? ' · pit lap' : ''}`}>
                {formatLapTime(lap.lapTimeS)}<TyreBadge compound={lap.compound} />
              </button>
            </td>
          })}
        </tr>)}
      </tbody>
    </table>
  </div>
}

export function LapExplorer({ selectedLaps, onLapsChange, onOpenTelemetry, onSelectionChange }) {
  const [year, setYear] = useState(2024)
  const [events, setEvents] = useState(null)
  const [round, setRound] = useState(null)
  const [sessionName, setSessionName] = useState(null)
  const [session, setSession] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedDrivers, setSelectedDrivers] = useState([])
  const [showOutliers, setShowOutliers] = useState(false)
  const [xMode, setXMode] = useState('lap')
  const [reloadKey, setReloadKey] = useState(0)

  useEffect(() => {
    let cancelled = false
    setEvents(null); setRound(null); setSessionName(null); setSession(null); setSelectedDrivers([])
    fetchWithRetry(`/api/f1/events?year=${year}`)
      .then((data) => {
        if (cancelled) return
        setEvents(data.events)
        const last = data.events[data.events.length - 1]
        if (last) {
          setRound(last.round)
          setSessionName(last.sessions.includes('Race') ? 'Race' : (last.sessions[last.sessions.length - 1] ?? null))
        }
      })
      .catch((err) => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [year])

  useEffect(() => {
    if (!round || !sessionName) return
    onSelectionChange?.({
      year,
      round,
      sessionName,
      event: (events ?? []).find((entry) => entry.round === round) ?? null,
    })
  }, [year, round, sessionName, events])

  useEffect(() => {
    if (!round || !sessionName) return undefined
    let cancelled = false
    setLoading(true); setError(null)
    fetchWithRetry(`/api/f1/session?year=${year}&round=${round}&session=${encodeURIComponent(sessionName)}`)
      .then((data) => {
        if (cancelled) return
        setSession(data)
        const ranked = [...data.drivers].sort((a, b) => (a.position ?? 99) - (b.position ?? 99))
        setSelectedDrivers(ranked.slice(0, 2).map((driver) => driver.abbr))
      })
      .catch((err) => { if (!cancelled) setError(err.message) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [year, round, sessionName, reloadKey])

  const event = (events ?? []).find((entry) => entry.round === round)

  const toggleDriver = (abbr) => setSelectedDrivers((prev) =>
    prev.includes(abbr) ? prev.filter((code) => code !== abbr) : [...prev, abbr])

  const toggleLap = (abbr, lap) => {
    const entry = { year, round, session: sessionName, driver: abbr, lap: lap.lapNumber, color: driverColor(session, abbr) }
    const exists = selectedLaps.some((sel) => lapKey(sel) === lapKey(entry))
    onLapsChange(exists ? selectedLaps.filter((sel) => lapKey(sel) !== lapKey(entry)) : [...selectedLaps, entry])
  }

  const compareFastest = () => {
    const entries = selectedDrivers.map((abbr) => {
      const clean = (session.laps ?? []).filter((lap) => lap.driver === abbr && lap.lapTimeS != null && !lap.isOutlier)
      if (!clean.length) return null
      const fastest = clean.reduce((best, lap) => (lap.lapTimeS < best.lapTimeS ? lap : best))
      return { year, round, session: sessionName, driver: abbr, lap: fastest.lapNumber, color: driverColor(session, abbr) }
    }).filter(Boolean)
    if (entries.length) {
      onLapsChange(entries)
      onOpenTelemetry()
    }
  }

  return <section className="lx-root">
    <div className="lx-panel-head">
      <span>LAP EXPLORER / FASTF1 2018–2025</span>
      <em>SESSION VIEW</em>
    </div>

    <div className="lx-body">
      <aside className="lx-side">
        <select className="lx-select" value={year} onChange={(e) => setYear(Number(e.target.value))}>
          {YEARS.map((entry) => <option key={entry} value={entry}>{entry}</option>)}
        </select>
        <select className="lx-select" value={round ?? ''} onChange={(e) => setRound(Number(e.target.value))} disabled={!events}>
          {!events && <option value="">Loading calendar…</option>}
          {(events ?? []).map((entry) => <option key={entry.round} value={entry.round}>{entry.name}</option>)}
        </select>
        <select className="lx-select" value={sessionName ?? ''} onChange={(e) => setSessionName(e.target.value)} disabled={!event}>
          {(event?.sessions ?? []).map((name) => <option key={name} value={name}>{name}</option>)}
        </select>

        <div className="lx-chips">
          {(session?.drivers ?? []).map((driver) => {
            const active = selectedDrivers.includes(driver.abbr)
            const color = driver.teamColorHex || FALLBACK_COLOR
            return <button
              key={driver.abbr}
              className={`lx-chip ${active ? 'active' : ''}`}
              style={{ borderColor: color, background: active ? color : 'transparent', color: active ? '#07100e' : color }}
              onClick={() => toggleDriver(driver.abbr)}
              title={`${driver.name} · ${driver.team ?? ''}`}
            >
              {driver.abbr}
            </button>
          })}
          {!session && !loading && <p className="lx-empty">No session loaded.</p>}
        </div>

        <button className="lx-compare" onClick={compareFastest} disabled={!selectedDrivers.length}>
          COMPARE FASTEST LAPS
        </button>
      </aside>

      <div className="lx-main">
        {loading && <div className="lx-loading">
          <span className="lx-spinner" />
          FETCHING SESSION DATA — the first load of a race downloads official timing and can take up to a minute
        </div>}
        {error && <div className="lx-error">
          {error}
          <button onClick={() => { setError(null); setReloadKey((key) => key + 1) }}>RETRY</button>
        </div>}
        {!loading && session && <>
          <LapTimeChart
            session={session}
            selectedDrivers={selectedDrivers}
            showOutliers={showOutliers}
            xMode={xMode}
            onToggleXMode={() => setXMode((mode) => (mode === 'lap' ? 'time' : 'lap'))}
          />
          <label className="lx-switch">
            <input type="checkbox" checked={showOutliers} onChange={(e) => setShowOutliers(e.target.checked)} />
            <i />
            Show outliers
          </label>
        </>}
      </div>
    </div>

    {session && !loading && selectedDrivers.length > 0 && <LapTable
      session={session}
      selectedDrivers={selectedDrivers}
      selectedLaps={selectedLaps}
      onToggleLap={toggleLap}
    />}
    <p className="lx-hint">Click a lap time to add it to the telemetry comparison · pit laps are tinted · N/A marks laps without an accurate time.</p>
  </section>
}
