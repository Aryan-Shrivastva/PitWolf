import React, { useEffect, useMemo, useRef, useState } from 'react'
import { fetchEnergyLap } from '../lib/f1api'
import '../simulationreplay.css'

const YEARS = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018]
const SPEEDS = [1, 2, 4, 8]
const MAX_VISUAL_SELECTIONS = 4
const SESSION_LABELS = {
  'Practice 1': 'FP1',
  'Practice 2': 'FP2',
  'Practice 3': 'FP3',
  'Sprint Qualifying': 'Sprint Qualifying',
  Sprint: 'Sprint Race',
  Qualifying: 'Qualifying',
  Race: 'Race',
}
const FALLBACK_COLORS = ['#63e6be', '#ff7043', '#a9bfff', '#ffbf69', '#f472b6', '#2dd4bf']

function requestJson(url) {
  return fetch(url).then(async (response) => {
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) throw new Error(payload.error || `request failed (${response.status})`)
    return payload
  })
}

function postJson(url, body) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(async (response) => {
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) throw new Error(payload.error || `request failed (${response.status})`)
    return payload
  })
}

function clock(seconds) {
  if (!Number.isFinite(seconds)) return '—'
  const minutes = Math.floor(seconds / 60)
  return `${minutes}:${(seconds - minutes * 60).toFixed(3).padStart(6, '0')}`
}

function displayDriverName(driver) {
  return String(driver?.name || driver?.driver || '—').replace(/^[A-Z]\s+/, '')
}

function displayPosition(driver, fallback) {
  return /^\d+$/.test(String(driver?.classifiedPosition ?? '')) ? driver.classifiedPosition : fallback
}

function projection(points, height = 520, padding = 46) {
  if (!points?.length) return null
  const normalWidth = Math.max(1, Math.max(...points.map((point) => point.x)) - Math.min(...points.map((point) => point.x)))
  const normalHeight = Math.max(1, Math.max(...points.map((point) => point.y)) - Math.min(...points.map((point) => point.y)))
  const viewportWidth = 760
  const targetAspect = 1.45
  // FastF1 circuit coordinates have no presentation orientation. Rotate the
  // whole recorded coordinate system when it better matches the player, while
  // preserving every real relative position and corner location.
  const rotate = Math.abs(Math.log((normalWidth / normalHeight) / targetAspect))
    > Math.abs(Math.log((normalHeight / normalWidth) / targetAspect))
  const orient = (x, y) => rotate ? { x: y, y: -x } : { x, y }
  const oriented = points.map((point) => orient(point.x, point.y))
  const xs = oriented.map((point) => point.x)
  const ys = oriented.map((point) => point.y)
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys)
  // Every circuit gets the same presentation viewport. Fit each real track
  // inside it instead of letting a long/narrow circuit expand its SVG viewBox
  // and make the whole map appear smaller than Barcelona.
  const scale = Math.min(
    (viewportWidth - padding * 2) / Math.max(1, maxX - minX),
    (height - padding * 2) / Math.max(1, maxY - minY),
  )
  const width = viewportWidth
  const ox = (width - (maxX - minX) * scale) / 2
  const oy = padding
  const project = (x, y) => {
    const point = orient(x, y)
    return { x: ox + (point.x - minX) * scale, y: height - oy - (point.y - minY) * scale }
  }
  return { width, height, project, outline: points.map((point) => project(point.x, point.y)) }
}

function pointAtDistance(points, distance) {
  if (!points?.length || !Number.isFinite(Number(distance))) return null
  return points.reduce((nearest, point) => (
    Math.abs(Number(point.d) - Number(distance)) < Math.abs(Number(nearest.d) - Number(distance)) ? point : nearest
  ), points[0])
}

function pointsForZone(points, startD, endD) {
  if (!points?.length || startD == null || endD == null) return []
  if (startD <= endD) return points.filter((point) => Number(point.d) >= startD && Number(point.d) <= endD)
  return [...points.filter((point) => Number(point.d) >= startD), ...points.filter((point) => Number(point.d) <= endD)]
}

function interpolateFrame(frames, time) {
  if (!frames?.length) return null
  if (time <= frames[0].t) return frames[0]
  if (time >= frames[frames.length - 1].t) return frames[frames.length - 1]
  let low = 0
  let high = frames.length - 1
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (frames[middle].t < time) low = middle + 1
    else high = middle
  }
  const next = frames[low]
  const previous = frames[Math.max(0, low - 1)]
  const span = Math.max(.001, next.t - previous.t)
  const fraction = Math.max(0, Math.min(1, (time - previous.t) / span))
  const nextByDriver = new Map(next.cars.map((car) => [car.driver, car]))
  return {
    t: time,
    cars: previous.cars.map((car) => {
      const later = nextByDriver.get(car.driver)
      if (!later) return car
      return { ...car, x: car.x + (later.x - car.x) * fraction, y: car.y + (later.y - car.y) * fraction }
    }),
  }
}

function trackDistance(point, trackPoints) {
  if (!point || !trackPoints?.length) return null
  let nearest = trackPoints[0]
  let nearestDistance = Number.POSITIVE_INFINITY
  for (const candidate of trackPoints) {
    const dx = Number(candidate.x) - Number(point.x)
    const dy = Number(candidate.y) - Number(point.y)
    const distance = dx * dx + dy * dy
    if (distance < nearestDistance) {
      nearest = candidate
      nearestDistance = distance
    }
  }
  return Number(nearest.d)
}

function selectedGapRows(frame, trackmap, drivers, selectedDrivers, lapTimeS) {
  const selected = new Set(selectedDrivers)
  const positions = (frame?.cars ?? [])
    .map((car) => ({ ...car, d: trackDistance(car, trackmap?.points) }))
    .filter((car) => Number.isFinite(Number(car.timingPosition)) || Number.isFinite(car.d))
  if (!positions.length) return []
  // The P-number must come from recorded lap timing. GPS/map distance is only
  // retained as a last-resort gap visualisation, never as a substitute for an
  // official position (it wraps at the finish line and can reverse the order).
  const ordered = [...positions].sort((left, right) => {
    const leftPosition = Number(left.timingPosition)
    const rightPosition = Number(right.timingPosition)
    if (Number.isFinite(leftPosition) && Number.isFinite(rightPosition)) return leftPosition - rightPosition
    if (Number.isFinite(leftPosition)) return -1
    if (Number.isFinite(rightPosition)) return 1
    return right.d - left.d
  })
  const positionByDriver = new Map(ordered.map((car, index) => [car.driver, Number.isFinite(Number(car.timingPosition)) ? Number(car.timingPosition) : index + 1]))
  const leader = ordered.find((car) => Number(car.timingPosition) === 1) ?? ordered[0]
  const length = Number(trackmap?.length ?? trackmap?.points?.at(-1)?.d ?? 0)
  const lapSeconds = Number(lapTimeS)
  return ordered
    .filter((car) => selected.has(car.driver))
    .map((car) => ({
      ...car,
      position: positionByDriver.get(car.driver),
      gapS: Number.isFinite(car.timingGapS)
        ? car.timingGapS
        // When the TimingData stream has not published a gap yet (for
        // example on the grid), a dash is more truthful than a GPS-derived
        // number. Map distance is never presented as an official gap.
        : car.timingSource === 'OFFICIAL_TIMING_STREAM' || car.timingSource === 'RECORDED_GRID_POSITION'
          ? null
          : car.driver === leader.driver || !length || !Number.isFinite(lapSeconds) || !Number.isFinite(car.d) || !Number.isFinite(leader.d)
            ? 0
            : Math.min(Math.abs(leader.d - car.d), length - Math.abs(leader.d - car.d)) / length * lapSeconds,
    }))
    .sort((left, right) => left.position - right.position)
}

function GapLeaderboard({ rows, drivers }) {
  const infoByDriver = new Map((drivers ?? []).map((driver) => [driver.driver, driver]))
  if (!rows.length) return null
  return <div className="sim-gap-leaderboard" aria-label="Selected driver gaps">
    <div className="sim-gap-heading">SELECTED DRIVERS · RECORDED TIMING</div>
    {rows.map((row, index) => {
      const info = infoByDriver.get(row.driver)
      const name = displayDriverName(info ?? { driver: row.driver })
      return <div className="sim-gap-row" key={row.driver}>
        <i style={{ background: info?.color ?? FALLBACK_COLORS[index % FALLBACK_COLORS.length] }} />
        <b>P{row.position} {name}</b>
        <span>{row.position === 1 ? 'LEAD' : Number.isFinite(row.gapS) ? `+${row.gapS.toFixed(2)}` : '—'}</span>
      </div>
    })}
  </div>
}

function formatGapToCarAhead(seconds) {
  if (!Number.isFinite(seconds)) return '—'
  const value = Math.max(0, Number(seconds))
  const minutes = Math.floor(value / 60)
  const remainder = (value - minutes * 60).toFixed(3).padStart(6, '0')
  return `-${minutes}:${remainder}`
}

function lapStateAt(boundaries, time) {
  const entries = boundaries ?? []
  return entries.find((entry, index) => {
    const isLast = index === entries.length - 1
    return time >= entry.startT && (time < entry.endT || (isLast && time <= entry.endT))
  }) ?? null
}

function fastestCompletedLapDriver(lapBoundariesByDriver, time) {
  let fastest = null
  Object.entries(lapBoundariesByDriver ?? {}).forEach(([driver, entries]) => {
    entries.forEach((entry) => {
      if (time < entry.endT || !Number.isFinite(Number(entry.lapTimeS))) return
      if (!fastest || Number(entry.lapTimeS) < Number(fastest.lapTimeS)) fastest = { driver, lapTimeS: Number(entry.lapTimeS) }
    })
  })
  return fastest
}

function raceControlAt(lapBoundariesByDriver, time) {
  const codes = new Set(Object.values(lapBoundariesByDriver ?? {}).map((entries) => lapStateAt(entries, time)?.trackStatus).filter(Boolean).flatMap((status) => String(status).split('')))
  if (codes.has('5')) return { label: 'RED FLAG', tone: 'red' }
  if (codes.has('4')) return { label: 'SAFETY CAR', tone: 'yellow' }
  if (codes.has('6') || codes.has('7')) return { label: 'VIRTUAL SAFETY CAR', tone: 'yellow' }
  return null
}

function recordedRaceOrder(frame, drivers, lapBoundariesByDriver, time) {
  const infoByDriver = new Map((drivers ?? []).map((driver) => [driver.driver, driver]))
  const fastestLap = fastestCompletedLapDriver(lapBoundariesByDriver, time)
  const classified = (frame?.cars ?? [])
    .filter((car) => Number.isFinite(Number(car.timingPosition)))
    // Grid positions are an honest opening-state fallback, but a car that
    // never receives a live timing update must not keep a stale grid P-number
    // in the classification later in the race (for example after retirement).
    .filter((car) => car.timingSource !== 'RECORDED_GRID_POSITION' || Number(time) <= 60)
    .sort((left, right) => Number(left.timingPosition) - Number(right.timingPosition))

  return classified.map((car, index) => {
    const carGapToLeader = Number(car.timingGapS)
    const aheadGapToLeader = Number(classified[index - 1]?.timingGapS)
    return {
      ...car,
      info: infoByDriver.get(car.driver),
      tyre: lapStateAt(lapBoundariesByDriver?.[car.driver], time)?.compound ?? null,
      hasFastestLap: fastestLap?.driver === car.driver,
      gapToAheadS: index === 0 || !Number.isFinite(carGapToLeader) || !Number.isFinite(aheadGapToLeader)
        ? null
        : Math.max(0, carGapToLeader - aheadGapToLeader),
    }
  })
}

function RaceOrderBoard({ rows }) {
  if (!rows.length) return null
  return <section className="sim-card sim-race-order" aria-label="Full recorded race classification">
    <div className="sim-panel-head"><span>RACE CLASSIFICATION</span><em>RECORDED TIMING</em></div>
    <p className="sim-order-help">LIVE CLASSIFICATION · GAP TO CAR AHEAD</p>
    <div className="sim-order-list">
      {rows.map((row, index) => {
        const info = row.info ?? { driver: row.driver }
        return <div className="sim-order-row" key={row.driver}>
          <span>P{row.timingPosition}</span>
          <i style={{ background: info.color ?? FALLBACK_COLORS[index % FALLBACK_COLORS.length] }} />
          <b>{displayDriverName(info)}</b>
          <em className={row.hasFastestLap ? 'is-fastest' : ''}>{index === 0 ? '----' : formatGapToCarAhead(row.gapToAheadS)}</em>
          <strong className={`sim-tyre tyre-${String(row.tyre ?? 'unknown').toLowerCase()}`} title={row.tyre ?? 'Tyre unknown'}>{String(row.tyre ?? '—').charAt(0)}</strong>
        </div>
      })}
    </div>
  </section>
}

function AutoRoleControl({ label, driver, info, emptyLabel }) {
  return <div className="sim-auto-role">
    <span>{label}</span>
    <b>{driver ? displayDriverName(info?.get(driver) ?? { driver }) : emptyLabel}</b>
  </div>
}

function RecordedTrack({ trackmap, frame, drivers, attacker, defender, selectedDrivers, viewMode, branchOverlay }) {
  const geometry = useMemo(() => projection(trackmap?.points), [trackmap])
  const driverByCode = useMemo(() => new Map((drivers ?? []).map((driver) => [driver.driver, driver])), [drivers])
  if (!geometry) return <div className="sim-loading">LOADING RECORDED CIRCUIT GEOMETRY…</div>
  const { project, outline, width, height } = geometry
  const polyline = outline.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(' ')
  const polylineFor = (points) => points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(' ')
  const zones = (trackmap?.visualOverlay?.zones ?? []).map((zone) => ({
    ...zone,
    points: pointsForZone(trackmap.points, Number(zone.startD), Number(zone.endD)).map((point) => project(point.x, point.y)),
  })).filter((zone) => zone.points.length > 1)
  const markers = (trackmap?.visualOverlay?.markers ?? []).map((marker) => ({ ...marker, point: pointAtDistance(trackmap.points, marker.d) }))
  return <svg className={`sim-track ${viewMode === '3D' ? 'is-3d' : ''}`} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={`${viewMode} recorded public position replay on circuit map`}>
    <polyline points={polyline} fill="none" stroke="#14201f" strokeWidth="20" strokeLinecap="round" strokeLinejoin="round" />
    <polyline points={polyline} fill="none" stroke="#6f817c" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="7 10" opacity=".85" />
    {zones.map((zone) => <polyline key={zone.id} className="sim-straight-zone" points={polylineFor(zone.points)} fill="none" strokeLinecap="round" strokeLinejoin="round" />)}
    {(trackmap?.corners ?? []).map((corner) => {
      if (corner.x == null || corner.y == null) return null
      const point = project(corner.x, corner.y)
      return <g key={corner.n} className="sim-corner" transform={`translate(${point.x},${point.y})`}><circle r="8" /><text y="3">{corner.n}</text></g>
    })}
    {markers.map((marker) => {
      if (!marker.point) return null
      const point = project(marker.point.x, marker.point.y)
      return <g key={marker.type} className={`sim-mode-marker ${marker.type.toLowerCase()}`} transform={`translate(${point.x},${point.y})`}><circle r="7" /><text x="11" y="3">{marker.type === 'DETECTION' ? 'DET' : 'ACT'}</text></g>
    })}
    {(frame?.cars ?? []).map((car, index) => {
      const point = project(car.x, car.y)
      const selected = selectedDrivers.includes(car.driver)
      const modelPair = car.driver === attacker || car.driver === defender
      const color = driverByCode.get(car.driver)?.color ?? FALLBACK_COLORS[index % FALLBACK_COLORS.length]
      return <g key={car.driver} className={`${selected ? 'sim-car selected' : 'sim-car'}${modelPair ? ' model-pair' : ''}`} transform={`translate(${point.x},${point.y})`}>
        <circle r={selected ? 9 : 3.6} fill={color} />
        {selected && <><circle className="sim-car-ring" r="13" /><text y="-18">{car.driver}</text></>}
      </g>
    })}
    {branchOverlay && (frame?.cars ?? []).filter((car) => car.driver === branchOverlay.driver || car.driver === branchOverlay.target).map((car) => {
      const point = project(car.x, car.y)
      const isSelected = car.driver === branchOverlay.driver
      return <g key={`branch-${car.driver}`} className={`sim-branch-car ${isSelected ? 'ours' : 'target'}`} transform={`translate(${point.x},${point.y})`}>
        <circle r={isSelected ? 17 : 14} />
        {isSelected && <><text y="-25">MODEL {branchOverlay.action}</text><text y="28">PASS {Math.round(branchOverlay.probability * 100)}%</text></>}
      </g>
    })}
  </svg>
}

function SelectControl({ label, value, onChange, children, disabled = false }) {
  return <label className="sim-control"><span>{label}</span><select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>{children}</select></label>
}

export function SimulationReplayView({ onOpenDashboard, onHome }) {
  const [selection, setSelection] = useState({ year: 2026, round: 7, session: 'Race', driver: 'HAM', lap: 1 })
  const [events, setEvents] = useState([])
  const [decision, setDecision] = useState({ loading: true })
  const [trackmap, setTrackmap] = useState(null)
  const [replay, setReplay] = useState({ loading: true })
  const [branch, setBranch] = useState({ idle: true })
  const [branchRequest, setBranchRequest] = useState(null)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [viewMode, setViewMode] = useState('2D')
  const [displayMode, setDisplayMode] = useState('OBSERVED')
  const [selectedDrivers, setSelectedDrivers] = useState([])
  const [frameIndex, setFrameIndex] = useState(0)
  const [playhead, setPlayhead] = useState(0)
  const cursorRef = useRef(0)
  const speedRef = useRef(1)
  const predictionCacheRef = useRef(new Map())
  const branchCacheRef = useRef(new Map())
  const update = (patch) => setSelection((current) => ({ ...current, ...patch }))
  const selectedEvent = events.find((item) => Number(item.round) === Number(selection.round))
  // A round can be on the published calendar without containing a completed,
  // locally ingested Race session. A recorded replay must never fake that.
  const raceReplayUnavailable = Boolean(selectedEvent && selectedEvent.raceDataAvailable === false)

  useEffect(() => {
    if (raceReplayUnavailable) {
      setDecision({ unavailable: true })
      return undefined
    }
    let live = true
    setEvents([])
    requestJson(`/api/f1/events?year=${selection.year}`).then((payload) => { if (live) setEvents(payload.events ?? []) }).catch(() => {
      requestJson(`/api/f1/rounds?year=${selection.year}`).then((payload) => { if (live) setEvents(payload.events ?? []) }).catch(() => { if (live) setEvents([]) })
    })
    return () => { live = false }
  }, [selection.year])

  useEffect(() => {
    if (!events.length) return
    const event = events.find((item) => Number(item.round) === Number(selection.round)) ?? events[0]
    const sessions = event.sessions ?? ['Race']
    const session = sessions.includes(selection.session) ? selection.session : (sessions.includes('Race') ? 'Race' : sessions[0])
    if (Number(event.round) !== Number(selection.round) || session !== selection.session) update({ round: event.round, session, lap: 1 })
    // Only changes a stale round after changing season.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events])

  useEffect(() => {
    let live = true
    setDecision({ loading: true })
    requestJson(`/api/f1/decisionpoints?year=${selection.year}&round=${selection.round}&session=${selection.session}`).then((payload) => {
      if (!live) return
      const roster = payload.participants ?? []
      const rows = payload.analysisRows ?? payload.rows ?? []
      let driver = roster.includes(selection.driver) ? selection.driver : roster[0]
      // A replay can show any participant, but the initial simulation state
      // must be a real detected pair rather than (for example) HAM paired
      // with the defender from somebody else's Barcelona battle.
      let firstBattle = rows.find((row) => row.driver === driver)
      if (!firstBattle && rows.length) {
        firstBattle = rows[0]
        driver = firstBattle.driver
      }
      setDecision({ data: payload })
      // Keep the user's jump/branch lap intact. A detected battle can inform
      // the initial pair, but it must not silently replace full-race playback
      // with a clip beginning at that battle.
      update({ driver: driver ?? selection.driver })
    }).catch((error) => { if (live) setDecision({ error: error.message }) })
    return () => { live = false }
    // Race selection, rather than pair selection, fetches the source state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raceReplayUnavailable, selection.year, selection.round, selection.session])

  useEffect(() => {
    if (raceReplayUnavailable) {
      setTrackmap({ unavailable: true })
      return undefined
    }
    let live = true
    setTrackmap(null)
    requestJson(`/api/f1/trackmap?year=${selection.year}&round=${selection.round}&session=${selection.session}`).then((payload) => { if (live) setTrackmap(payload) }).catch(() => { if (live) setTrackmap({ error: 'no recorded circuit geometry' }) })
    return () => { live = false }
  }, [raceReplayUnavailable, selection.year, selection.round, selection.session])

  useEffect(() => {
    if (raceReplayUnavailable) {
      cursorRef.current = 0
      setFrameIndex(0)
      setPlayhead(0)
      setPlaying(false)
      setReplay({ unavailable: true })
      setSelectedDrivers([])
      setBranch({ idle: true })
      return undefined
    }
    if (!selection.lap) return undefined
    let live = true
    cursorRef.current = 0
    setFrameIndex(0)
    setPlayhead(0)
    setPlaying(false)
    setReplay({ loading: true })
    requestJson(`/api/f1/racereplay?year=${selection.year}&round=${selection.round}&session=${selection.session}`).then((payload) => { if (live) setReplay({ data: payload }) }).catch((error) => { if (live) setReplay({ error: error.message }) })
    return () => { live = false }
  }, [raceReplayUnavailable, selection.year, selection.round, selection.session])

  useEffect(() => { speedRef.current = speed }, [speed])
  useEffect(() => {
    if (!playing || !replay.data?.frames?.length) return undefined
    let animation
    let previous = performance.now()
    const { frames } = replay.data
    const duration = replay.data.window.durationS
    const tick = (now) => {
      // 1× is the real elapsed time recorded for the selected driver's lap.
      // Higher settings are exact multiples of that recorded timebase.
      cursorRef.current += ((now - previous) / 1000) * speedRef.current
      previous = now
      if (cursorRef.current >= duration) { cursorRef.current = duration; setPlaying(false) }
      const next = frames.findIndex((item) => item.t >= cursorRef.current)
      setFrameIndex(next < 0 ? frames.length - 1 : next)
      setPlayhead(cursorRef.current)
      if (cursorRef.current < duration) animation = requestAnimationFrame(tick)
    }
    animation = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(animation)
  }, [playing, replay.data])

  const orderedDrivers = useMemo(() => [...(replay.data?.drivers ?? [])].sort((left, right) => {
    const leftPosition = Number(left.classifiedPosition)
    const rightPosition = Number(right.classifiedPosition)
    if (Number.isFinite(leftPosition) && Number.isFinite(rightPosition)) return leftPosition - rightPosition
    if (Number.isFinite(leftPosition)) return -1
    if (Number.isFinite(rightPosition)) return 1
    return String(left.driver).localeCompare(String(right.driver))
  }), [replay.data])
  const roster = decision.data?.participants ?? orderedDrivers.map((driver) => driver.driver)
  useEffect(() => {
    if (!roster.length) return
    setSelectedDrivers((current) => {
      const valid = current.filter((driver) => roster.includes(driver))
      if (valid.length) return valid.slice(0, MAX_VISUAL_SELECTIONS)
      return [selection.driver].filter((driver) => driver && roster.includes(driver)).slice(0, MAX_VISUAL_SELECTIONS)
    })
  // Keep a user's visual selection stable while the replay frame changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [decision.data, replay.data])
  const rows = decision.data?.analysisRows ?? decision.data?.rows ?? []
  // The recorded-input panel is intentionally strict: if this exact lap was
  // not a detected close battle, it must say so rather than borrow a battle
  // from another lap for the same pair.
  const totalLaps = replay.data?.totalLaps ?? decision.data?.totalLaps ?? selection.lap
  const driverLapBoundaries = replay.data?.lapBoundariesByDriver?.[selection.driver] ?? []
  const availableLaps = useMemo(() => driverLapBoundaries.map((entry) => entry.lap), [driverLapBoundaries])
  const frames = replay.data?.frames ?? []
  const currentFrame = interpolateFrame(frames, playhead) ?? frames[frameIndex] ?? frames[0]
  const event = selectedEvent
  const infoByDriver = new Map((replay.data?.drivers ?? []).map((driver) => [driver.driver, driver]))
  const visualOverlay = trackmap?.visualOverlay
  const raceTime = replay.data ? replay.data.window.sessionStartS + (currentFrame?.t ?? 0) : null
  // Adjacent laps share a boundary. Treat the end as exclusive so a jump to
  // L20's start is visibly L20, not the tail end of L19.
  const playbackLap = lapStateAt(driverLapBoundaries, currentFrame?.t ?? 0) ?? driverLapBoundaries.at(-1)
  const selectedBranchLap = driverLapBoundaries.find((entry) => entry.lap === selection.lap)
  const gapRows = selectedGapRows(currentFrame, trackmap, replay.data?.drivers, selectedDrivers, playbackLap?.lapTimeS ?? replay.data?.window?.durationS)
  const playbackTime = currentFrame?.t ?? 0
  const lapBoundariesByDriver = replay.data?.lapBoundariesByDriver ?? {}
  const raceOrder = recordedRaceOrder(currentFrame, replay.data?.drivers, lapBoundariesByDriver, playbackTime)
  // The branch input must be resolved at the lap selected in JUMP / BRANCH
  // LAP, not at whichever moment the observed replay happens to be showing.
  // Otherwise choosing (for example) Hadjar at L8 would accidentally use the
  // car ahead of Hadjar at L1 as the model opponent.
  const branchTime = selectedBranchLap?.startT ?? playbackTime
  const branchFrame = selectedBranchLap ? (interpolateFrame(frames, branchTime) ?? currentFrame) : currentFrame
  const branchRaceOrder = recordedRaceOrder(branchFrame, replay.data?.drivers, lapBoundariesByDriver, branchTime)
  const raceControl = raceControlAt(lapBoundariesByDriver, playbackTime)
  const selectedDriverIndex = branchRaceOrder.findIndex((row) => row.driver === selection.driver)
  const attackingDriver = selectedDriverIndex > 0 ? branchRaceOrder[selectedDriverIndex - 1].driver : null
  const defendingDriver = selectedDriverIndex >= 0 && selectedDriverIndex < branchRaceOrder.length - 1 ? branchRaceOrder[selectedDriverIndex + 1].driver : null
  const selectedDriverRow = selectedDriverIndex >= 0 ? branchRaceOrder[selectedDriverIndex] : null
  // A selected driver can legitimately be either side of a recorded battle:
  // attacking the car ahead, or defending from the car behind. The extractor
  // stores each event from the attacking car's viewpoint, so keep that row for
  // scoring but re-orient the rollout identity to the driver the user chose.
  const attackBattle = attackingDriver
    ? rows.find((row) => row.driver === selection.driver && row.defender === attackingDriver && Number(row.lap) === Number(selection.lap)) ?? null
    : null
  const defendBattle = defendingDriver
    ? rows.find((row) => row.driver === defendingDriver && row.defender === selection.driver && Number(row.lap) === Number(selection.lap)) ?? null
    : null
  const branchRole = attackBattle ? 'ATTACKING' : defendBattle ? 'DEFENDING' : null
  const battle = attackBattle ?? defendBattle
  const modelOpponent = branchRole === 'DEFENDING' ? defendingDriver : attackingDriver
  const branchFocus = battle
    ? branchRole === 'DEFENDING'
      ? {
          ...battle,
          // Preserve observed feature orientation for the classifier and the
          // defensive threat calculation, but make selected/opponent identity
          // explicit for the future-blind two-car rollout.
          driver: selection.driver,
          defender: defendingDriver,
          position: battle.defenderPosition,
          defenderPosition: battle.position,
          selectedRole: 'DEFENDING',
          sourcePerspective: 'OBSERVED_OPPONENT_ATTACK',
        }
      : {
          ...battle,
          selectedRole: 'ATTACKING',
          sourcePerspective: 'OBSERVED_SELECTED_ATTACK',
        }
    : null
  const toggleVisualDriver = (driver) => setSelectedDrivers((current) => {
    if (current.includes(driver)) {
      if (current.length <= 2) return current
      return current.filter((item) => item !== driver)
    }
    if (current.length >= MAX_VISUAL_SELECTIONS) return current
    return [...current, driver]
  })
  const restart = () => { cursorRef.current = 0; setFrameIndex(0); setPlayhead(0); setPlaying(false); setBranchRequest(null); setBranch({ idle: true }); setDisplayMode('OBSERVED') }
  const jumpToLap = (lap) => {
    const boundary = driverLapBoundaries.find((entry) => entry.lap === lap)
    if (!boundary) return
    cursorRef.current = boundary.startT
    const index = frames.findIndex((frame) => frame.t >= boundary.startT)
    setFrameIndex(index < 0 ? Math.max(0, frames.length - 1) : index)
    setPlayhead(boundary.startT)
    setPlaying(false)
    update({ lap })
  }
  const startPitwolfBranch = () => {
    jumpToLap(selection.lap)
    setDisplayMode('BRANCH')
    if (!battle || !modelOpponent || !branchFocus) {
      setBranchRequest(null)
      setBranch({ error: 'No extracted close battle exists for this driver and recorded lap. PitWolf needs one real public two-car state to start a branch.' })
      return
    }
    setBranchRequest({
      id: `${Date.now()}:${selection.year}:${selection.round}:${selection.session}:${selection.driver}:${modelOpponent}:${branchRole}:${selection.lap}`,
      year: selection.year,
      round: selection.round,
      session: selection.session,
      driver: selection.driver,
      defender: modelOpponent,
      lap: selection.lap,
      role: branchRole,
      focus: branchFocus,
      predictionRow: battle,
      totalLaps: decision.data?.totalLaps ?? totalLaps,
      finishPositions: decision.data?.finishPositions ?? {},
      ruleContext: decision.data?.ruleContext,
    })
  }

  useEffect(() => {
    if (!availableLaps.length || availableLaps.includes(selection.lap)) return
    update({ lap: availableLaps[0] })
  // The server returns real lap boundaries for the selected driver. If they
  // did not complete the currently selected lap, choose their first real lap.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection.driver, replay.data])

  useEffect(() => {
    if (!branchRequest) return undefined
    let live = true
    const ruleKey = branchRequest.ruleContext?.eventKey ?? branchRequest.ruleContext?.status ?? 'no-rule-context'
    const pairKey = `${branchRequest.year}:${branchRequest.round}:${branchRequest.session}:${branchRequest.driver}:${branchRequest.defender}:${branchRequest.role}:${ruleKey}`
    const branchKey = `${pairKey}:${branchRequest.lap}:race-to-flag`
    const cachedBranch = branchCacheRef.current.get(branchKey)
    if (cachedBranch) {
      setBranch({ data: cachedBranch })
      setPlaying(true)
      return undefined
    }
    const focusRow = branchRequest.focus
    const predictionRow = branchRequest.predictionRow ?? focusRow
    setBranch({ loading: true })
    Promise.all([
      predictionCacheRef.current.get(pairKey)
        ? Promise.resolve(predictionCacheRef.current.get(pairKey))
        : postJson('/api/f1/overtake/predict', { rows: [predictionRow], ruleContext: branchRequest.ruleContext }).then((payload) => {
          predictionCacheRef.current.set(pairKey, payload)
          return payload
        }),
      fetchEnergyLap(branchRequest.year, branchRequest.round, branchRequest.session, branchRequest.driver, branchRequest.lap),
      fetchEnergyLap(branchRequest.year, branchRequest.round, branchRequest.session, branchRequest.defender, branchRequest.lap),
    ]).then(([prediction, attackerEnergy, defenderEnergy]) => {
      const focus = { ...focusRow, pred: prediction.predictions?.[0] }
      if (!focus.pred) throw new Error('The selected battle could not be scored for this exact lap.')
      return postJson('/api/f1/replay/strategy', {
        focus,
        mode: 'FUTURE_BLIND_RACE_ROLLOUT',
        rows: [],
        totalLaps: branchRequest.totalLaps,
        finishPositions: branchRequest.finishPositions,
        // The public energy calculation supplies a modelled lap-start SoC.
        // It is explicitly a surrogate, never private team battery telemetry.
        energyLaps: {
          [branchRequest.driver]: [{ lap: branchRequest.lap, socEndMj: attackerEnergy?.summary?.socStartMj ?? (branchRequest.role === 'DEFENDING' ? focusRow.defenderSoCMj : focusRow.attackerSoCMj) }],
          [branchRequest.defender]: [{ lap: branchRequest.lap, socEndMj: defenderEnergy?.summary?.socStartMj ?? (branchRequest.role === 'DEFENDING' ? focusRow.attackerSoCMj : focusRow.defenderSoCMj) }],
        },
        year: branchRequest.year,
        regulationEra: Number(branchRequest.year) >= 2026 ? '2026' : '2018_2025',
        ruleContext: branchRequest.ruleContext,
      }).then((tree) => ({ tree, focus, attackerEnergy, defenderEnergy }))
    }).then((data) => {
      branchCacheRef.current.set(branchKey, data)
      if (live) {
        setBranch({ data })
        setPlaying(true)
      }
    }).catch((error) => {
      if (live) setBranch({ error: error.message })
    })
    return () => { live = false }
  }, [branchRequest])

  const branchMatchesSelection = Boolean(
    branchRequest
    && branchRequest.year === selection.year
    && branchRequest.round === selection.round
    && branchRequest.session === selection.session
    && branchRequest.driver === selection.driver
    && branchRequest.lap === selection.lap
  )
  const activeBranch = branchMatchesSelection ? branch : { idle: true }
  const branchTree = activeBranch.data?.tree
  const branchStep = branchTree?.path?.[0]
  const branchPrediction = activeBranch.data?.focus?.pred
  const actionProbabilities = branchPrediction?.probabilities
  const commandGate = branchTree?.decisionContext?.ruleContext?.application
  const branchStartLap = branchMatchesSelection ? branchRequest.lap : selection.lap
  const branchStepIndex = Math.max(0, Math.min((branchTree?.path?.length ?? 1) - 1, (playbackLap?.lap ?? branchStartLap) - branchStartLap))
  const activeBranchStep = branchTree?.path?.[branchStepIndex] ?? null
  const branchOverlay = displayMode === 'BRANCH' && activeBranchStep && activeBranch.data?.focus?.driver && activeBranch.data?.focus?.defender
    ? {
        driver: activeBranch.data.focus.driver,
        target: activeBranch.data.focus.defender,
        action: activeBranchStep.action,
        probability: activeBranchStep.probability,
        aheadProbability: activeBranchStep.aheadProbability,
        role: branchTree?.selectedRoleAtJump ?? branchRequest?.role,
        lap: activeBranchStep.lap,
      }
    : null
  const branchSocCapacity = Number(branchTree?.energyConfig?.capacityMj)
  const branchSocWithinWindow = branchTree?.path?.every((step) => (
    Number.isFinite(Number(step.ourSoc)) && Number.isFinite(Number(step.defenderSoc))
      && Number(step.ourSoc) >= 0 && Number(step.defenderSoc) >= 0
      && Number(step.ourSoc) <= branchSocCapacity && Number(step.defenderSoc) <= branchSocCapacity
  ))

  if (raceReplayUnavailable) {
    return <main className="sim-root">
      <header className="sim-header"><button type="button" className="sim-brand" onClick={onHome} aria-label="Back to home"><span>✦</span><strong>PIT<em>WOLF</em></strong></button><div className="sim-header-title"><b>SIMULATION / RECORDED RACE STATE</b><span>SCHEDULED EVENT · RECORDED REPLAY UNAVAILABLE</span></div><button type="button" className="sim-dashboard" onClick={onOpenDashboard}>STRATEGY DASHBOARD ↗</button></header>
      <section className="sim-controls">
        <SelectControl label="SEASON" value={selection.year} onChange={(year) => update({ year: Number(year) })}>{YEARS.map((year) => <option key={year} value={year}>{year}</option>)}</SelectControl>
        <SelectControl label="RACE" value={selection.round} onChange={(round) => { const next = events.find((item) => Number(item.round) === Number(round)); const sessions = next?.sessions ?? ['Race']; update({ round: Number(round), session: sessions.includes('Race') ? 'Race' : sessions[0], lap: 1 }) }}>{events.map((item) => <option key={item.round} value={item.round}>{item.name}</option>)}</SelectControl>
      </section>
      <section className="sim-unavailable-wrap">
        <div className="sim-card sim-race-unavailable">
          <div className="sim-panel-head"><span>{selectedEvent?.name?.toUpperCase() ?? 'SCHEDULED GRAND PRIX'}</span><em>RACE DATA PENDING</em></div>
          <b>RECORDED REPLAY NOT AVAILABLE YET</b>
          <p>Currently no race has taken place for this Grand Prix. Hold your seats—we’ll simulate it as soon as the race happens and we receive the data.</p>
          {selectedEvent?.date && <span>CALENDAR DATE · {selectedEvent.date}</span>}
        </div>
      </section>
    </main>
  }

  return <main className="sim-root">
    <header className="sim-header"><button type="button" className="sim-brand" onClick={onHome} aria-label="Back to home"><span>✦</span><strong>PIT<em>WOLF</em></strong></button><div className="sim-header-title"><b>SIMULATION / RECORDED RACE STATE</b><span>{branchTree ? 'PUBLIC POSITION + TIMING DATA · MODELLED TWO-CAR BRANCH · ANALYSIS ONLY' : 'PUBLIC FASTF1 POSITION + TIMING DATA · MODEL BRANCH LOADING'}</span></div><button type="button" className="sim-dashboard" onClick={onOpenDashboard}>STRATEGY DASHBOARD ↗</button></header>
    <section className="sim-controls">
      <SelectControl label="SEASON" value={selection.year} onChange={(year) => update({ year: Number(year) })}>{YEARS.map((year) => <option key={year} value={year}>{year}</option>)}</SelectControl>
      <SelectControl label="RACE" value={selection.round} onChange={(round) => { const event = events.find((item) => Number(item.round) === Number(round)); const sessions = event?.sessions ?? ['Race']; update({ round: Number(round), session: sessions.includes('Race') ? 'Race' : sessions[0], lap: 1 }) }}>{events.map((item) => <option key={item.round} value={item.round}>{item.name}</option>)}</SelectControl>
      <SelectControl label="SESSION" value={selection.session} onChange={(session) => update({ session, lap: 1 })}>{(event?.sessions ?? ['Race']).map((session) => <option key={session} value={session}>{SESSION_LABELS[session] ?? session}</option>)}</SelectControl>
      <SelectControl label="DRIVER" value={selection.driver} onChange={(driver) => { update({ driver, lap: 1 }); setSelectedDrivers((current) => [driver, ...current.filter((item) => item !== driver)].slice(0, MAX_VISUAL_SELECTIONS)) }}>{roster.map((driver) => <option key={driver} value={driver}>{displayDriverName(infoByDriver.get(driver) ?? { driver })}</option>)}</SelectControl>
      <AutoRoleControl label="ATTACKING" driver={attackingDriver} info={infoByDriver} emptyLabel={selectedDriverRow ? 'NONE — LEADING' : 'LOADING…'} />
      <AutoRoleControl label="DEFENDING" driver={defendingDriver} info={infoByDriver} emptyLabel={selectedDriverRow ? 'NONE — LAST' : 'LOADING…'} />
      <label className="sim-lap-control"><span>JUMP / BRANCH LAP</span><select value={selection.lap} onChange={(event) => update({ lap: Number(event.target.value) })}>{availableLaps.map((lap) => <option key={lap} value={lap}>Lap {lap}</option>)}</select><b>SELECT L{selection.lap} / {totalLaps}</b></label>
      <div className="sim-jump-control"><span>FUTURE-BLIND BRANCH</span><button type="button" onClick={startPitwolfBranch}>JUMP</button><b>{battle ? `${branchRole} · L${selection.lap} → FLAG` : 'NEEDS DETECTED BATTLE'}</b></div>
    </section>
    <section className="sim-content">
      <aside className="sim-driver-rail">
        <section className="sim-card sim-driver-list">
          <div className="sim-panel-head"><span>DRIVERS</span><em>{selectedDrivers.length} / {roster.length}</em></div>
          <p className="sim-driver-help">Select up to four cars to show on the track and in the gap board.</p>
          <div className="sim-driver-options">
            {orderedDrivers.map((driver, index) => {
              const selected = selectedDrivers.includes(driver.driver)
              const disabled = !selected && selectedDrivers.length >= MAX_VISUAL_SELECTIONS
              return <label className={`sim-driver-option${selected ? ' selected' : ''}${disabled ? ' disabled' : ''}`} key={driver.driver}>
                <span className="sim-driver-position">{displayPosition(driver, index + 1)}</span>
                <span className="sim-driver-color" style={{ background: driver.color ?? FALLBACK_COLORS[index % FALLBACK_COLORS.length] }} />
                <span className="sim-driver-name">{displayDriverName(driver)}</span>
                <input type="checkbox" checked={selected} disabled={disabled} onChange={() => toggleVisualDriver(driver.driver)} aria-label={`Show ${displayDriverName(driver)}`} />
              </label>
            })}
          </div>
        </section>
      </aside>
      <section className="sim-track-panel">
        <div className="sim-panel-head"><span>{(event?.name ?? replay.data?.event?.name ?? 'LOADING RACE').toUpperCase()} / TRACK VISUALIZATION</span><span className="sim-track-modes"><span className="sim-branch-mode" role="group" aria-label="Replay mode"><button type="button" className={displayMode === 'OBSERVED' ? 'active' : ''} aria-pressed={displayMode === 'OBSERVED'} onClick={() => setDisplayMode('OBSERVED')}>OBSERVED</button><button type="button" className={displayMode === 'BRANCH' ? 'active' : ''} aria-pressed={displayMode === 'BRANCH'} onClick={() => setDisplayMode('BRANCH')}>PITWOLF BRANCH</button></span><span className="sim-view-toggle" role="group" aria-label="Track visualisation mode"><button type="button" className={viewMode === '2D' ? 'active' : ''} aria-pressed={viewMode === '2D'} onClick={() => setViewMode('2D')}>2D</button><button type="button" className={viewMode === '3D' ? 'active' : ''} aria-pressed={viewMode === '3D'} onClick={() => setViewMode('3D')}>3D</button></span></span></div>
        <div className="sim-status-row"><b>LAP {playbackLap?.lap ?? selection.lap} / {totalLaps}</b><span>{raceTime == null ? 'LOADING SESSION TIME…' : `RACE T+${clock(raceTime)}`}</span><i>{playbackLap?.compound ?? '—'} · {playbackLap?.trackStatus === '1' ? 'GREEN' : 'TRACK STATUS UNCONFIRMED'}</i></div>
        {raceControl && <div className={`sim-race-control is-${raceControl.tone}`}>{raceControl.label}</div>}
        {displayMode === 'BRANCH' && (branchOverlay ? <div className="sim-branch-banner"><b>PITWOLF RACE BRANCH · L{branchOverlay.lap} → FLAG</b><span>{branchOverlay.role} · {branchOverlay.action} · {Math.round(branchOverlay.probability * 100)}% {branchOverlay.role === 'DEFENDING' ? 'opponent pass-threat signal' : 'immediate pass signal'} · {Math.round(branchOverlay.aheadProbability * 100)}% position-ahead estimate</span><em>FUTURE-BLIND TWO-CAR MODEL OVER RECORDED GHOSTS · ANALYSIS ONLY</em></div> : <div className="sim-branch-banner is-pending"><b>PITWOLF RACE BRANCH</b><span>{activeBranch.loading ? 'SCORING THE FUTURE-BLIND RACE-TO-FLAG ROLLOUT…' : activeBranch.error ? `BRANCH UNAVAILABLE · ${activeBranch.error}` : branchMatchesSelection ? 'PREPARING THE MODELLED RACE-TO-FLAG BRANCH…' : 'SELECT A LAP, THEN PRESS JUMP TO CREATE A BRANCH'}</span><em>RECORDED REPLAY REMAINS VISIBLE · THE MODEL RECEIVES ONLY THE SELECTED START STATE</em></div>)}
        <div className="sim-track-wrap">{replay.loading || !trackmap ? <div className="sim-loading">LOADING RECORDED POSITION WINDOW…</div> : replay.error || trackmap.error ? <div className="sim-error">Recorded replay unavailable: {replay.error ?? trackmap.error}</div> : <><RecordedTrack trackmap={trackmap} frame={currentFrame} drivers={replay.data?.drivers} attacker={selection.driver} defender={modelOpponent} selectedDrivers={selectedDrivers} viewMode={viewMode} branchOverlay={branchOverlay} /><GapLeaderboard rows={gapRows} drivers={replay.data?.drivers} /></>}</div>
        <div className="sim-player"><button type="button" onClick={() => setPlaying((value) => !value)} disabled={!frames.length}>{playing ? 'Ⅱ PAUSE' : '▷ PLAY FULL RACE'}</button><button type="button" onClick={restart} disabled={!frames.length}>↻ RESTART</button><input aria-label="Replay frame" type="range" min="0" max={Math.max(0, frames.length - 1)} value={frameIndex} onChange={(event) => { const index = Number(event.target.value); const time = frames[index]?.t ?? 0; cursorRef.current = time; setFrameIndex(index); setPlayhead(time); setPlaying(false) }} /><span className="sim-playback-lap">PLAYING L{playbackLap?.lap ?? '—'} · {branchMatchesSelection ? `BRANCH L${branchStartLap} → FLAG` : 'NO BRANCH'}</span><div className="sim-speeds">{SPEEDS.map((value) => <button key={value} type="button" className={speed === value ? 'active' : ''} onClick={() => setSpeed(value)}>{value}×</button>)}</div></div>
      </section>
      <aside className="sim-side">
        <RaceOrderBoard rows={raceOrder} />
        <section className="sim-card"><div className="sim-panel-head"><span>SELECTED DRIVER STATE</span><em>RECORDED INPUT · L{selection.lap}</em></div><div className="sim-pair"><div><span>DRIVER</span><b>{selection.driver}</b><em>{selectedDriverRow ? `P${selectedDriverRow.timingPosition}` : '—'}</em></div><div><span>ATTACKING</span><b>{attackingDriver ?? 'NONE'}</b><em>{attackingDriver ? `P${branchRaceOrder.find((row) => row.driver === attackingDriver)?.timingPosition ?? '—'}` : 'LEADING'}</em></div></div><div className="sim-facts"><span>DEFENDING AGAINST <b>{defendingDriver ?? 'NONE — LAST'}</b></span>{battle ? <><span>MODEL ROLE <b>{branchRole}</b></span><span>RELATIVE SPEED <b>{battle.speedDeltaKph == null ? '—' : `${battle.speedDeltaKph > 0 ? '+' : ''}${battle.speedDeltaKph} km/h`}</b></span><span>TYRES <b>{battle.attackerCompound ?? '—'} / {battle.defenderCompound ?? '—'}</b></span><span>MODEL STATE <b>{activeBranch.loading ? 'SCORING' : branchTree ? 'MODELLED BRANCH READY' : activeBranch.error ? 'UNAVAILABLE' : 'AWAITING JUMP'}</b></span></> : <span>MODEL STATE <b>{attackingDriver || defendingDriver ? 'NO DETECTED BATTLE AT THIS LAP' : 'NO ADJACENT CAR'}</b></span>}</div>{!battle && <p className="sim-note">The relationship is automatic from the recorded classification at the selected branch lap. A branch can begin from either an extracted attack on the car ahead or an extracted defence against the car behind.</p>}</section>
        <section className="sim-card sim-branch"><div className="sim-panel-head"><span>PITWOLF RACE BRANCH</span><em>{branchTree ? `L${branchTree.startLap} → L${branchTree.finishLap}` : 'AWAITING JUMP'}</em></div>{activeBranch.loading ? <p className="sim-note">Freezing the selected public state, then rolling PitWolf’s energy and overtake policy forward to the flag without later recorded inputs…</p> : activeBranch.error ? <p className="sim-note sim-error">Modelled branch unavailable: {activeBranch.error}</p> : branchTree && branchStep && actionProbabilities ? <><div className="sim-action"><b>{branchStep.action}</b><span>FIRST FUTURE-BLIND POLICY ACTION</span><em>{Math.round((actionProbabilities[branchStep.action] ?? 0) * 100)}% START-STATE ACTION-MODEL SIGNAL</em></div><div className="sim-probabilities">{['ATTACK', 'DELAY', 'SAVE'].map((action) => <span key={action}><i className={`sim-prob-${action.toLowerCase()}`} style={{ width: `${Math.round((actionProbabilities[action] ?? 0) * 100)}%` }} />{action} <b>{Math.round((actionProbabilities[action] ?? 0) * 100)}%</b></span>)}</div><div className="sim-facts"><span>MODEL ROLE AT JUMP <b>{branchTree.selectedRoleAtJump ?? branchRequest?.role ?? '—'}</b></span><span>MODELLED SoC AT JUMP <b>{branchTree.tree.ourSoc.toFixed(2)} / {branchTree.tree.defenderSoc.toFixed(2)} MJ</b></span><span>AFTER FIRST ACTION <b>{branchStep.ourSoc.toFixed(2)} / {branchStep.defenderSoc.toFixed(2)} MJ</b></span><span>PAIR-AHEAD ESTIMATE AT FLAG <b>{Math.round((branchTree.modelledPairAheadProbabilityAtFlag ?? branchTree.path.at(-1)?.aheadProbability ?? 0) * 100)}%</b></span><span>OPPONENT RESPONSE <b>{branchStep.opponentAction}</b></span></div><ol className="sim-branch-log">{(branchTree.actionChanges?.length ? branchTree.actionChanges : branchTree.path).slice(0, 12).map((step, index) => <li key={`${step.lap}-${index}`}><b>L{step.lap}</b><span>{step.action}{step.opponentAction ? ` · opponent ${step.opponentAction}` : ' · policy change'}</span><em>{step.aheadProbability == null ? 'ROLLOUT' : `${Math.round(step.aheadProbability * 100)}% ahead · ${step.ourSoc.toFixed(2)} MJ`}</em></li>)}</ol><p className="sim-note">The rollout receives only the public state at JUMP. It evolves the selected driver and their real recorded opponent in either an attacking or defensive role; it does not read later race rows or claim a full-grid counterfactual.</p></> : <p className="sim-note">Choose a recorded lap, then press JUMP to start a future-blind race-to-flag branch.</p>}</section>
        {branchTree && <section className="sim-card sim-comparison"><div className="sim-panel-head"><span>OBSERVED / MODELLED FLAG COMPARISON</span><em>TWO-CAR ONLY</em></div><div className="sim-facts"><span>MODELLED PAIR ORDER AT FLAG <b>{branchTree.modelledPairAheadAtFlag ? 'SELECTED DRIVER AHEAD' : 'SELECTED DRIVER BEHIND'}</b></span><span>MODELLED PAIR-AHEAD ESTIMATE <b>{Math.round((branchTree.modelledPairAheadProbabilityAtFlag ?? 0) * 100)}%</b></span><span>OBSERVED PAIR ORDER AT FLAG <b>{branchTree.actualPairAheadAtFlag ? 'SELECTED DRIVER AHEAD' : 'SELECTED DRIVER BEHIND'}</b></span><span>REAL SELECTED FINISH <b>{branchTree.actualFinishPosition ? `P${branchTree.actualFinishPosition}` : 'NOT CLASSIFIED'}</b></span><span>MODELLED ENERGY WINDOW <b>{branchSocWithinWindow ? `WITHIN 0–${branchSocCapacity.toFixed(0)} MJ` : 'OUTSIDE MODEL WINDOW'}</b></span><span>FIA COMMAND GATE <b>{commandGate ?? 'ANALYSIS ONLY'}</b></span></div><p className="sim-note">{branchTree.finishComparison?.pairOrderMatchesObserved ? 'The modelled pair order matches the recorded flag order.' : 'The modelled pair order differs from the recorded flag order.'} A better/worse/same full-race finishing position is intentionally not claimed: the branch does not yet model the rest of the grid, BOX decisions, tyre evolution, traffic, retirements, or later race-control events.</p></section>}
        <section className="sim-card"><div className="sim-panel-head"><span>TRACK MARKERS</span><em>PROVENANCE</em></div><div className="sim-facts"><span>CARS ON MAP <b>{currentFrame?.cars?.length ?? 0}</b></span><span>CIRCUIT SOURCE <b>PUBLIC GPS / TELEMETRY</b></span><span>CORNER SOURCE <b>{trackmap?.cornerSource?.includes('fastf1-circuit-info') ? 'NUMBERED CIRCUIT DATA' : trackmap?.cornerSource ? 'FALLBACK CIRCUIT DATA' : 'NOT LOADED'}</b></span><span>TURN / ZONE OVERLAY <b>{visualOverlay?.source === 'USER_VISUAL_TURN_REFERENCE' ? 'VISUAL REFERENCE' : 'NOT LOADED'}</b></span><span>ZONE COMMAND GATE <b>ANALYSIS ONLY</b></span></div><p className="sim-note">Corner labels use the numbered circuit sequence, projected onto the recorded telemetry map. Straight Mode and DET/ACT marks use the supplied turn-range reference only; they are not FIA distance-aligned command lines.</p></section>
      </aside>
    </section>
  </main>
}
