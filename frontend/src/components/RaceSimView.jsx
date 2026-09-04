import React, { useEffect, useRef, useState, useMemo } from 'react'
import scenario from '../data/scenarios/las-vegas-2023-lec-per.json'
import { CircuitMap, formatLapTime } from './CircuitMap'
import '../racesim.css'

const { meta, attacker, defender, distance_m: distance, derived } = scenario
const atk       = scenario.attacker_telemetry
const def       = scenario.defender_telemetry
const lastIndex = distance.length - 1

const TOTAL_LAPS    = Number(meta.total_laps ?? 50)
const SPEEDS        = [1, 2, 4, 8]
const RACE_ROUND    = 21
const RACE_SESSION  = 'R'

const selectedDriverHistory = scenario.drivers?.[attacker.code]?.lap_history ?? []
const selectedLaps = new Map(selectedDriverHistory.map((lap) => [Number(lap.lap), lap]))

function telemetryForLap(payload, fallback) {
  const trace = payload?.trace
  if (!trace?.time?.length || trace.x?.length !== trace.time.length || trace.y?.length !== trace.time.length) return fallback
  return {
    ...fallback,
    speed_kph: trace.speed,
    throttle_pct: trace.throttle,
    brake_pct: trace.brake,
    gear: trace.gear,
    rpm: trace.rpm,
    drs_active: trace.drs,
    x: trace.x,
    y: trace.y,
    elapsed_s: trace.time,
  }
}

// Sector marker indices
const S1_IDX = scenario.timing.markers.find((m) => m.label === 'S1')?.index ?? 65
const S2_IDX = scenario.timing.markers.find((m) => m.label === 'S2')?.index ?? 131
const S3_IDX = scenario.timing.markers.find((m) => m.label === 'S3')?.index ?? 239

const BASE_S1  = scenario.timing.attacker.sector_1_s
const BASE_S2  = scenario.timing.attacker.sector_2_s
const BASE_S3  = scenario.timing.attacker.sector_3_s
const BASE_LAP = BASE_S1 + BASE_S2 + BASE_S3

const passZone = derived.braking_zones.reduce(
  (best, zone) => (zone.entry_speed_kph > best.entry_speed_kph ? zone : best),
  derived.braking_zones[0],
)
const PASS_INDEX = Math.max(0, distance.findIndex((m) => m >= passZone.start_m - 110))

// ─────────────────────────────────────────────────────────────────────────────
// Overtake logic
// ─────────────────────────────────────────────────────────────────────────────
const OVERTAKE_ZONES_S = [
  [0, 10],   // SF to T1
  [15, 27],  // T2 to T3
  [45, 78],  // Strip
  [85, 96]   // Pit straight
]

function isInOvertakeZone(t) {
  return OVERTAKE_ZONES_S.some(([start, end]) => t >= start && t <= end)
}

const OVERTAKE_ZONE_SEGS = OVERTAKE_ZONES_S.map(([start, end]) => {
  let sIdx = 0; while (sIdx < atk.elapsed_s.length && atk.elapsed_s[sIdx] < start) sIdx++;
  let eIdx = sIdx; while (eIdx < atk.elapsed_s.length && atk.elapsed_s[eIdx] <= end) eIdx++;
  return { start: Math.max(0, sIdx - 1), end: Math.min(lastIndex, eIdx) };
})

const CLOSE_RATE_ZONE = 0.06
const CLOSE_RATE_CORN = 0.015
const OPEN_RATE       = 0.12
const INITIAL_GAP     = 1.2
const MAX_GAP         = 1.8

// ─────────────────────────────────────────────────────────────────────────────
// Lap timing replay
// ─────────────────────────────────────────────────────────────────────────────
function buildLapTimes() {
  const times = new Array(TOTAL_LAPS + 1)
  for (let lap = 1; lap <= TOTAL_LAPS; lap++) {
    const observed = Number(selectedLaps.get(lap)?.lap_time_s)
    times[lap] = Number.isFinite(observed) && observed > 50 ? observed : BASE_LAP
  }
  return times
}
const LAP_TIMES = buildLapTimes()

function lapSectorTimes(lap) {
  const observed = selectedLaps.get(lap)
  if ([observed?.sector_1_s, observed?.sector_2_s, observed?.sector_3_s].every((value) => Number.isFinite(Number(value)))) {
    return { s1: Number(observed.sector_1_s), s2: Number(observed.sector_2_s), s3: Number(observed.sector_3_s) }
  }
  const total = LAP_TIMES[lap]
  const delta = total - BASE_LAP
  return {
    s1: BASE_S1 + delta * (BASE_S1 / BASE_LAP),
    s2: BASE_S2 + delta * (BASE_S2 / BASE_LAP),
    s3: BASE_S3 + delta * (BASE_S3 / BASE_LAP),
  }
}

function toFocusFrac(t, elapsed) {
  if (t <= 0) return 0
  if (t >= elapsed[elapsed.length - 1]) return elapsed.length - 1

  let lo = 0
  let hi = elapsed.length - 2
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (elapsed[mid] <= t) lo = mid
    else hi = mid - 1
  }
  const span = elapsed[lo + 1] - elapsed[lo]
  const frac = span > 0 ? (t - elapsed[lo]) / span : 0
  return lo + Math.max(0, Math.min(1, frac))
}

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────
export function RaceSimView({ onOpenDashboard, onHome }) {
  const [currentLap,   setCurrentLap]   = useState(1)
  const [fastestLap,   setFastestLap]   = useState(null)
  const [lastLapTime,  setLastLapTime]  = useState(null)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [lapTelemetry, setLapTelemetry] = useState({ attacker: atk, defender: def, lap: 1, real: true })

  const [lecFrac, setLecFrac] = useState(0)
  const [perFrac, setPerFrac] = useState(0)
  const [gapState, setGapState] = useState({ leader: 'PER', gapS: INITIAL_GAP, inZone: false })

  const lapRef          = useRef(1)
  const simTimeRef      = useRef(0)
  const playbackRateRef = useRef(1)

  const gapRef          = useRef(INITIAL_GAP)
  const leaderRef       = useRef('PER')
  const postOvertakeRef = useRef(false)
  const telemetryRef    = useRef(lapTelemetry)

  useEffect(() => {
    telemetryRef.current = lapTelemetry
  }, [lapTelemetry])

  useEffect(() => {
    if (currentLap === 1) return undefined
    let cancelled = false
    const query = (driver) => fetch(`/api/f1/telemetry?year=2023&round=${RACE_ROUND}&session=${RACE_SESSION}&driver=${driver}&lap=${currentLap}`)
      .then((response) => response.ok ? response.json() : null)
      .catch(() => null)

    Promise.all([query(attacker.code), query(defender.code)]).then(([attackerLap, defenderLap]) => {
      if (cancelled) return
      const nextAttacker = telemetryForLap(attackerLap, atk)
      const nextDefender = telemetryForLap(defenderLap, def)
      setLapTelemetry({
        attacker: nextAttacker,
        defender: nextDefender,
        lap: currentLap,
        real: nextAttacker !== atk && nextDefender !== def,
      })
    })

    return () => { cancelled = true }
  }, [currentLap])

  useEffect(() => {
    simTimeRef.current = 0
    lapRef.current     = 1
    gapRef.current     = INITIAL_GAP
    leaderRef.current  = 'PER'
    postOvertakeRef.current = false

    let frame
    let last = performance.now()

    const tick = (now) => {
      const dt = ((now - last) / 1000) * playbackRateRef.current
      last = now
      simTimeRef.current += dt

      const lapDuration = LAP_TIMES[lapRef.current] || BASE_LAP
      if (simTimeRef.current >= lapDuration) {
        const completedLap  = lapRef.current
        const completedTime = LAP_TIMES[completedLap]

        simTimeRef.current = 0
        const nextLap  = completedLap >= TOTAL_LAPS ? 1 : completedLap + 1
        lapRef.current = nextLap

        setLastLapTime(completedTime)
        setFastestLap((prev) => {
          if (!prev || completedTime < prev.time) {
            return { lap: completedLap, time: completedTime }
          }
          return prev
        })
        setCurrentLap(nextLap)
      }

      // ── Gap Dynamics ──
      const inZone = isInOvertakeZone(simTimeRef.current)
      
      if (postOvertakeRef.current) {
        // Leader pulls away
        gapRef.current += OPEN_RATE * dt
        if (gapRef.current >= MAX_GAP) {
          postOvertakeRef.current = false
        }
      } else {
        // Follower closing
        const closeRate = inZone ? CLOSE_RATE_ZONE : CLOSE_RATE_CORN
        gapRef.current -= closeRate * dt

        if (gapRef.current <= 0) {
          if (inZone) {
            gapRef.current = 0
            leaderRef.current = leaderRef.current === 'PER' ? 'LEC' : 'PER'
            postOvertakeRef.current = true
          } else {
            gapRef.current = 0.02 // blocked side-by-side
          }
        }
      }

      // ── Compute positions ──
      const leaderTime = simTimeRef.current
      const followerTime = Math.max(0, leaderTime - gapRef.current)

      const currentTelemetry = telemetryRef.current
      const fLeader = toFocusFrac(leaderTime, currentTelemetry.attacker.elapsed_s)
      const fFollower = toFocusFrac(followerTime, currentTelemetry.attacker.elapsed_s)

      if (leaderRef.current === 'PER') {
        setPerFrac(fLeader)
        setLecFrac(fFollower)
      } else {
        setLecFrac(fLeader)
        setPerFrac(fFollower)
      }

      setGapState({ leader: leaderRef.current, gapS: gapRef.current, inZone })

      frame = requestAnimationFrame(tick)
    }

    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [])

  const leaderFrac = gapState.leader === 'PER' ? perFrac : lecFrac
  const focus    = Math.floor(leaderFrac)
  const speed    = atk.speed_kph[Math.min(focus, lastIndex)]
  const lapTime  = formatLapTime(simTimeRef.current)

  const s1Passed = focus >= S1_IDX
  const s2Passed = focus >= S2_IDX
  const s3Passed = focus >= S3_IDX

  const sectors  = useMemo(() => lapSectorTimes(currentLap), [currentLap])
  const anyLapDone = lastLapTime !== null

  // ── Legend formatting ──
  const isBattle = gapState.gapS < 0.5 && !gapState.inZone
  const isDRS    = gapState.gapS < 0.5 && gapState.inZone
  
  let gapClass = ''
  let gapLabel = `GAP +${gapState.gapS.toFixed(3)}s`
  if (isDRS) {
    gapClass = 'rs-gap-drs'
    gapLabel = `⚡ OVERTAKE +${gapState.gapS.toFixed(3)}s`
  } else if (isBattle) {
    gapClass = 'rs-gap-battle'
    gapLabel = `⚠ BATTLE +${gapState.gapS.toFixed(3)}s`
  }

  return (
    <div className="rs-root">
      <header className="rs-header">
        <button type="button" className="rs-brand" onClick={onHome} aria-label="Back to home">
          <span>✦</span>
          <strong>PIT<em>WOLF</em></strong>
        </button>

        <div className="rs-race-info">
          <b>LAS VEGAS GRAND PRIX 2023</b>
          <span>RACE · {meta.circuit.toUpperCase()} · {meta.lap_length_m.toLocaleString()} m</span>
        </div>

        <div className="rs-lap-counter">
          <span>LAP</span>
          <b>{currentLap}</b>
          <small>/ {TOTAL_LAPS}</small>
        </div>

        <div className="rs-speed-group">
          {SPEEDS.map((s) => (
            <button
              key={s}
              className={`rs-speed-btn ${playbackRate === s ? 'active' : ''}`}
              onClick={() => {
                playbackRateRef.current = s
                setPlaybackRate(s)
              }}
            >
              {s}×
            </button>
          ))}
        </div>

        <button className="rs-dashboard-btn" onClick={onOpenDashboard}>
          STRATEGY DASHBOARD ↗
        </button>
      </header>

      <div className="rs-body">
        <div className="rs-map-panel">
          <div className="rs-map-label">
            <span>{meta.circuit.toUpperCase()} / REAL-RACE REPLAY</span>
            <span className="rs-badge real">● REAL GPS · L{lapTelemetry.lap}</span>
          </div>

          <div className="rs-map-wrap">
            <CircuitMap
              attacker={lapTelemetry.attacker}
              defender={lapTelemetry.defender}
              lecFrac={lecFrac}
              perFrac={perFrac}
              circuitName={meta.circuit}
              passIndex={PASS_INDEX}
            />
          </div>

          <div className="rs-map-readout">
            <b>{lapTime}</b>
            <span>REAL LAP {currentLap} · {gapState.leader} {speed.toFixed(0)} km/h</span>
          </div>

          <div className="rs-map-legend">
            <span className={gapState.leader === 'LEC' ? 'rs-leading' : ''}>
              <i className="rs-dot lec" /> LEC
            </span>
            <span className={gapState.leader === 'PER' ? 'rs-leading' : ''}>
              <i className="rs-dot per" /> PER
            </span>
            <em className={gapClass}>{gapLabel} · MODELLED GAP</em>
          </div>
        </div>

        <div className="rs-timeline-panel">
          <div className="rs-panel-head">
            <span>LAP TIMELINE</span>
            <span className="rs-badge real">● SECTOR TIMES</span>
          </div>

          <div className="rs-timeline-track">
            <div className="rs-timeline-line">
              <i style={{ height: `${(leaderFrac / lastIndex) * 100}%` }} />
            </div>

            <ol className="rs-timeline-items">
              <li className={anyLapDone ? 'is-passed is-fastest' : ''}>
                <div className="rs-tl-dot" />
                <span>
                  FASTEST LAP
                  {fastestLap && <em className="rs-tl-sub"> · L{fastestLap.lap}</em>}
                </span>
                <b className="rs-fastest-val">
                  {fastestLap ? formatLapTime(fastestLap.time) : '—'}
                </b>
              </li>

              <li className={s1Passed ? 'is-passed' : ''}>
                <div className="rs-tl-dot" />
                <span>S1</span>
                <b>{s1Passed ? `${sectors.s1.toFixed(3)}s` : '—'}</b>
              </li>

              <li className={s2Passed ? 'is-passed' : ''}>
                <div className="rs-tl-dot" />
                <span>S2</span>
                <b>{s2Passed ? `${sectors.s2.toFixed(3)}s` : '—'}</b>
              </li>

              <li className={s3Passed ? 'is-passed' : ''}>
                <div className="rs-tl-dot" />
                <span>S3</span>
                <b>{s3Passed ? `${sectors.s3.toFixed(3)}s` : '—'}</b>
              </li>

              <li className={anyLapDone ? 'is-passed' : ''}>
                <div className="rs-tl-dot" />
                <span>LAST LAP TIME</span>
                <b>{lastLapTime ? formatLapTime(lastLapTime) : '—'}</b>
              </li>
            </ol>
          </div>

          <div className="rs-pit-section">
            <span>PIT EXTRA TIMING</span>
            {scenario.pit_extras.map((pit) => (
              <div key={`${pit.driver}-${pit.in_lap}`} className="rs-pit-item">
                <b>{pit.driver}</b>
                <em>L{pit.in_lap}→{pit.out_lap}</em>
                <strong>{pit.stationary_s.toFixed(3)}s</strong>
                <small>{pit.compound_in} → {pit.compound_out}</small>
              </div>
            ))}
            <p>
              Stationary time from official pit-in to pit-out. Laps 22 and 27
              position swaps were excluded as these pit cycles.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
