import React, { useEffect, useMemo, useRef, useState } from 'react'
import scenario from '../data/scenarios/las-vegas-2023-lec-per.json'
import { formatLapTime } from './CircuitMap'
import { LapExplorer, fetchJson } from './LapExplorer'
import { TrackRaceMap } from './TrackRaceMap'
import { TelemetryCompare } from './TelemetryCompare'
import {
  computeEnergyTrace,
  attackCostMj,
  recoveryAheadMj,
  ENERGY_MODEL_VERSION,
  REGULATION,
  CALIBRATION,
  DEFAULT_START_RESERVE_PCT,
} from '../lib/energyModel'
import { recommend, feasibilityScore, STRATEGIES, DECISION_ENGINE_VERSION } from '../lib/decisionEngine'
import { RaceSelector, useRaceEngine, StrategyTab, EnergyTab, OvertakeTab } from './DecisionTabs'

const tabs = ['TRACK', 'TELEMETRY', 'STRATEGY', 'ENERGY', 'OVERTAKE', 'LEGENDS']

const { meta, attacker, defender, distance_m: distance, derived } = scenario
const atk = scenario.attacker_telemetry
const def = scenario.defender_telemetry
const driverData = scenario.drivers ?? {}
const attackerRace = driverData[attacker.code] ?? {}
const defenderRace = driverData[defender.code] ?? {}
const attackerFocusLap = (attackerRace.lap_history ?? []).find((lap) => lap.lap === meta.focus_lap) ?? attackerRace.fastest_lap
const defenderFocusLap = (defenderRace.lap_history ?? []).find((lap) => lap.lap === meta.focus_lap) ?? defenderRace.fastest_lap
const lastIndex = distance.length - 1

// The energy trace is deterministic for a given scenario, so it is computed once
// at module load rather than on every scrub.
const energy = computeEnergyTrace({
  distanceM: distance,
  speedKph: atk.speed_kph,
  throttlePct: atk.throttle_pct,
  brakePct: atk.brake_pct,
})

const onTrackPasses = scenario.decision_points.filter((point) => point.on_track_pass)
const excludedPasses = scenario.decision_points.filter((point) => !point.on_track_pass)

// Open on the approach to the heaviest braking zone on the lap, which is the
// Turn 14 zone the pass was actually made into. Picking it by entry speed rather
// than by distance keeps this correct if the scenario is ever re-exported.
const passZone = derived.braking_zones.reduce(
  (best, zone) => (zone.entry_speed_kph > best.entry_speed_kph ? zone : best),
  derived.braking_zones[0],
)
const DEFAULT_FOCUS = Math.max(0, distance.findIndex((m) => m >= passZone.start_m - 110))

const SERIES = {
  speed: { primary: atk.speed_kph, secondary: def.speed_kph, max: 380, min: 0, unit: 'km/h' },
  gap: { primary: derived.gap_s, secondary: null, max: 1.4, min: -0.3, unit: 's' },
  throttle: { primary: atk.throttle_pct, secondary: def.throttle_pct, max: 100, min: 0, unit: '%' },
  brake: { primary: atk.brake_pct, secondary: def.brake_pct, max: 100, min: 0, unit: '%' },
  drs: { primary: atk.drs_active.map((active) => active ? 1 : 0), secondary: def.drs_active.map((active) => active ? 1 : 0), max: 1, min: 0, unit: '' },
  reserve: { primary: energy.reservePct, secondary: null, max: 100, min: 0, unit: '%' },
}

function Chart({ type = 'speed', focus }) {
  const width = 760
  const height = 205
  const pad = { x: 42, y: 20, r: 20, b: 30 }
  const { primary, secondary, min, max, unit } = SERIES[type]

  const path = useMemo(() => {
    const point = (v, i) =>
      `${pad.x + (i / lastIndex) * (width - pad.x - pad.r)},` +
      `${pad.y + (1 - (v - min) / (max - min)) * (height - pad.y - pad.b)}`
    return {
      primary: primary.map(point).join(' '),
      secondary: secondary ? secondary.map(point).join(' ') : null,
    }
  }, [type, primary, secondary, min, max])

  const markerX = pad.x + (focus / lastIndex) * (width - pad.x - pad.r)
  const markerY = pad.y + (1 - (primary[focus] - min) / (max - min)) * (height - pad.y - pad.b)

  return <div className="ov-chart">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${type} trace for lap ${meta.focus_lap}`}>
      <g className="chart-grid">
        <line x1={pad.x} y1={pad.y} x2={width - pad.r} y2={pad.y} />
        <line x1={pad.x} y1={(height - pad.b) / 2} x2={width - pad.r} y2={(height - pad.b) / 2} />
        <line x1={pad.x} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} />
      </g>
      {path.secondary && <polyline className="chart-secondary" points={path.secondary} />}
      <polyline className="chart-primary" points={path.primary} />
      <line className="chart-marker" x1={markerX} y1={pad.y} x2={markerX} y2={height - pad.b} />
      <circle className="chart-marker-dot" cx={markerX} cy={markerY} r="5" />
    </svg>
    <div className="chart-axis">
      <span>{type === 'drs' ? 'OPEN' : `${max}${unit}`}</span>
      <span>{type === 'drs' ? '—' : `${((max + min) / 2).toFixed(unit === 's' ? 2 : 0)}${unit}`}</span>
      <span>{type === 'drs' ? 'CLOSED' : `${min}${unit}`}</span>
      <b>DISTANCE / {meta.lap_length_m.toLocaleString()} m</b>
    </div>
    <div className="chart-legend">
      <span><i className="line-real" /> {attacker.code} {attacker.name.toUpperCase()}</span>
      {secondary && <span><i className="line-reference" /> {defender.code} {defender.name.toUpperCase()}</span>}
      <em>● {Math.round(distance[focus]).toLocaleString()} m</em>
    </div>
  </div>
}

function LapHistoryChart({ driver, selectedLap }) {
  const width = 760
  const height = 205
  const pad = { x: 42, y: 20, r: 20, b: 30 }
  const rows = (driver.lap_history ?? []).filter((row) => row.lap_time_s != null)
  if (!rows.length) return <p className="ov-notes">No lap-time history was available for this driver.</p>
  const values = rows.map((row) => row.lap_time_s)
  const min = Math.min(...values) - 0.5
  const max = Math.max(...values) + 0.5
  const last = Math.max(rows.length - 1, 1)
  const point = (row, index) => `${pad.x + (index / last) * (width - pad.x - pad.r)},${pad.y + (1 - (row.lap_time_s - min) / (max - min)) * (height - pad.y - pad.b)}`
  const selectedIndex = Math.max(0, rows.findIndex((row) => row.lap === selectedLap))
  const selected = rows[selectedIndex] ?? rows[rows.length - 1]
  const selectedPoint = point(selected, selectedIndex)
  const [markerX, markerY] = selectedPoint.split(',')
  return <div className="ov-chart">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${driver.code} lap time history`}>
      <g className="chart-grid">
        <line x1={pad.x} y1={pad.y} x2={width - pad.r} y2={pad.y} />
        <line x1={pad.x} y1={(height - pad.b) / 2} x2={width - pad.r} y2={(height - pad.b) / 2} />
        <line x1={pad.x} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} />
      </g>
      <polyline className="chart-primary" points={rows.map(point).join(' ')} />
      <line className="chart-marker" x1={markerX} y1={pad.y} x2={markerX} y2={height - pad.b} />
      <circle className="chart-marker-dot" cx={markerX} cy={markerY} r="5" />
    </svg>
    <div className="chart-axis"><span>{formatLapTime(max)}</span><span>{formatLapTime((max + min) / 2)}</span><span>{formatLapTime(min)}</span><b>LAP NUMBER / {driver.race_laps ?? rows.length}</b></div>
    <div className="chart-legend"><span><i className="line-real" /> {driver.code} REAL LAP TIMES</span><em>L{selected.lap} · {formatLapTime(selected.lap_time_s)} · {selected.compound ?? 'TYRE N/A'}</em></div>
  </div>
}

function DataBadge({ children, tone = 'real' }) {
  return <span className={`data-badge ${tone}`}><i />{children}</span>
}

export function StrategyDashboard({ onHome }) {
  const [tab, setTab] = useState('STRATEGY')
  const [telemetryLaps, setTelemetryLaps] = useState([])
  const [explorerSel, setExplorerSel] = useState(null)
  const [explorerMaps, setExplorerMaps] = useState({})
  const explorerMapRequested = useRef(new Set())
  const focus = DEFAULT_FOCUS
  const [strategy, setStrategy] = useState('ATTACK')
  const [drsOverride, setDrsOverride] = useState(null)
  const [raceSel, setRaceSel] = useState({ year: 2023, round: 21, session: 'R', driver: 'LEC' })
  const engine = useRaceEngine(raceSel, tab)

  // When a season's extracted rounds arrive and exclude the current round
  // (e.g. switching to 2026, which has fewer completed races), snap to the
  // nearest available round so the panels never land on an empty state.
  // Keyed on the round list only, so manual round picks are never overridden.
  useEffect(() => {
    const rounds = engine.events ?? []
    if (!rounds.length || rounds.some((e) => e.round === raceSel.round)) return
    const nearest = rounds.reduce((best, e) =>
      (Math.abs(e.round - raceSel.round) < Math.abs(best.round - raceSel.round) ? e : best), rounds[0])
    setRaceSel((s) => ({ ...s, round: nearest.round }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine.events])

  const selKey = explorerSel ? `${explorerSel.year}:${explorerSel.round}:${explorerSel.sessionName}` : null
  const explorerMap = selKey ? explorerMaps[selKey] : null

  useEffect(() => {
    if (!explorerSel) return
    const key = `${explorerSel.year}:${explorerSel.round}:${explorerSel.sessionName}`
    if (explorerMapRequested.current.has(key)) return
    explorerMapRequested.current.add(key)
    setExplorerMaps((prev) => ({ ...prev, [key]: { loading: true } }))
    fetchJson(`/api/f1/trackmap?year=${explorerSel.year}&round=${explorerSel.round}&session=${encodeURIComponent(explorerSel.sessionName)}`)
      .then((data) => setExplorerMaps((prev) => ({ ...prev, [key]: data.points?.length ? data : { error: data.error || 'no position data' } })))
      .catch((err) => setExplorerMaps((prev) => ({ ...prev, [key]: { error: err.message } })))
  }, [explorerSel])

  const gap = derived.gap_s[focus]
  const speed = atk.speed_kph[focus]
  const speedDelta = derived.speed_delta_kph[focus]
  const drsReal = atk.drs_active[focus]
  const drs = drsOverride === null ? drsReal : drsOverride
  const reserve = energy.reservePct[focus]

  const recovery = useMemo(
    () => recoveryAheadMj(derived.braking_zones, distance[focus], speed),
    [focus, speed],
  )
  const distanceToZone = recovery.nextZoneM === null ? null : recovery.nextZoneM - distance[focus]

  const decision = useMemo(
    () => recommend({
      gapS: gap,
      closingRateKph: speedDelta,
      drsActive: drs,
      reservePct: reserve,
      distanceToBrakingZoneM: distanceToZone,
    }),
    [gap, speedDelta, drs, reserve, distanceToZone],
  )
  const feasibility = feasibilityScore({ gapS: gap, closingRateKph: speedDelta, drsActive: drs, reservePct: reserve })
  const cost = attackCostMj()

  return <main className="ov-dashboard">
    <header className="ov-header">
      <button type="button" className="ov-brand" onClick={onHome} aria-label="Back to home">
        <span>✦</span>
        <strong>PIT<em>WOLF</em></strong>
        <small>RACE STRATEGY INTELLIGENCE</small>
      </button>
      <div className="ov-header-center">
        <b>{meta.event_date.slice(0, 4)} {meta.event.toUpperCase()}</b>
        <span>{meta.session.toUpperCase()} / LAP {meta.focus_lap} OF {meta.total_laps} / {meta.circuit.toUpperCase()}</span>
      </div>
      <div className="ov-header-right"><DataBadge tone="real">FASTF1 {meta.fastf1_version}</DataBadge><button className="ov-menu">☰</button></div>
    </header>

    <section className="ov-toolbar">
      <div className="ov-select"><span>ATTACKER</span><b>{attacker.name.toUpperCase()}</b><i>{attacker.team}</i></div>
      <div className="ov-select"><span>DEFENDER</span><b>{defender.name.toUpperCase()}</b><i>{defender.team}</i></div>
      <div className="ov-select"><span>SCENARIO</span><b>{meta.title.toUpperCase()}</b><i>P{defender.finish_position} → P{attacker.finish_position}</i></div>
      <div className="ov-toolbar-note"><span>SOURCE</span><b>{meta.source.toUpperCase()}</b></div>
    </section>

    <nav className="ov-tabs">
      {tabs.map((item) => <button key={item} className={tab === item ? 'active' : ''} onClick={() => setTab(item)}>
        {item === 'TELEMETRY' && telemetryLaps.length ? `TELEMETRY · ${telemetryLaps.length} LAP${telemetryLaps.length === 1 ? '' : 'S'}` : item}
      </button>)}
      <span className="ov-tab-caveat">Cached {meta.event_date} session · every value labelled by source</span>
    </nav>

    <section className="ov-content">
      <div className="ov-title-row">
        <div>
          <p className="ov-eyebrow">SCENARIO ANALYSIS / {tab}</p>
          <h1>Should we spend<br /><em>energy here?</em></h1>
        </div>
        <div className="ov-scenario-summary">
          <DataBadge tone="real">REAL RACE CONTEXT</DataBadge>
          <strong>{attacker.code} <span>vs</span> {defender.code}</strong>
          <p>Lap {meta.focus_lap} · {meta.lap_length_m.toLocaleString()} m · {meta.circuit}</p>
        </div>
      </div>

      {['STRATEGY', 'ENERGY', 'OVERTAKE'].includes(tab) && (
        <RaceSelector sel={raceSel} onChange={setRaceSel} drivers={engine.drivers} events={engine.events} />
      )}

      {tab === 'STRATEGY' && (
        <StrategyTab sel={raceSel} decision={engine.decision} preds={engine.preds} energy={engine.energy} />
      )}

      {tab === 'TRACK' && <>
      <div className="ov-track-layout">
        <article className="ov-panel ov-track-card">
          <div className="ov-panel-head">
            <span>{explorerSel
              ? `${(explorerSel.event?.name ?? 'CIRCUIT').toUpperCase()} / ${explorerSel.sessionName.toUpperCase()}`
              : 'CIRCUIT / SELECTED SESSION'}</span>
            <DataBadge tone="real">REAL GPS</DataBadge>
          </div>
          {explorerMap?.points?.length > 0 && <TrackRaceMap trackmap={explorerMap} loadedLaps={[]} cursorT={0} />}
          {(!explorerMap || explorerMap.loading) && <div className="lx-loading"><span className="lx-spinner" />LOADING CIRCUIT MAP</div>}
          {explorerMap?.error && <p className="lx-empty">No position data for this session.</p>}
          <div className="ov-track-readout">
            <b>{explorerMap?.trackLength ? `${Math.round(explorerMap.trackLength).toLocaleString()} m` : '—'}</b>
            <span>{explorerSel?.event
              ? `${String(explorerSel.event.country).toUpperCase()} · ${String(explorerSel.event.location).toUpperCase()}`
              : 'FOLLOWING LAP EXPLORER SELECTION'}</span>
          </div>
        </article>

        <aside className="ov-panel ov-timeline-panel">
          <div className="ov-panel-head"><span>LAP TIMELINE</span><DataBadge tone="real">SECTOR TIMES</DataBadge></div>
          <div className="ov-timeline-line"><i style={{ height: `${(focus / lastIndex) * 100}%` }} /></div>
          <ol>
            {[
              { label: 'GRID EXIT', index: 0, value: '00:00' },
              { label: 'S1', index: scenario.timing.markers.find((m) => m.label === 'S1')?.index ?? 0, value: `${scenario.timing.attacker.sector_1_s.toFixed(3)}s` },
              { label: 'S2', index: scenario.timing.markers.find((m) => m.label === 'S2')?.index ?? 0, value: `${scenario.timing.attacker.sector_2_s.toFixed(3)}s` },
              { label: 'TURN 14', index: DEFAULT_FOCUS, value: `${Math.round(distance[DEFAULT_FOCUS]).toLocaleString()} m` },
              { label: 'FINISH', index: lastIndex, value: formatLapTime(scenario.timing.attacker.lap_time_s) },
            ].map((item) => <li key={item.label} className={focus >= item.index ? 'is-passed' : ''}>
              <span>{item.label}</span><b>{item.value}</b>
            </li>)}
          </ol>
          <div className="ov-pit-extra">
            <span>PIT EXTRA TIMING</span>
            {scenario.pit_extras.map((pit) => <div key={`${pit.driver}-${pit.in_lap}`}>
              <b>{pit.driver}</b>
              <em>L{pit.in_lap}→{pit.out_lap}</em>
              <strong>{pit.stationary_s.toFixed(3)}s</strong>
              <small>{pit.compound_in} → {pit.compound_out}</small>
            </div>)}
            <p>Stationary time from official pit-in to pit-out. Laps 22 and 27 position swaps were excluded as these pit cycles.</p>
          </div>
          <div className="ov-pit-extra">
            <span>TYRE STINTS / REAL TIMING</span>
            {[attackerRace, defenderRace].map((driver) => <div key={driver.code}>
              <b>{driver.code}</b>
              <em>{(driver.tyre_stints ?? []).map((stint) => `${stint.compound} L${stint.lap_start}–${stint.lap_end}`).join(' · ') || 'N/A'}</em>
            </div>)}
            {scenario.weather_summary?.available && <p>Weather: track {scenario.weather_summary.ranges.track_temp_c.min}–{scenario.weather_summary.ranges.track_temp_c.max}°C · air {scenario.weather_summary.ranges.air_temp_c.min}–{scenario.weather_summary.ranges.air_temp_c.max}°C.</p>}
          </div>
        </aside>
      </div>

      <LapExplorer
        selectedLaps={telemetryLaps}
        onLapsChange={setTelemetryLaps}
        onOpenTelemetry={() => setTab('TELEMETRY')}
        onSelectionChange={setExplorerSel}
      />
      </>}

      {tab === 'TELEMETRY' && <TelemetryCompare selectedLaps={telemetryLaps} onLapsChange={setTelemetryLaps} />}

      {tab === 'ENERGY' && (
        <EnergyTab sel={raceSel} energy={engine.energy} />
      )}

      {tab === 'OVERTAKE' && (
        <OvertakeTab sel={raceSel} decision={engine.decision} preds={engine.preds} report={engine.report} />
      )}

      {tab === 'LEGENDS' && <div className="ov-legend-grid">
        <section className="ov-panel">
          <div className="ov-panel-head"><span>DATA PROVENANCE</span><b>READ THIS FIRST</b></div>
          <div className="ov-legend-item"><DataBadge tone="real">REAL</DataBadge><p>Loaded from FastF1 official timing and car telemetry: {meta.provenance.real.join(', ')}.</p></div>
          <div className="ov-legend-item"><DataBadge tone="derived">DERIVED</DataBadge><p>Calculated from those real values: {meta.provenance.derived.join(', ')}.</p></div>
          <div className="ov-legend-item"><DataBadge tone="simulated">SIMULATED</DataBadge><p>Produced by energy model {ENERGY_MODEL_VERSION}. ERS deployment and battery state of charge are not public and are never presented as measured team data.</p></div>
          <div className="ov-panel-head second"><span>SESSION COVERAGE</span><DataBadge tone="real">REAL FASTF1</DataBadge></div>
          <div className="ov-factor"><span>Weather samples</span><b className="positive">{scenario.weather_summary?.samples?.toLocaleString() ?? '—'}</b><em>Air, track, humidity, pressure and wind ranges included.</em></div>
          <div className="ov-factor"><span>Race-control messages</span><b className="positive">{scenario.race_control?.length ?? 0}</b><em>Flags, DRS notices, safety-car context and lap references where supplied.</em></div>
        </section>
        <section className="ov-panel">
          <div className="ov-panel-head"><span>FASTEST RACE LAPS</span><DataBadge tone="real">REAL TIMING</DataBadge></div>
          {(scenario.fastest_laps ?? []).slice(0, 5).map((lap, index) => <div className="ov-factor" key={`${lap.Driver}-${lap.LapNumber}`}>
            <span>#{index + 1} · {lap.Driver} · lap {lap.LapNumber}</span>
            <b className={lap.Driver === attacker.code ? 'positive' : ''}>{formatLapTime(lap.LapTime)}</b>
            <em>{lap.Compound ?? 'TYRE N/A'} · tyre life {lap.TyreLife ?? '—'} · speed FL {lap.SpeedFL ?? '—'} km/h</em>
          </div>)}
          <div className="ov-panel-head second"><span>DECISION POINTS IN THIS RACE</span><b>OBSERVABLE GROUND TRUTH</b></div>
          {onTrackPasses.map((point) => <div className="ov-factor" key={point.lap}>
            <span>Lap {point.lap} · {point.gained_position} took the position</span>
            <b className="positive">{point.gap_before_s}s before</b>
            <em>{point.attacker_tyre.compound} {point.attacker_tyre.age_laps}L vs {point.defender_tyre.compound} {point.defender_tyre.age_laps}L</em>
          </div>)}
          <p className="ov-notes">
            {excludedPasses.length} further position swaps (laps {excludedPasses.map((p) => p.lap).join(', ')}) were
            excluded as pit-stop cycles rather than on-track passes. Counting them would inflate the label set.
          </p>
          <p className="ov-notes">
            Scrub the lap to inspect any point. Every chart, metric and recommendation follows the selected distance.
          </p>
        </section>
      </div>}

    </section>

    <footer className="ov-footer">
      <span>PITWOLF / {meta.scenario_id}</span>
      <span>REAL TELEMETRY · DERIVED FEATURES · ENERGY MODEL {ENERGY_MODEL_VERSION} · ENGINE {DECISION_ENGINE_VERSION}</span>
    </footer>
  </main>
}
