import React, { useEffect, useMemo, useRef, useState } from 'react'
import { formatLapTime, project } from './CircuitMap'
import { fetchJson } from './LapExplorer'
import '../teamenergy.css'
import '../genericlapsim.css'
import '../genericlapshud.css'
import '../optimal.css'

const YEARS = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018]

const ZONES = {
  BRAKING_RECOVERY: { label: 'BRAKING RECOVERY', color: '#ff625d' },
  LIFT_COAST: { label: 'LIFT + COAST', color: '#b94545' },
  SUPERCLIPPING: { label: 'SUPERCLIPPING', color: '#ffb13d' },
  ICE_FULL: { label: 'ICE-ONLY FULL THROTTLE', color: '#41d59d' },
  ERS_DEPLOY: { label: 'INFERRED ERS DEPLOYMENT', color: '#7194ff' },
  FULL_DEPLOY: { label: 'INFERRED NEAR-350 kW ERS DEPLOYMENT', color: '#3e73f1' },
  ON_POWER: { label: 'ICE PROPULSION / NO ERS DEPLOYMENT', color: '#41d59d' },
}

const fmtDelta = (value) => value == null || !Number.isFinite(value)
  ? '—'
  : `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(3)}s`

const deploymentPolicyDisclosure = (summary) => {
  const policy = summary?.qualifyingBatteryDeploymentPolicy
  if (!policy?.enforced) return null
  if (policy.zoneCount > 0) return <p className="te-disclosure"><b>USER STRAIGHT-MODE BATTERY POLICY</b> · Qualifying ERS deployment is limited to {policy.zoneCount} {String(policy.mappingBasis ?? '').startsWith('PITWOLF_') ? 'user-confirmed circuit-topology' : 'supplied'} range{policy.zoneCount === 1 ? '' : 's'}. The FIA qualifying power curve still caps it inside those ranges; this placement policy is not FIA activation data.</p>
  return <p className="te-disclosure"><b>USER STRAIGHT-MODE BATTERY POLICY</b> · No supplied circuit range is available for this event, so qualifying ERS deployment is disabled rather than inferred elsewhere. Braking recovery remains modelled separately.</p>
}

const valueAtDistance = (trace, field, distance) => {
  const distances = trace?.distance ?? []
  const values = trace?.[field] ?? []
  if (!distances.length || !values.length) return 0
  let lo = 0
  let hi = distances.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (distances[mid] < distance) lo = mid + 1
    else hi = mid
  }
  return values[lo] ?? values[values.length - 1] ?? 0
}

function classifyZone(energyTrace, telemetryTrace, distance) {
  const harvest = valueAtDistance(energyTrace, 'pHarvestKw', distance)
  const electric = valueAtDistance(energyTrace, 'pElecKw', distance)
  const ice = valueAtDistance(energyTrace, 'pIceKw', distance)
  const clipping = Boolean(valueAtDistance(energyTrace, 'clipping', distance))
  const brake = Boolean(valueAtDistance(telemetryTrace, 'brake', distance))
  const throttle = valueAtDistance(telemetryTrace, 'throttle', distance)
  const speed = valueAtDistance(telemetryTrace, 'speed', distance)

  if (brake && harvest >= 8) return 'BRAKING_RECOVERY'
  // A zone that is actually drawing ERS-K must remain blue. Clipping is a
  // shortfall/depletion warning, not a replacement for real deployment.
  if (electric >= 332.5) return 'FULL_DEPLOY'
  // Blue is reserved for electrical deployment. Green must never imply that
  // the model is drawing meaningful energy from the usable SoC window.
  if (electric >= 1) return 'ERS_DEPLOY'
  if (clipping) return 'SUPERCLIPPING'
  if (!brake && throttle < 35 && speed > 90) return 'LIFT_COAST'
  if (throttle >= 90 && ice > 20 && electric < 1) return 'ICE_FULL'
  return 'ON_POWER'
}

function linePath(values, width, height, min, max) {
  if (!values?.length) return ''
  const span = Math.max(1e-9, max - min)
  return values.map((value, index) => {
    const x = 34 + (index / Math.max(1, values.length - 1)) * (width - 48)
    const y = 14 + (1 - (Math.max(min, Math.min(max, value)) - min) / span) * (height - 32)
    return `${index ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ')
}

function projectedPointAtDistance(projected, distances, distance) {
  let index = distances.findIndex((item) => item >= distance)
  if (index <= 0) return projected[0]
  if (index < 0) return projected[projected.length - 1]
  const from = projected[index - 1]
  const to = projected[index]
  const span = Math.max(1e-6, distances[index] - distances[index - 1])
  const fraction = Math.max(0, Math.min(1, (distance - distances[index - 1]) / span))
  return { x: from.x + (to.x - from.x) * fraction, y: from.y + (to.y - from.y) * fraction }
}

// The track geometry does not change during animation. Isolating it prevents
// hundreds of SVG segments being reconciled on every animation frame.
const EnergyMapLayers = React.memo(function EnergyMapLayers({ outline, projected, colors, corners }) {
  return <>
    <polyline points={outline} fill="none" stroke="#21312f" strokeWidth="15" strokeLinecap="round" strokeLinejoin="round" />
    {projected.slice(1).map((point, index) => <line
      key={index}
      x1={projected[index].x.toFixed(1)} y1={projected[index].y.toFixed(1)}
      x2={point.x.toFixed(1)} y2={point.y.toFixed(1)}
      stroke={ZONES[colors[index + 1]]?.color ?? ZONES.ON_POWER.color}
      strokeWidth="9" strokeLinecap="round"
    />)}
    {corners.map((corner) => <g key={corner.n} className="te-corner"><circle cx={corner.x} cy={corner.y} r="7" /><text x={corner.x} y={corner.y + 3} textAnchor="middle">{corner.n}</text></g>)}
  </>
})

function MiniTrace({ title, values, color, max, unit }) {
  const width = 520
  const height = 126
  const top = max ?? Math.max(1, ...values)
  return <section className="te-trace">
    <div><span>{title}</span><b>{values?.length ? `${Math.round(values.at(-1) ?? 0)}${unit}` : '—'}</b></div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
      {[0.2, 0.5, 0.8].map((factor) => <line key={factor} x1="34" x2={width - 14} y1={14 + factor * (height - 32)} y2={14 + factor * (height - 32)} />)}
      <path d={linePath(values, width, height, 0, top)} stroke={color} />
    </svg>
    <small>TRACK DISTANCE →</small>
  </section>
}

function EnergyMap({ trackmap, energy, telemetry, mapMode, onMapModeChange, genericLabel, runnerDistance, hideSwitch = false, overlay = null }) {
  const { projected, distances, colors, renderedCorners } = useMemo(() => {
    const points = trackmap?.points ?? []
    if (!points.length) return {}
    const projectedPoints = project(points.map((point) => point.x), points.map((point) => point.y))
    const zoneKeys = points.map((point) => classifyZone(energy?.trace, telemetry?.trace, point.d))
    const pointDistances = points.map((point) => point.d)
    return {
      projected: projectedPoints,
      distances: pointDistances,
      colors: zoneKeys,
      renderedCorners: (trackmap.corners ?? []).map((corner) => ({ ...corner, ...projectedPointAtDistance(projectedPoints, pointDistances, corner.d) })),
    }
  }, [trackmap, energy, telemetry])

  if (!projected) return <div className="te-loading">LOADING TRACK DISTANCE MAP…</div>
  const outline = projected.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(' ')
  const runner = Number.isFinite(runnerDistance) ? projectedPointAtDistance(projected, distances, runnerDistance) : null
  return <section className="te-map-panel">
    <div className="te-map-topline">
      <span>MODELLED ENERGY-STATE MAP</span>
      {!hideSwitch && <div className="te-map-switch" role="group" aria-label="Energy map reference">
        <button type="button" className={mapMode === 'TEAM' ? 'active' : ''} onClick={() => onMapModeChange('TEAM')}>TEAM TRACE</button>
        <button type="button" className={mapMode === 'GENERIC' ? 'active' : ''} onClick={() => onMapModeChange('GENERIC')}>GENERIC REF</button>
      </div>}
    </div>
    <div className="te-map-stage"><svg viewBox="0 0 640 280" role="img" aria-label="Modelled energy state map">
      <EnergyMapLayers outline={outline} projected={projected} colors={colors} corners={renderedCorners} />
      {runner && <g className="te-sim-runner" aria-label="Simulated qualifying car"><circle cx={runner.x} cy={runner.y} r="10" /><circle cx={runner.x} cy={runner.y} r="4" /></g>}
    </svg>{overlay}</div>
    <div className="te-zone-key">{Object.entries(ZONES).map(([key, zone]) => <span key={key}><i style={{ background: zone.color }} />{zone.label}</span>)}</div>
    <p>{mapMode === 'GENERIC'
      ? `${genericLabel} is the best clean lap of the driver classified P1 in this 2026 qualifying session. It is the recorded pace reference.`
      : mapMode === 'OPTIMAL'
        ? `${genericLabel} is a PitWolf 2026 qualifying calibration and regulation-bounded energy optimisation. It is modelled from the recorded P1 lap, not measured team deployment telemetry.`
        : 'Blue segments show inferred ERS-K battery deployment; the darker blue is near 350 kW. In qualifying, blue can appear only within the user-supplied Straight Mode ranges. Green segments show ICE propulsion with no inferred ERS draw. The inference combines recorded public throttle, brake, speed and RPM with the FIA-constrained energy model; it is not team deployment telemetry.'}</p>
  </section>
}

function WorkspaceControls({ workspace, qualifyingOnly = false }) {
  const { selection, update, events, session, loadingSession, teams, teamDrivers, only2026 } = workspace
  const event = events.find((item) => Number(item.round) === Number(selection.round))
  const sessions = qualifyingOnly
    ? ['Qualifying']
    : (event?.sessions?.filter((name) => name === 'Qualifying' || name === 'Race') ?? ['Qualifying', 'Race'])
  return <section className="te-controls">
    <label><span>SEASON</span><select value={selection.year} onChange={(event) => update({ year: Number(event.target.value) })} disabled={only2026}>{(only2026 ? [2026] : YEARS).map((year) => <option key={year} value={year}>{year}</option>)}</select></label>
    <label><span>GRAND PRIX</span><select value={selection.round ?? ''} onChange={(event) => update({ round: Number(event.target.value) })} disabled={!events.length}>{events.map((event) => <option key={event.round} value={event.round}>{event.name}</option>)}</select></label>
    <label><span>SESSION</span><select value={selection.session} onChange={(event) => update({ session: event.target.value })} disabled={qualifyingOnly}>{sessions.map((session) => <option key={session} value={session}>{session}</option>)}</select></label>
    <label><span>TEAM</span><select value={selection.team} onChange={(event) => update({ team: event.target.value })} disabled={!teams.length}>{teams.map((team) => <option key={team} value={team}>{team}</option>)}</select></label>
    <label><span>DRIVER / INSPECT</span><select value={selection.driver} onChange={(event) => update({ driver: event.target.value })} disabled={!teamDrivers.length}>{teamDrivers.map((driver) => <option key={driver.abbr} value={driver.abbr}>{driver.abbr} · {driver.name}</option>)}</select></label>
    <div className="te-data-state"><span>SESSION DATA</span><b>{loadingSession ? 'LOADING' : session ? `${session.laps?.length ?? 0} LAPS` : 'WAITING'}</b></div>
  </section>
}

function bestLap(laps, driver) {
  const clean = (laps ?? []).filter((lap) => lap.driver === driver && lap.lapTimeS != null && !lap.isOutlier)
  const candidates = clean.length ? clean : (laps ?? []).filter((lap) => lap.driver === driver && lap.lapTimeS != null)
  return candidates.reduce((best, lap) => !best || lap.lapTimeS < best.lapTimeS ? lap : best, null)
}

function bestQualifyingSoftLap(laps, driver) {
  const soft = (laps ?? []).filter((lap) => lap.driver === driver && String(lap.compound ?? '').toUpperCase() === 'SOFT')
  return bestLap(soft, driver)
}

export function useTeamEnergyWorkspace(enabled) {
  const [selection, setSelection] = useState({ year: 2026, round: null, session: 'Qualifying', team: '', driver: '' })
  const [events, setEvents] = useState([])
  const [session, setSession] = useState(null)
  const [trackmap, setTrackmap] = useState(null)
  const [trace, setTrace] = useState(null)
  const [teamBestTrace, setTeamBestTrace] = useState(null)
  const [genericTrace, setGenericTrace] = useState(null)
  const [paceModel, setPaceModel] = useState(null)
  const [loadingSession, setLoadingSession] = useState(false)
  const [error, setError] = useState(null)

  // These two energy workspaces are deliberately scoped to the 2026 formula
  // and its 350 kW deployment rules. Do not accidentally load a historical
  // season and present its different regulations as a 2026 comparison.
  const update = (patch) => setSelection((current) => ({ ...current, ...patch, year: 2026 }))

  useEffect(() => {
    if (!enabled) return undefined
    let live = true
    fetchJson(`/api/f1/events?year=${selection.year}`)
      .then((data) => {
        if (!live) return
        // Default to a completed event whenever the schedule carries both
        // historical and future rounds. The energy pages must not ask FastF1
        // for a future session and then appear to have an empty team trace.
        const scheduled = (data.events ?? []).filter((event) => event.sessions?.includes('Qualifying') || event.sessions?.includes('Race'))
        const available = scheduled.filter((event) => event.raceDataAvailable !== false)
        const displayEvents = available.length ? available : scheduled
        setEvents(displayEvents)
        const current = displayEvents.find((event) => Number(event.round) === Number(selection.round))
        const next = current ?? displayEvents.at(-1)
        if (!next) return
        const nextSession = next.sessions?.includes(selection.session)
          ? selection.session
          : (next.sessions?.includes('Qualifying') ? 'Qualifying' : 'Race')
        setSelection((previous) => ({ ...previous, round: next.round, session: nextSession }))
      })
      .catch((fetchError) => { if (live) setError(fetchError.message) })
    return () => { live = false }
  }, [enabled, selection.year])

  useEffect(() => {
    if (!enabled) return undefined
    let live = true
    fetchJson('/api/f1/qualifying-pace/status?year=2026')
      .then((status) => { if (live) setPaceModel(status) })
      .catch(() => { if (live) setPaceModel({ error: true }) })
    return () => { live = false }
  }, [enabled])

  useEffect(() => {
    if (!enabled || !selection.round || !selection.session) return undefined
    let live = true
    setLoadingSession(true); setError(null); setSession(null); setTrackmap(null); setTrace(null); setTeamBestTrace(null); setGenericTrace(null)
    const params = `year=${selection.year}&round=${selection.round}&session=${encodeURIComponent(selection.session)}`
    // Timing data is small and is enough to populate all controls and lap
    // cards. Do not delay those on the heavier circuit geometry request.
    fetchJson(`/api/f1/session?${params}`)
      .then((sessionData) => {
        if (!live) return
        setSession(sessionData)
        const ranked = [...(sessionData.drivers ?? [])].sort((a, b) => (a.position ?? 99) - (b.position ?? 99))
        const initialTeam = ranked[0]?.team ?? ''
        const team = (sessionData.drivers ?? []).some((driver) => driver.team === selection.team) ? selection.team : initialTeam
        const candidates = (sessionData.drivers ?? []).filter((driver) => driver.team === team).sort((a, b) => (a.position ?? 99) - (b.position ?? 99))
        const driver = candidates.some((entry) => entry.abbr === selection.driver) ? selection.driver : candidates[0]?.abbr ?? ''
        setSelection((previous) => ({ ...previous, team, driver }))
      })
      .catch((fetchError) => { if (live) setError(fetchError.message) })
      .finally(() => { if (live) setLoadingSession(false) })

    fetchJson(`/api/f1/trackmap?${params}`)
      .then((mapData) => { if (live) setTrackmap(mapData.points?.length ? mapData : null) })
      .catch(() => { if (live) setTrackmap(null) })
    return () => { live = false }
  }, [enabled, selection.year, selection.round, selection.session])

  const teams = useMemo(() => [...new Set((session?.drivers ?? []).map((driver) => driver.team).filter(Boolean))].sort(), [session])
  const teamDrivers = useMemo(() => (session?.drivers ?? []).filter((driver) => driver.team === selection.team).sort((a, b) => (a.position ?? 99) - (b.position ?? 99)), [session, selection.team])
  useEffect(() => {
    if (!teamDrivers.length || teamDrivers.some((driver) => driver.abbr === selection.driver)) return
    setSelection((previous) => ({ ...previous, driver: teamDrivers[0].abbr }))
  }, [teamDrivers, selection.driver])
  const initialTeamDriver = teamDrivers[0] ?? null
  const inspectedDriver = teamDrivers.find((driver) => driver.abbr === selection.driver) ?? initialTeamDriver
  const inspectedLap = selection.session === 'Qualifying'
    ? bestQualifyingSoftLap(session?.laps, inspectedDriver?.abbr)
    : bestLap(session?.laps, inspectedDriver?.abbr)
  const teamReferences = teamDrivers.map((driver) => ({
    driver,
    lap: selection.session === 'Qualifying'
      ? bestQualifyingSoftLap(session?.laps, driver.abbr)
      : bestLap(session?.laps, driver.abbr),
  })).filter((entry) => entry.lap)
  const teamReference = teamReferences.reduce((best, entry) => !best || entry.lap.lapTimeS < best.lap.lapTimeS ? entry : best, null)
  const teamBestDriver = teamReference?.driver ?? initialTeamDriver
  const teamBestLap = teamReference?.lap ?? null
  const genericLap = useMemo(() => {
    // Qualifying uses the driver officially classified P1 as its reference.
    // A smallest raw timing sample can disagree with the official session order.
    if (selection.session === 'Qualifying') {
      const poleDriver = [...(session?.drivers ?? [])].sort((a, b) => (a.position ?? 99) - (b.position ?? 99))[0]
      return bestLap(session?.laps, poleDriver?.abbr)
    }
    return (session?.laps ?? []).filter((lap) => lap.lapTimeS != null && !lap.isOutlier).reduce((best, lap) => !best || lap.lapTimeS < best.lapTimeS ? lap : best, null)
  }, [session, selection.session])

  useEffect(() => {
    if (!enabled || !inspectedLap || !inspectedDriver) return undefined
    let live = true
    const params = `year=${selection.year}&round=${selection.round}&session=${encodeURIComponent(selection.session)}&driver=${inspectedDriver.abbr}&lap=${inspectedLap.lapNumber}`
    setTrace({ loading: true })
    Promise.all([fetchJson(`/api/f1/telemetry?${params}`), fetchJson(`/api/f1/energy?${params}`)])
      .then(([telemetry, energy]) => { if (live) setTrace({ telemetry, energy }) })
      .catch((fetchError) => { if (live) setTrace({ error: fetchError.message }) })
    return () => { live = false }
  }, [enabled, selection.year, selection.round, selection.session, inspectedDriver?.abbr, inspectedLap?.lapNumber])

  useEffect(() => {
    if (!enabled || !teamBestLap || !teamBestDriver) return undefined
    // Reuse the inspected trace when it already is the team's qualifying
    // reference. Otherwise fetch the team's best recorded lap independently
    // from the driver-inspection dropdown.
    if (teamBestDriver.abbr === inspectedDriver?.abbr && teamBestLap.lapNumber === inspectedLap?.lapNumber) {
      setTeamBestTrace(null)
      return undefined
    }
    let live = true
    const params = `year=${selection.year}&round=${selection.round}&session=${encodeURIComponent(selection.session)}&driver=${teamBestDriver.abbr}&lap=${teamBestLap.lapNumber}`
    setTeamBestTrace({ loading: true })
    Promise.all([fetchJson(`/api/f1/telemetry?${params}`), fetchJson(`/api/f1/energy?${params}`)])
      .then(([telemetry, energy]) => { if (live) setTeamBestTrace({ telemetry, energy }) })
      .catch((fetchError) => { if (live) setTeamBestTrace({ error: fetchError.message }) })
    return () => { live = false }
  }, [enabled, selection.year, selection.round, selection.session, teamBestDriver?.abbr, teamBestLap?.lapNumber, inspectedDriver?.abbr, inspectedLap?.lapNumber])

  useEffect(() => {
    if (!enabled || !genericLap || genericLap.driver === inspectedDriver?.abbr && genericLap.lapNumber === inspectedLap?.lapNumber) {
      setGenericTrace(null)
      return undefined
    }
    let live = true
    const params = `year=${selection.year}&round=${selection.round}&session=${encodeURIComponent(selection.session)}&driver=${genericLap.driver}&lap=${genericLap.lapNumber}`
    setGenericTrace({ loading: true })
    Promise.all([fetchJson(`/api/f1/telemetry?${params}`), fetchJson(`/api/f1/energy?${params}`)])
      .then(([telemetry, energy]) => { if (live) setGenericTrace({ telemetry, energy }) })
      .catch((fetchError) => { if (live) setGenericTrace({ error: fetchError.message }) })
    return () => { live = false }
  }, [enabled, selection.year, selection.round, selection.session, genericLap?.driver, genericLap?.lapNumber, inspectedDriver?.abbr, inspectedLap?.lapNumber])

  return { selection, update, events, session, trackmap, trace, teamBestTrace, genericTrace, paceModel, loadingSession, error, teams, teamDrivers, teamBestDriver, inspectedDriver, inspectedLap, teamBestLap, genericLap, only2026: true }
}

function TeamReference({ workspace }) {
  const { session, selection, teamBestDriver, teamBestLap, inspectedDriver, inspectedLap, trace, paceModel } = workspace
  const actual = teamBestLap?.lapTimeS
  const inspected = inspectedLap?.lapTimeS
  const energy = trace?.energy?.summary
  const regulation = trace?.energy?.regulation
  return <>
    <div className="te-reference-grid">
      <article><span>{selection.session === 'Qualifying' ? 'TEAM FASTEST REAL SOFT QUALIFYING LAP' : 'TEAM BEST REAL RACE LAP'}</span><b>{actual == null ? '—' : formatLapTime(actual)}</b><em>{teamBestDriver?.abbr ?? '—'} · P{teamBestDriver?.position ?? '—'} · {teamBestLap?.compound ?? 'TYRE N/A'} · L{teamBestLap?.lapNumber ?? '—'}</em></article>
      <article><span>INSPECTED DRIVER LAP</span><b>{inspected == null ? '—' : formatLapTime(inspected)}</b><em>{inspectedDriver?.abbr ?? '—'} · {inspectedLap?.compound ?? 'TYRE N/A'} · {fmtDelta(inspected != null && actual != null ? inspected - actual : null)}</em></article>
      <article className="te-model-card"><span>TEAM-OPTIMAL LAP</span><b>{paceModel?.deploymentEligible ? 'MODEL FIT READY' : paceModel?.coverageComplete ? 'PACE BASELINE EVALUATED' : 'TELEMETRY COVERAGE PENDING'}</b><em>{paceModel?.deploymentEligible ? 'Validated pace fit is ready for the FIA-constrained team-optimal phase.' : paceModel?.coverageComplete ? `13 / 13 completed qualifying sessions have public telemetry. The observed-pace baseline is evaluated (${paceModel?.selectedHeldOutMetrics?.maeS?.toFixed?.(3) ?? '—'}s held-out MAE), but cannot claim a battery or team-optimal output yet.` : `${paceModel?.coverage?.sessionsWithTelemetry ?? 0} / ${paceModel?.coverage?.completedQualifyingSessions ?? 13} completed qualifying sessions have usable public telemetry. No team-optimal time claim is shown yet.`}</em></article>
    </div>
    {trace?.loading && <div className="te-loading">FETCHING RECORDED TELEMETRY + MODELLED ENERGY TRACE…</div>}
    {trace?.error && <p className="te-error">Telemetry trace unavailable: {trace.error}</p>}
    {energy && <div className="te-summary-grid">
      <div><span>MODELLED DEPLOY</span><b>{energy.deployMj.toFixed(2)} MJ</b></div><div><span>MODELLED HARVEST</span><b>{energy.harvestMj.toFixed(2)} MJ</b></div><div><span>{energy.socDisplayMode === 'QUALIFYING_NORMALISED_USABLE_WINDOW' ? 'QUALIFYING USABLE SoC' : 'SoC START → END'}</span><b>{energy.socDisplayMode === 'QUALIFYING_NORMALISED_USABLE_WINDOW' ? `${energy.socStartPercent.toFixed(0)}% → ${energy.socEndPercent.toFixed(0)}%` : `${energy.socStartMj.toFixed(2)} → ${energy.socEndMj.toFixed(2)} MJ`}</b></div><div><span>CLIPPING</span><b>{energy.clipSeconds.toFixed(1)} s</b></div>
    </div>}
    {regulation && <p className="te-disclosure"><b>{regulation.eventSpecificLimitsLoaded ? 'FIA EVENT LIMITS LOADED' : 'FIA GLOBAL LIMITS ONLY'}</b> · {regulation.note}</p>}
    {energy?.qualifyingBatteryNote && <p className="te-disclosure">{energy.qualifyingBatteryNote}</p>}
    {deploymentPolicyDisclosure(energy)}
    {energy?.qualifyingErsInference && <p className="te-disclosure"><b>TELEMETRY-CONSTRAINED QUALIFYING ERS INFERENCE</b> · PitWolf prioritises observed throttle, acceleration and tractive-power demand inside the selected Straight Mode ranges, then applies the 350 kW, speed-curve, recharge and usable-SoC limits. {energy.inferredFull350KwSeconds?.toFixed?.(1) ?? '—'} s are inferred near 350 kW; this is not measured team ERS telemetry.</p>}
    {energy?.socDisplayMode === 'QUALIFYING_NORMALISED_USABLE_WINDOW' && !energy?.qualifyingBatteryCalibrated && <p className="te-disclosure">NOT CALIBRATED TO 0% · This recorded lap cannot consume the full 4 MJ usable window without exceeding its observed wheel-demand and FIA power limits. PitWolf keeps the residual rather than inventing deployment.</p>}
    {session && <p className="te-disclosure">Real lap time, tyre, driver rank and public telemetry come from FastF1. All energy values are produced by PitWolf’s regulation-bounded model; they are not team battery or deployment telemetry.</p>}
  </>
}

export function TeamLapPage({ workspace }) {
  const telemetry = workspace.trace?.telemetry?.trace
  const teamBestIsInspected = workspace.teamBestDriver?.abbr === workspace.inspectedDriver?.abbr
    && workspace.teamBestLap?.lapNumber === workspace.inspectedLap?.lapNumber
  const teamReferenceTrace = teamBestIsInspected ? workspace.trace : workspace.teamBestTrace
  return <section className="te-root">
    <WorkspaceControls workspace={workspace} />
    {workspace.error && <p className="te-error">Session data unavailable: {workspace.error}</p>}
    <TeamReference workspace={workspace} />
    {teamReferenceTrace?.loading && <div className="te-loading">LOADING RECORDED TEAM-BEST LAP PLAYBACK…</div>}
    {teamReferenceTrace?.error && <p className="te-error">Team-best lap playback unavailable: {teamReferenceTrace.error}</p>}
    {workspace.trackmap && teamReferenceTrace?.energy && teamReferenceTrace?.telemetry && <TeamBestLapSimulation workspace={workspace} referenceTrace={teamReferenceTrace} />}
    {telemetry && <section className="te-observed-grid">
      <article className="te-panel"><div className="te-panel-head"><span>OBSERVED TEAM LAP / SPEED</span><em>PUBLIC TELEMETRY</em></div><MiniTrace title="SPEED" values={telemetry.speed} color="#43d6af" max={360} unit=" km/h" /></article>
      <article className="te-panel"><div className="te-panel-head"><span>OBSERVED TEAM LAP / THROTTLE</span><em>PUBLIC TELEMETRY</em></div><MiniTrace title="THROTTLE" values={telemetry.throttle} color="#ff9b78" max={100} unit="%" /></article>
      <article className="te-panel te-driver-card"><div className="te-panel-head"><span>TEAM REFERENCE</span><em>QUALIFYING RANK</em></div><b>{workspace.teamBestDriver?.team ?? '—'}</b><strong>{workspace.teamBestDriver?.abbr ?? '—'}</strong><p>The team reference is the best-ranked qualifying driver. The driver selector remains available for inspecting the other driver’s actual trace.</p></article>
    </section>}
  </section>
}

function pointIndexAtTime(times, time) {
  if (!times?.length) return 0
  let lo = 0
  let hi = times.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (times[mid] < time) lo = mid + 1
    else hi = mid
  }
  return lo
}

// FastF1 samples are not equally spaced. Interpolating their values on the
// elapsed-time axis keeps the runner continuous through sparse GPS segments.
function traceValueAtTime(times, values, time) {
  if (!times?.length || !values?.length) return 0
  const upper = pointIndexAtTime(times, time)
  if (upper <= 0) return values[0] ?? 0
  const lower = upper - 1
  const span = Math.max(1e-6, (times[upper] ?? 0) - (times[lower] ?? 0))
  const fraction = Math.max(0, Math.min(1, (time - (times[lower] ?? 0)) / span))
  return (values[lower] ?? 0) + ((values[upper] ?? values[lower] ?? 0) - (values[lower] ?? 0)) * fraction
}

function simulationTime(seconds) {
  if (!Number.isFinite(seconds)) return '0:00.000'
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds - minutes * 60
  return `${minutes}:${remainder.toFixed(3).padStart(6, '0')}`
}

function speedPathByTime(times, speeds, width, height) {
  if (!times?.length || !speeds?.length) return ''
  const total = Math.max(times.at(-1) ?? 1, 1)
  return speeds.map((speed, index) => {
    const x = 34 + ((times[index] ?? 0) / total) * (width - 52)
    const y = 14 + (1 - Math.min(1, Math.max(0, speed / 380))) * (height - 34)
    return `${index ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ')
}

function LapPlayback({ workspace, referenceTrace, referenceId, label, mapMode, timeLabel, traceLabel, method }) {
  const trace = referenceTrace?.telemetry?.trace
  const energy = referenceTrace?.energy
  // A qualifying reference is exactly one flying lap. Playback is deliberately
  // opt-in and never loops into an imaginary second lap.
  const [running, setRunning] = useState(false)
  // At 1×, the simulator advances one recorded telemetry second per wall-clock
  // second. Faster selections are simple real-time multiples of that lap.
  const [rate, setRate] = useState(1)
  const [progress, setProgress] = useState(0)
  const [runVersion, setRunVersion] = useState(0)
  const progressRef = useRef(0)
  const animationTokenRef = useRef(0)

  const totalTime = trace?.time?.at(-1) ?? 0
  useEffect(() => {
    setRunning(false)
    setProgress(0)
    progressRef.current = 0
    setRunVersion((version) => version + 1)
  }, [referenceId, totalTime])

  useEffect(() => {
    if (!running || !totalTime) return undefined
    const token = animationTokenRef.current + 1
    animationTokenRef.current = token
    const baseProgress = progressRef.current
    let startedAt = null
    let frameId = null
    const tick = (now) => {
      if (token !== animationTokenRef.current) return
      if (startedAt == null) startedAt = now
      const elapsedWallSeconds = (now - startedAt) / 1000
      const nextProgress = Math.min(1, baseProgress + (elapsedWallSeconds * rate) / totalTime)
      progressRef.current = nextProgress
      setProgress(nextProgress)
      if (nextProgress >= 1) {
        // Finish at the timing line and stop. Do not wrap to lap two.
        setRunning(false)
        return
      }
      frameId = requestAnimationFrame(tick)
    }
    frameId = requestAnimationFrame(tick)
    return () => {
      animationTokenRef.current += 1
      cancelAnimationFrame(frameId)
    }
  }, [running, rate, totalTime, runVersion])

  if (!trace || !energy || !workspace.trackmap) return <div className="te-loading">BUILDING CIRCUIT BENCHMARK SIMULATION…</div>
  const elapsed = progress * totalTime
  const speed = traceValueAtTime(trace.time, trace.speed, elapsed)
  const distance = traceValueAtTime(trace.time, trace.distance, elapsed)
  const soc = traceValueAtTime(energy.trace?.time, energy.trace?.socMj, elapsed)
  const socPct = traceValueAtTime(energy.trace?.time, energy.trace?.socPct, elapsed)
  const inferredErsKw = traceValueAtTime(energy.trace?.time, energy.trace?.pElecKw, elapsed)
  const inferenceScore = traceValueAtTime(energy.trace?.time, energy.trace?.ersInferenceScore, elapsed)
  const normalisedQualifyingSoc = energy.summary?.socDisplayMode === 'QUALIFYING_NORMALISED_USABLE_WINDOW'
  const qualifyingErsInference = energy.summary?.qualifyingErsInference?.status === 'PUBLIC_TELEMETRY_CONSTRAINED_INFERENCE'
  const usableSocWindowMj = Number(energy.summary?.socWindowMj) || 4
  const batteryPercent = Math.min(100, Math.max(0, normalisedQualifyingSoc
    ? socPct
    : (soc / usableSocWindowMj) * 100))
  const batteryMj = Math.max(0, soc)
  const vmax = Math.max(...(trace.speed ?? [0]))
  const zone = classifyZone(energy.trace, trace, distance)
  const width = 760
  const height = 152
  const cursorX = 34 + progress * (width - 52)
  const cursorY = 14 + (1 - Math.min(1, Math.max(0, speed / 380))) * (height - 34)
  const mapHud = <div className="glh-root">
    <div className="glh-left">
      <div><span>{timeLabel}</span><b>{simulationTime(elapsed)}</b><em>{running ? `RUNNING · ${rate}× REAL TIME` : progress >= 1 ? 'LAP COMPLETE · 1 LAP' : 'READY · PRESS PLAY'}</em></div>
      <div><span>TRACK SPEED</span><b>{Math.round(speed)} km/h</b><em>Vmax {Math.round(vmax)} km/h</em></div>
    </div>
    <div className="glh-right">
      <div className="glh-energy-state">
        <span>ENERGY STATE</span>
        <b style={{ color: ZONES[zone]?.color }}>{ZONES[zone]?.label ?? 'ON POWER'}</b>
        <em>{normalisedQualifyingSoc ? `${Math.round(socPct)}% MODELLED USABLE SoC` : `${Math.round(soc * 100) / 100} MJ MODELLED SoC`}</em>
        {qualifyingErsInference && <em>INFERRED ERS-K {Math.round(inferredErsKw)} kW · {Math.round(inferenceScore * 100)}% TELEMETRY EVIDENCE</em>}
      </div>
      <div className="glh-battery" aria-label="Live modelled battery simulation">
        <span>BATTERY SIMULATION</span>
        <div className="glh-battery-reading"><b>{Math.round(batteryPercent)}%</b><em>{batteryMj.toFixed(2)} / {usableSocWindowMj.toFixed(2)} MJ</em></div>
        <div className="glh-battery-track" role="progressbar" aria-label="Modelled usable state of charge" aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(batteryPercent)}><i style={{ width: `${batteryPercent}%` }} /></div>
        <em>{normalisedQualifyingSoc ? 'LIVE MODELLED USABLE SoC · 100% → 0%' : 'LIVE MODELLED SoC'}</em>
      </div>
    </div>
    <div className="glh-controls"><button type="button" onClick={() => { if (running) { setRunning(false); return } if (progressRef.current >= 1) { progressRef.current = 0; setProgress(0) } setRunning(true) }}>{running ? 'Ⅱ PAUSE' : '▷ PLAY'}</button><button type="button" onClick={() => { progressRef.current = 0; setProgress(0); setRunning(false); setRunVersion((version) => version + 1) }}>↻ RESTART</button><div>{[1, 2, 4, 8].map((candidate) => <button key={candidate} type="button" className={rate === candidate ? 'active' : ''} onClick={() => setRate(candidate)}>{candidate}×</button>)}</div></div>
  </div>

  return <section className="gls-root">
    <EnergyMap trackmap={workspace.trackmap} energy={energy} telemetry={{ trace }} mapMode={mapMode} genericLabel={label} runnerDistance={distance} hideSwitch overlay={mapHud} />
    <p className="gls-method">{method}</p>
    <section className="gls-speed"><div><span>{traceLabel}</span><b>{Math.round(speed)} km/h</b></div><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Changing speed graph"><line x1="34" x2={width - 18} y1="38" y2="38" /><line x1="34" x2={width - 18} y1="84" y2="84" /><line x1="34" x2={width - 18} y1="130" y2="130" /><path d={speedPathByTime(trace.time, trace.speed, width, height)} /><line className="gls-cursor" x1={cursorX} x2={cursorX} y1="14" y2={height - 20} /><circle cx={cursorX} cy={cursorY} r="4" /></svg><small>{timeLabel} →</small></section>
  </section>
}

function GenericLapSimulation({ workspace, referenceTrace }) {
  const label = `${workspace.genericLap?.driver ?? 'REFERENCE'} L${workspace.genericLap?.lapNumber ?? '—'}`
  return <LapPlayback
    workspace={workspace}
    referenceTrace={referenceTrace}
    referenceId={`generic-${label}`}
    label={label}
    mapMode="GENERIC"
    timeLabel="SIMULATED LAP TIME"
    traceLabel="SIMULATED SPEED TRACE"
    method="CIRCUIT-BENCHMARK SIMULATION · It advances through the best clean recorded qualifying speed profile of the driver classified P1 while the regulation-bounded energy controller integrates a single-car qualifying lap. The moving car, SoC and coloured states are simulated; the seed speed trace is public timing/telemetry. This is not yet a trained claim of the absolute best possible car."
  />
}

function TeamBestLapSimulation({ workspace, referenceTrace }) {
  const label = `${workspace.teamBestDriver?.abbr ?? 'TEAM'} L${workspace.teamBestLap?.lapNumber ?? '—'}`
  return <LapPlayback
    workspace={workspace}
    referenceTrace={referenceTrace}
    referenceId={`team-${workspace.selection.team}-${label}-${workspace.selection.session}`}
    label={label}
    mapMode="TEAM"
    timeLabel="RECORDED TEAM LAP TIME"
    traceLabel="RECORDED SPEED TRACE"
    method={`RECORDED TEAM-LAP PLAYBACK · ${label} is the selected team's ${workspace.selection.session === 'Qualifying' ? 'fastest clean Soft qualifying' : 'best clean race'} lap. The moving car and speed trace follow public FastF1 telemetry at the lap's actual recorded time; the coloured energy states and SoC are PitWolf's modelled interpretation.`}
  />
}

function ModelOptimalLapSimulation({ workspace, optimum }) {
  const label = `PITWOLF OPTIMUM · ${optimum.driver ?? workspace.genericLap?.driver ?? 'REFERENCE'} L${optimum.lapNumber ?? workspace.genericLap?.lapNumber ?? '—'}`
  const training = optimum.training ?? {}
  return <LapPlayback
    workspace={workspace}
    referenceTrace={optimum}
    referenceId={`optimum-${workspace.selection.round}-${optimum.driver}-${optimum.lapNumber}`}
    label={label}
    mapMode="OPTIMAL"
    timeLabel="MODELLED LAP TIME"
    traceLabel="MODELLED OPTIMAL SPEED"
    method={`PITWOLF QUALIFYING OPTIMISER · Pace-calibrated on ${training.sessions ?? 0} completed 2026 qualifying sessions / ${training.cleanLaps ?? 0} clean laps, with ${training.telemetryDriverLaps ?? 0} fastest clean-driver telemetry traces (${training.telemetrySamples?.toLocaleString?.() ?? 0} observed samples). It redistributes a regulation-bounded electrical budget from lower-value acceleration to higher-value full-throttle zones. The predicted ${optimum.energy?.summary?.expectedGainS?.toFixed(3) ?? '—'}s improvement is MODELLED, not a recorded lap or measured team deployment.`}
  />
}

export function OptimalLapPage({ workspace }) {
  const [sourceMode, setSourceMode] = useState('RECORDED')
  const [optimum, setOptimum] = useState(null)
  useEffect(() => {
    if (workspace.selection.session !== 'Qualifying') workspace.update({ session: 'Qualifying' })
  }, [workspace.selection.session])
  useEffect(() => {
    if (sourceMode !== 'MODEL' || !workspace.genericLap) return undefined
    let live = true
    const { year, round } = workspace.selection
    const { driver, lapNumber } = workspace.genericLap
    setOptimum({ loading: true })
    fetchJson(`/api/f1/optimal-lap?year=${year}&round=${round}&driver=${driver}&lap=${lapNumber}`)
      .then((payload) => { if (live) setOptimum(payload) })
      .catch((fetchError) => { if (live) setOptimum({ error: fetchError.message }) })
    return () => { live = false }
  }, [sourceMode, workspace.selection.year, workspace.selection.round, workspace.genericLap?.driver, workspace.genericLap?.lapNumber])
  const genericLabel = workspace.genericLap ? `${workspace.genericLap.driver} L${workspace.genericLap.lapNumber}` : 'No generic reference lap'
  const recordedTrace = workspace.genericTrace?.energy ? workspace.genericTrace : workspace.trace
  const modelSummary = optimum?.energy?.summary
  const activeTrace = sourceMode === 'MODEL' ? optimum : recordedTrace
  const activeSummary = activeTrace?.energy?.summary
  const activeRegulation = activeTrace?.regulation || activeTrace?.energy?.regulation
  return <section className="te-root">
    <WorkspaceControls workspace={workspace} qualifyingOnly />
    <section className="te-optimal-header">
      <div><span>CIRCUIT-LEVEL ENERGY OPTIMISER</span><h2>Optimal lap<br /><em>energy map.</em></h2><p>Compare the recorded P1 qualifying lap with PitWolf’s regulation-bounded energy simulation. The animated optimiser is a modelled preview until the qualifying pace dataset passes its telemetry-coverage and held-out validation gates; it is never recorded timing or measured team deployment data.</p></div>
      <div className="te-generic-card">
        <span>{sourceMode === 'MODEL' ? 'PITWOLF MODELLED QUALIFYING OPTIMUM' : 'RECORDED 2026 POLE REFERENCE'}</span>
        <b>{sourceMode === 'MODEL' ? (modelSummary?.modelledLapTimeS == null ? '—' : formatLapTime(modelSummary.modelledLapTimeS)) : (workspace.genericLap?.lapTimeS == null ? '—' : formatLapTime(workspace.genericLap.lapTimeS))}</b>
        <em>{sourceMode === 'MODEL' ? `${genericLabel} · ${modelSummary?.trainingSessions ?? 0} Q sessions · ${modelSummary?.trainingCleanLaps ?? 0} clean laps` : `${genericLabel} · ${workspace.genericLap?.compound ?? 'TYRE N/A'} · best clean lap of the driver classified P1`}</em>
          <strong>{sourceMode === 'MODEL' ? (modelSummary ? `${workspace.paceModel?.deploymentEligible ? 'MODELLED' : 'PREVIEW ONLY'} −${modelSummary.expectedGainS.toFixed(3)}s VS RECORDED P1 · LIMITED CONFIDENCE` : 'FITTING 2026 QUALIFYING CALIBRATION…') : 'RECORDED INPUT'}</strong>
        <label className="te-optimum-select"><span>SIMULATION SOURCE</span><select value={sourceMode} onChange={(event) => setSourceMode(event.target.value)}><option value="RECORDED">Recorded P1 reference</option><option value="MODEL" disabled={!workspace.genericLap}>PitWolf optimal simulation</option></select></label>
      </div>
    </section>
    {activeTrace?.loading && <div className="te-loading">{sourceMode === 'MODEL' ? 'FITTING 2026 QUALIFYING CALIBRATION + OPTIMISING ENERGY PLAN…' : 'BUILDING THE RECORDED ENERGY-STATE MAP…'}</div>}
    {activeTrace?.error && <p className="te-error">{sourceMode === 'MODEL' ? 'Optimal simulation unavailable' : 'Energy map unavailable'}: {activeTrace.error}</p>}
    {workspace.trackmap && sourceMode === 'MODEL' && optimum?.energy && optimum?.telemetry && <ModelOptimalLapSimulation workspace={workspace} optimum={optimum} />}
    {workspace.trackmap && sourceMode === 'RECORDED' && recordedTrace?.energy && recordedTrace?.telemetry && <GenericLapSimulation workspace={workspace} referenceTrace={recordedTrace} />}
    {activeRegulation && <p className="te-disclosure"><b>{activeRegulation.eventSpecificLimitsLoaded ? 'FIA EVENT LIMITS LOADED' : 'FIA GLOBAL LIMITS ONLY'}</b> · {activeRegulation.note}</p>}
    {deploymentPolicyDisclosure(activeSummary)}
    {activeSummary && <div className="te-summary-grid">
      <div><span>DEPLOYMENT</span><b>{activeSummary.deployMj.toFixed(2)} MJ</b></div><div><span>BRAKING HARVEST</span><b>{activeSummary.harvestMj.toFixed(2)} MJ</b></div><div><span>FULL-THROTTLE DEPLOY</span><b>{activeSummary.deployFullThrottleMj.toFixed(2)} MJ</b></div><div><span>{sourceMode === 'MODEL' ? 'EXPECTED TIME GAIN' : 'LAP ENERGY BALANCE'}</span><b>{sourceMode === 'MODEL' ? `−${activeSummary.expectedGainS.toFixed(3)} s` : `${activeSummary.netSocDeltaMj >= 0 ? '+' : ''}${activeSummary.netSocDeltaMj.toFixed(2)} MJ`}</b></div>
    </div>}
  </section>
}
