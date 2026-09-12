import React, { useEffect, useMemo, useState } from 'react'
import {
  fetchEnergyRace,
  fetchDecisionPoints,
  fetchModelReport,
  fetchEvents,
  predictOvertake,
  fetchStrategyReplay,
  fetchCachedRaces,
  fetchBatteryClip,
  fetchStrategyStory,
  fetchEnergyTrend,
  STRATEGY_COLORS,
  STRATEGY_ORDER,
} from '../lib/f1api'

// ─── Shared helpers ──────────────────────────────────────────────────────────

const pct = (v, d = 0) => (v == null || Number.isNaN(v) ? '—' : `${Number(v).toFixed(d)}%`)
const mj = (v, d = 2) => (v == null || Number.isNaN(v) ? '—' : `${Number(v).toFixed(d)} MJ`)
const num = (v, d = 1) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(d))
const modelName = (name) => ({
  RandomForestClassifier: 'Random Forest',
  RandomForestSigmoidCalibrated: 'Random Forest + Sigmoid',
  LogisticRegression: 'Logistic Regression',
  GradientBoostingClassifier: 'Gradient Boosted Trees',
}[name] ?? name)

function Badge({ tone = 'real', children }) {
  return <span className={`data-badge ${tone}`}><i />{children}</span>
}

function racePosition(position) {
  return position == null ? '—' : `P${position}`
}

function finishPos(value) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return `P${Math.round(Number(value))}`
}

function takeLine(items) {
  const rows = (items || []).filter(Boolean)
  if (!rows.length) return 'no extra whole place to the flag'
  return rows.map((item) => {
    const hold = item.pOppHold == null ? '' : ` · ${item.ahead || 'ahead'} holds ${Math.round(Number(item.pOppHold) * 100)}%`
    const pass = item.pPass == null ? '' : ` · net ${Math.round(Number(item.pPass) * 100)}%`
    return `${item.note || `L${item.lap} ${item.kind === 'HOLD' ? 'hold' : 'take'} ${item.ahead || ''}`}${pass}${hold}`
  }).join(' · ')
}

function oddsPct(value) {
  return value == null || Number.isNaN(Number(value)) ? '—' : `${Math.round(Number(value) * 100)}%`
}

function interestingCall(model) {
  if (!model) return 'ATTACK'
  if (model.whatIfCall) return mapSimCall(model.whatIfCall)
  if (model.theyDid === model.call && model.theyDid === 'ATTACK') return 'SAVE'
  return mapSimCall(model.clips && model.call === 'ATTACK' ? 'HOLD' : model.call)
}

function StrategyRaceStory({ sel, onSelectDriver }) {
  const [story, setStory] = useState({ loading: true })
  useEffect(() => {
    if (!sel?.year || !sel?.round || !sel?.driver) return undefined
    let live = true
    setStory({ loading: true })
    fetchStrategyStory({ year: sel.year, round: sel.round, session: sel.session, driver: sel.driver })
      .then((data) => { if (live) setStory(data?.error ? { error: data.error } : { data }) })
      .catch((error) => { if (live) setStory({ error: error.message }) })
    return () => { live = false }
  }, [sel.year, sel.round, sel.session, sel.driver])

  if (story.loading) return <section className="ov-panel dt-story-panel"><div className="lx-loading"><span className="lx-spinner" />BUILDING DRIVER RACE STORY…</div></section>
  if (story.error) return <section className="ov-panel dt-story-panel"><p className="lx-empty">Race story unavailable. {story.error}</p></section>
  const data = story.data
  const selected = data?.selected
  if (!selected) return null
  const net = selected.netPlaces
  const recs = data.recommendations ?? []
  return <section className="ov-panel dt-story-panel">
    <div className="ov-panel-head">
      <span>DRIVER RACE STORY / {selected.driver} · {String(data.event?.name || '').toUpperCase()}</span>
      <Badge tone="derived">TIMING + 2018–2025 DRS RECIPE</Badge>
    </div>
    <p className="dt-race-position-note">Start is first timed-lap running order. Finish is the classified result. Extra places are modelled from missed 1.0s windows times this driver’s historical DRS efficiency — not a claim they would have finished there.</p>
    <div className="dt-race-position-focus">
      <div><span>{selected.driver} STARTED</span><b>{racePosition(selected.startPosition)}</b></div>
      <div><span>FINISHED</span><b>{racePosition(selected.finishPosition)}</b></div>
      <div><span>NET PLACES</span><b className={net > 0 ? 'positive' : net < 0 ? 'negative' : ''}>{net == null ? '—' : `${net > 0 ? '+' : ''}${net}`}</b></div>
      <div><span>OBSERVED TAKES</span><b>{selected.overtakes ?? 0}</b></div>
    </div>
    <div className="dt-story-focus">
      <div><span>DRS WINDOWS</span><b>{selected.drsConverted}/{selected.drsWindows}</b><em>converted / seen</em></div>
      <div><span>EFFICIENCY</span><b>{selected.efficiency == null ? '—' : Number(selected.efficiency).toFixed(2)}</b><em>0–1 from 2018–2025</em></div>
      <div><span>MISSED WINDOWS</span><b>{selected.drsMissed}</b><em>expected extra {selected.expectedExtraPlaces ?? 0}</em></div>
      <div><span>COULD FINISH</span><b className={selected.couldFinishBetter ? 'positive' : ''}>{selected.couldFinishBetter ? racePosition(selected.potentialFinish) : 'NO LIFT'}</b><em>{selected.couldFinishBetter ? `from P${selected.finishPosition} if missed DRS converted at their rate` : 'historical rate does not move the classified result'}</em></div>
    </div>
    {selected.changes?.length ? <div className="dt-race-position-events">
      <span>{selected.driver} PLACE CHANGES</span>
      <div>{selected.changes.map((event, index) => <b key={`${event.lap}-${index}`}>L{event.lap} {racePosition(event.fromPosition)}→{racePosition(event.toPosition)} ({event.places > 0 ? '+' : ''}{event.places}{event.ahead ? ` vs ${event.ahead}` : ''}{event.pit ? ' · PIT' : ''})</b>)}</div>
    </div> : <p className="ov-notes">No timing place change recorded for {selected.driver}.</p>}
    {selected.missedWindows?.length > 0 && <div className="dt-story-windows">
      <span>MISSED DRS WINDOWS · HOW A TAKE COULD HAVE HAPPENED</span>
      {selected.missedWindows.map((item) => <em key={`${item.startLap}-${item.ahead}`}>L{item.startLap} on {item.ahead} · held {item.waitLaps} laps · typical convert wait {selected.typicalWaitLaps ?? '—'} laps at efficiency {selected.efficiency}</em>)}
    </div>}
    {recs.map((item, index) => <div className="dt-story-rec" key={`${item.driver}-${item.lap}-${index}`}>
      <b>L{item.lap ?? '—'} RECIPE · {item.action}</b>
      {(item.lines ?? []).slice(0, 4).map((line) => <strong key={line}>{line}</strong>)}
    </div>)}
    <div className="dt-race-position-head"><span>DRIVER</span><span>START</span><span>FINAL</span><span>NET</span><span>DRS / RECIPE</span></div>
    <div className="dt-race-position-list">
      {(data.drivers ?? []).map((item) => <button
        type="button"
        className={`dt-race-position-row dt-story-row ${item.driver === selected.driver ? 'selected' : ''}`}
        key={item.driver}
        onClick={() => onSelectDriver?.(item.driver)}
      >
        <b>{item.driver}</b>
        <span>{racePosition(item.startPosition)}</span>
        <span>{racePosition(item.finishPosition)}</span>
        <span className={item.netPlaces > 0 ? 'positive' : item.netPlaces < 0 ? 'negative' : ''}>{item.netPlaces == null ? '—' : `${item.netPlaces > 0 ? '+' : ''}${item.netPlaces}`}</span>
        <em>{item.drsConverted}/{item.drsWindows} DRS · eff {item.efficiency ?? '—'}{item.couldFinishBetter ? ` · could P${item.potentialFinish}` : ''}</em>
      </button>)}
    </div>
  </section>
}

// Three-segment ATTACK/DELAY/SAVE probability bar.
function ProbBar({ probabilities, height = 7 }) {
  const total = STRATEGY_ORDER.reduce((s, k) => s + (probabilities?.[k] ?? 0), 0) || 1
  return <div className="dt-probbar" style={{ height }}>
    {STRATEGY_ORDER.map((k) => {
      const w = ((probabilities?.[k] ?? 0) / total) * 100
      return w > 0.4
        ? <i key={k} style={{ width: `${w}%`, background: STRATEGY_COLORS[k] }} title={`${k} ${w.toFixed(0)}%`} />
        : null
    })}
  </div>
}

function GatePill({ pass }) {
  return <b className={`dt-gate ${pass ? 'pass' : 'fail'}`}>{pass ? '✓ PASS' : '✗ FAIL'}</b>
}

// ─── Backend strategy replay ────────────────────────────────────────────────
//
// The backend is deliberately the only authority for the tactical tree.  A
// former frontend fallback used its own action profile and could add energy at
// a pit stop; that could conflict with the source-controlled model and FIA
// constraints.  If the replay service is unavailable, show that state rather
// than inventing an alternative calculation in the browser.

function StrategyTreePanel({ tree }) {
  if (!tree) return null
  const persistenceDelta = tree.expectedLeadLaps - tree.actualLeadLaps
  const comparisonHorizon = tree.comparisonHorizonLaps ?? tree.horizon ?? 6
  const comparisonExplanation = tree.success
    ? `BETTER means the tree estimates ${persistenceDelta.toFixed(1)} more lap${persistenceDelta === 1 ? '' : 's'} ahead than the real race within the ${comparisonHorizon}-lap comparison horizon.`
    : `NOT YET means the tree estimates ${Math.abs(persistenceDelta).toFixed(1)} fewer or equal lap${Math.abs(persistenceDelta) === 1 ? '' : 's'} ahead than the real race within the ${comparisonHorizon}-lap comparison horizon.`
  const [showBranches, setShowBranches] = useState(false)
  const treeAction = tree.path?.[0]?.action ?? tree.tree?.bestAction ?? '—'
  const pathNodes = []
  let cursor = tree.tree
  while (cursor?.children?.length) {
    pathNodes.push(cursor)
    const best = cursor.children.find((child) => child.best) ?? cursor.children[0]
    cursor = best.next
  }
  return <section className="ov-panel dt-tree-panel">
    <div className="ov-panel-head"><span>RECURSIVE STRATEGY TREE / 3 ACTIONS PER LAP</span><span><Badge tone="derived">{tree.nodeCount.toLocaleString()} NODES</Badge> <Badge tone={tree.treeVersion ? 'simulated' : 'derived'}>{tree.treeVersion ?? 'LOCAL FALLBACK'}</Badge>{tree.regulationEra && <Badge tone="simulated">{tree.regulationEra}</Badge>}</span></div>
    <div className="dt-tree-signals">
      <div><span>ML RECOMMENDATION</span><b className={tree.classifierAction === 'ATTACK' ? 'tree-attack' : tree.classifierAction === 'SAVE' ? 'tree-save' : 'tree-delay'}>{tree.classifierAction ?? '—'}</b><em>lap-chip classifier output</em></div>
      <div><span>RECURSIVE BEST ACTION</span><b className={treeAction === 'ATTACK' ? 'tree-attack' : treeAction === 'SAVE' ? 'tree-save' : 'tree-delay'}>{treeAction}</b><em>tree choice after both-car energy response</em></div>
      {tree.decisionContext && <div><span>RACE-CONTROL GATE</span><b className={tree.decisionContext.overtakeActionsEnabled ? 'tree-save' : 'tree-attack'}>{tree.decisionContext.raceControl}</b><em>{tree.decisionContext.pitDistorted ? 'pit-cycle context · no pass claim' : 'normal overtake window'}</em></div>}
      {tree.pitTyreContext && <div><span>PIT / TYRE CONTEXT</span><b className={tree.pitTyreContext.futurePitCycleWithinHorizon ? 'tree-delay' : 'tree-save'}>{tree.pitTyreContext.futurePitCycleWithinHorizon ? 'PIT CYCLE OBSERVED' : 'STAY-OUT CONTEXT'}</b><em>{tree.pitTyreContext.futurePitCycleWithinHorizon ? `BOX HELD FIXED · ${tree.pitTyreContext.observedPitEvents?.length ?? 0} event(s)` : `no observed pit cycle in ${tree.pitTyreContext.horizonLaps ?? tree.horizon}-lap horizon`}</em></div>}
      {tree.decisionContext?.ruleContext && <div><span>OVERTAKE MODE RULE</span><b className={tree.decisionContext.ruleContext.eventSpecificDataLoaded ? 'tree-save' : 'tree-delay'}>{tree.decisionContext.ruleContext.eventSpecificDataLoaded ? 'EVENT APPENDIX LOADED' : 'APPENDIX NOT LOADED'}</b><em>{tree.decisionContext.ruleContext.application === 'TRACK_DISTANCE_ALIGNMENT_REQUIRED' ? 'track-distance alignment required before use' : 'tree uses the modelled analysis window only'}</em></div>}
      {tree.opponentPolicy && <div><span>OPPONENT RESPONSE</span><b className="tree-delay">CONSERVATIVE BEST RESPONSE</b><em>model assumption · not observed radio strategy</em></div>}
      {tree.stateProvenance && <div><span>FORWARD STATE CONTEXT</span><b className={tree.stateProvenance.carriedContextLaps ? 'tree-delay' : 'tree-save'}>{tree.stateProvenance.observedContextLaps} / {tree.stateProvenance.horizonLaps} OBSERVED</b><em>{tree.stateProvenance.carriedContextLaps ? `${tree.stateProvenance.carriedContextLaps} later lap(s) carry the focus context` : 'all tree steps use matched observed context'}</em></div>}
    </div>
    <div className="dt-tree-summary">
      <div><b>{tree.expectedLeadLaps.toFixed(1)}</b><span>TREE-ESTIMATED LAPS AHEAD</span></div>
      <div><b>{tree.actualLeadLaps}</b><span>REAL LAPS AHEAD · CAPPED AT {comparisonHorizon}</span></div>
      <div><b>{tree.actualFinishPosition ? `P${tree.actualFinishPosition}` : '—'}</b><span>REAL FINISH POSITION</span></div>
      <div><b className={tree.success ? 'positive' : ''}>{tree.success ? 'BETTER' : 'NOT YET'}</b><span>{persistenceDelta >= 0 ? '+' : ''}{persistenceDelta.toFixed(1)} LAPS VS REAL</span></div>
    </div>
    {tree.persistenceByHorizon?.length > 0 && <div className="dt-persistence">
      <div className="dt-persistence-head"><span>POSITION DURABILITY PROBABILITY</span><em>MODEL ESTIMATE / REAL RACE</em></div>
      <div className="dt-persistence-grid">
        {tree.persistenceByHorizon.map((item) => <div className="dt-persistence-cell" key={item.horizon}>
          <b>{item.horizon} LAP{item.horizon === 1 ? '' : 'S'}</b>
          <span>{item.estimatedProbability == null ? '—' : `${Math.round(item.estimatedProbability * 100)}%`}</span>
          <em>{item.estimatedProbability == null ? 'OUTSIDE HORIZON' : `REAL ${item.observed ? 'YES' : 'NO'}`}</em>
        </div>)}
      </div>
    </div>}
    {tree.socSensitivity?.cases?.length > 0 && <div className="dt-sensitivity">
      <div className="dt-persistence-head"><span>MODELLED ENERGY SENSITIVITY / ±{tree.socSensitivity.perturbationMj?.toFixed(2)} MJ</span><em>{tree.socSensitivity.stableRecommendation ? 'RECOMMENDATION STABLE' : 'RECOMMENDATION CHANGES'}</em></div>
      <div className="dt-sensitivity-grid">
        {tree.socSensitivity.cases.map((item) => <div className="dt-sensitivity-cell" key={item.id}>
          <span>{item.label}</span>
          <b className={item.recommendedAction === 'ATTACK' ? 'tree-attack' : item.recommendedAction === 'SAVE' ? 'tree-save' : 'tree-delay'}>{item.recommendedAction ?? '—'}</b>
          <em>SoC {item.attackerStartSocMj.toFixed(2)} / defender {item.defenderStartSocMj.toFixed(2)} MJ · {item.expectedLeadLaps.toFixed(1)}L</em>
        </div>)}
      </div>
      <p>{tree.socSensitivity.note}</p>
    </div>}
    {tree.energyCalibrationSensitivity?.cases?.length > 0 && <div className="dt-sensitivity">
      <div className="dt-persistence-head"><span>MODELLED ENERGY-CALIBRATION SENSITIVITY / ±{tree.energyCalibrationSensitivity.variationPercent}%</span><em>{tree.energyCalibrationSensitivity.stableRecommendation ? 'RECOMMENDATION STABLE' : 'RECOMMENDATION CHANGES'}</em></div>
      <div className="dt-sensitivity-grid">
        {tree.energyCalibrationSensitivity.cases.map((item) => <div className="dt-sensitivity-cell" key={item.id}>
          <span>{item.label}</span>
          <b className={item.recommendedAction === 'ATTACK' ? 'tree-attack' : item.recommendedAction === 'SAVE' ? 'tree-save' : 'tree-delay'}>{item.recommendedAction ?? '—'}</b>
          <em>Deploy ×{item.deployScale.toFixed(2)} · harvest ×{item.harvestScale.toFixed(2)} · {item.expectedLeadLaps.toFixed(1)}L</em>
        </div>)}
      </div>
      <p>{tree.energyCalibrationSensitivity.note}</p>
    </div>}
    <div className="dt-tree-path">
      {tree.path.map((step) => <div className="dt-tree-step" key={`${step.lap}-${step.action}`}>
        <span>L{step.lap}</span><b className={step.action === 'ATTACK' ? 'tree-attack' : step.action === 'SAVE' ? 'tree-save' : 'tree-delay'}>{step.action}</b>
        <em>{step.role} · {Math.round(step.probability * 100)}% · lead {step.leadLaps.toFixed(1)}L · OPP {step.opponentAction ?? '—'} · {step.pitPlan} · SoC {step.ourSoc.toFixed(2)} / defender {step.defenderSoc.toFixed(2)} MJ · {step.contextSource === 'FOCUS_CONTEXT_CARRIED' ? 'CONTEXT CARRIED' : 'OBSERVED CONTEXT'}</em>
      </div>)}
    </div>
    <button className="dt-tree-toggle" type="button" onClick={() => setShowBranches((value) => !value)}>
      {showBranches ? 'HIDE ALL BRANCHES' : 'SHOW ALL 3-ACTION BRANCHES'}
    </button>
    {showBranches && <div className="dt-tree-branches">
      {pathNodes.map((node) => <div className="dt-tree-branch-row" key={`branches-${node.lap}`}>
        <div className="dt-tree-branch-label">L{node.lap} · {node.role}</div>
        <div className="dt-tree-branch-grid">
          {(node.children ?? []).map((child) => <div className={`dt-tree-branch ${child.best ? 'selected' : ''}`} key={`${node.lap}-${child.action}`}>
            <div><b className={child.action === 'ATTACK' ? 'tree-attack' : child.action === 'SAVE' ? 'tree-save' : 'tree-delay'}>{child.action}</b>{child.best && <small> BEST</small>}</div>
            <span>{Math.round((child.probability ?? 0) * 100)}% · lead {(child.leadLaps ?? 0).toFixed(1)}L</span>
            <span>OPP {child.opponentAction ?? '—'} · SoC {(child.ourSoc ?? 0).toFixed(2)} / {(child.defenderSoc ?? 0).toFixed(2)} MJ</span>
          </div>)}
        </div>
      </div>)}
    </div>}
    <p className="ov-notes">The lap-chip letter is the classifier recommendation; the recursive path is a separate decision. Every tree node evaluates ATTACK, SAVE, and DELAY, then models the opponent’s response and role reversal after a pass. {tree.opponentPolicy?.description} {tree.stateProvenance?.note} {comparisonExplanation} The horizon is {tree.horizon} laps and energy values are modelled surrogates, not measured battery telemetry. {tree.pitTyreContext?.handling}</p>
  </section>
}

// ─── Race / driver selector ──────────────────────────────────────────────────

const YEARS = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018]
const ROUNDS = Array.from({ length: 24 }, (_, i) => i + 1)

export function RaceSelector({ sel, onChange, drivers, events }) {
  const set = (patch) => onChange({ ...sel, ...patch })
  // Prefer real race names from the decision-point cache; fall back to bare
  // round codes only while the list loads or a season has no extracted data.
  const rounds = events?.length ? events : ROUNDS.map((r) => ({ round: r, name: null }))
  const hasCurrent = rounds.some((e) => Number(e.round) === Number(sel.round))
  return <div className="ov-toolbar dt-selector">
    <label className="ov-select"><span>SEASON</span>
      <select value={sel.year} onChange={(e) => set({ year: Number(e.target.value) })}>
        {YEARS.map((y) => <option key={y} value={y}>{y}</option>)}
      </select>
    </label>
    <label className="ov-select"><span>ROUND</span>
      <select value={sel.round} onChange={(e) => set({ round: Number(e.target.value) })}>
        {!hasCurrent && <option value={sel.round}>{`R${sel.round} (no data)`}</option>}
        {rounds.map((e) => <option key={e.round} value={e.round}>{e.name || `R${e.round}`}</option>)}
      </select>
    </label>
    <label className="ov-select"><span>SESSION</span>
      <select value={sel.session} onChange={(e) => set({ session: e.target.value })}>
        {['R', 'Q', 'P', 'S', 'FP1', 'FP2', 'FP3'].map((s) => <option key={s} value={s}>{s}</option>)}
      </select>
    </label>
    <label className="ov-select"><span>DRIVER (ENERGY)</span>
      <select value={sel.driver} onChange={(e) => set({ driver: e.target.value })}>
        {(drivers?.length ? drivers : [sel.driver]).map((d) => <option key={d} value={d}>{d}</option>)}
      </select>
    </label>
    <div className="ov-toolbar-note"><span>DATA</span><b>{events?.length ? `${events.length} RACES IN DROPDOWN` : 'PICK A CACHED RACE'}</b></div>
  </div>
}

const INCIDENT_TONE = {
  MISS_ATTACK: 'ATTACK CHANCE',
  DEFEND: 'DEFEND',
  TOOK: 'OBSERVED TAKE',
  SAVE_CHANCE: 'SAVE BATTERY',
}

function sessionNameOf(sel) {
  if (!sel?.session || sel.session === 'R') return 'Race'
  if (sel.session === 'Q') return 'Qualifying'
  if (sel.session === 'S') return 'Sprint'
  return sel.session
}

function mapSimCall(call) {
  if (call === 'ATTACK' || call === 'SAVE') return call
  return 'HOLD'
}

function buildWhatIfRequest(sel, focus, call) {
  const model = focus?.model
  const chosen = mapSimCall(call || interestingCall(model))
  return {
    year: sel.year,
    round: sel.round,
    session: sessionNameOf(sel),
    driver: sel.driver,
    lap: focus.lap,
    otherDriver: focus.otherDriver,
    call: chosen,
    kind: focus.kind,
    problem: model?.resultIf || focus.problem,
    theyDid: model?.theyDid || focus.live?.action,
    position: focus.live?.position,
    gapToAheadS: focus.live?.gapToAheadS,
    gapToBehindS: focus.live?.gapToBehindS,
    leftPct: focus.live?.leftPct,
    observedFinish: model?.observedFinish,
    raceEnd: model?.raceEndIfCall,
    takes: model?.takesIfCall,
    keyTake: model?.keyTake,
    opponent: model?.opponent,
    netPass: model?.netPassIfAttack,
    pPass: model?.pPassIfAttack,
    whatIfCall: chosen,
    story: model?.story,
  }
}

export function TelemetryIncidentBoard({ sel, onLoadLaps, onOpenSimulation }) {
  const [trend, setTrend] = useState({ loading: true })
  const [focusId, setFocusId] = useState(null)
  useEffect(() => {
    if (!sel?.year || !sel?.round || !sel?.driver) return undefined
    let live = true
    setTrend({ loading: true })
    setFocusId(null)
    fetchEnergyTrend({ year: sel.year, round: sel.round, session: sel.session, driver: sel.driver })
      .then((data) => { if (live) setTrend(data?.error ? { error: data.error } : { data }) })
      .catch((error) => { if (live) setTrend({ error: error.message }) })
    return () => { live = false }
  }, [sel.year, sel.round, sel.session, sel.driver])

  if (trend.loading) return <section className="ov-panel dt-incident-panel"><div className="lx-loading"><span className="lx-spinner" />FINDING ENERGY × OVERTAKE INCIDENTS FOR {sel.driver}…</div></section>
  if (trend.error) return <section className="ov-panel dt-incident-panel"><p className="lx-empty">Incidents unavailable. {trend.error}</p></section>
  const data = trend.data
  const incidents = data?.incidents ?? []
  const focus = incidents.find((item) => item.id === focusId) ?? incidents[0]
  const live = focus?.live
  const model = focus?.model
  const pct01 = (value) => value == null ? '—' : `${Math.round(Number(value) * 100)}%`
  const loadTraces = () => {
    if (!focus || !onLoadLaps) return
    const session = sessionNameOf(sel)
    const rows = [{
      year: sel.year, round: sel.round, session, driver: sel.driver, lap: focus.lap, color: '#ff7043',
    }]
    if (focus.otherDriver) {
      rows.push({
        year: sel.year, round: sel.round, session, driver: focus.otherDriver, lap: focus.lap, color: '#9db7ff',
      })
    }
    onLoadLaps(rows)
  }

  return <section className="ov-panel dt-incident-panel">
    <div className="ov-panel-head">
      <span>LIVE INCIDENTS / {data.driver} · {String(data.event?.name || '').toUpperCase()}</span>
      <Badge tone="derived">THIS RACE + 2018–2025 FOREST</Badge>
    </div>
      <p className="ov-notes">Each chip is a real close fight; ATTACK spends ~{data.attackExtraMj ?? 0.45} MJ extra, SAVE keeps ~{data.saveKeepMj ?? 0.27} MJ, HOLD leaves energy as-is.</p>
    {!incidents.length && <p className="lx-empty">No close DRS, chase, or high-spend laps for {sel.driver} in this race.</p>}
    <div className="dt-incident-list">
      {incidents.map((item) => <button key={item.id} type="button" className={`dt-incident-chip ${item.kind} ${focus?.id === item.id ? 'active' : ''}`} onClick={() => setFocusId(item.id)}>
        <em>{INCIDENT_TONE[item.kind] || item.kind}</em>
        <b>L{item.lap}</b>
        <span>{item.otherDriver ? `vs ${item.otherDriver}` : item.live?.event}</span>
      </button>)}
    </div>
    {focus && <div className="dt-incident-live">
      <div className="dt-incident-radio">
        <span>LIVE AT LAP {focus.lap}</span>
        <strong>P{live?.position ?? '—'} {live?.ahead ? `· ${live.gapToAheadS?.toFixed?.(2) ?? live.gapToAheadS}s vs ${live.ahead}` : ''}</strong>
        <em>{live?.chased ? `${live.behind} is ${live.gapToBehindS?.toFixed?.(2) ?? live.gapToBehindS}s behind — close enough to pass` : 'nobody within 1.2s behind'}</em>
        <p>{focus.problem}</p>
      </div>
      <div className="dt-incident-battery">
        <div><span>ENERGY LEFT</span><b>{live?.leftPct == null ? '—' : `${Math.round(live.leftPct)}%`}</b><em>unused share of the 4 MJ energy-store model — not the team battery</em></div>
        <div><span>THEY SPENT</span><b>{live?.action || '—'}</b><em>{mj(live?.consumedMj)} deployed on this lap from that store</em></div>
        <div><span>MODEL PICK</span><b className={`call-${model?.call?.toLowerCase?.() || 'hold'}`}>{model?.call || '—'}</b><em>{model?.theyDid === model?.call ? 'same as what they spent — the useful branch is the other call' : (model?.couldDiffer ? `classified finish ${finishPos(model.observedFinish)} → ${finishPos(model.raceEndIfCall)}` : 'no extra whole place before the flag')}</em></div>
      </div>
      {model?.opponent && <div className="dt-incident-odds">
        <div><span>IF THEY ATTACK</span><b>{oddsPct(model.pPassIfAttack)}</b><em>trained chance this pass lands in 2 laps</em></div>
        <div><span>{model.opponent.driver} HOLDS</span><b>{oddsPct(model.opponent.pHold)}</b><em>they {model.opponent.theyDid || 'spent'} · {model.opponent.leftPct == null ? '—' : `${Math.round(model.opponent.leftPct)}%`} energy left</em></div>
        <div><span>NET PASS</span><b>{oddsPct(model.netPassIfAttack)}</b><em>our pass chance after their hold chance</em></div>
      </div>}
      <p className="dt-incident-why">{model?.why}</p>
      <div className="dt-incident-finish">
        <div><span>REAL FINISH</span><b>{finishPos(model?.observedFinish)}</b><em>classified result from the real race</em></div>
        <div><span>WHAT-IF FINISH IF {model?.whatIfCall || model?.call || 'CALL'}</span><b className={model?.placesVsHold > 0 ? 'positive' : ''}>{finishPos(model?.raceEndIfCall)}</b><em>{model?.extraPlacesIfCall ? `+${model.extraPlacesIfCall} whole place${model.extraPlacesIfCall === 1 ? '' : 's'} · key lap ${model.keyTake ? `L${model.keyTake.lap} vs ${model.keyTake.ahead}` : 'this fight'}` : 'same classified finish'}</em></div>
        <div><span>CHANCE OF BEING PASSED</span><b>{pct01(model?.hold?.pLose)} → {pct01((model?.whatIfCall === 'ATTACK' || (!model?.whatIfCall && model?.call === 'ATTACK') ? model?.attack : model?.whatIfCall === 'SAVE' || model?.call === 'SAVE' ? model?.save : model?.hold)?.pLose)}</b><em>{model?.avoidedLose >= 0.04 ? 'lower chance the car behind passes them in the next 2 laps' : 'chance of being passed in the next 2 laps barely changes'}</em></div>
      </div>
      {!!model?.takesIfCall?.length && <p className="dt-incident-takes">{takeLine(model.takesIfCall)}</p>}
      <div className="dt-incident-alts">
        {['attack', 'save', 'hold'].map((key) => {
          const choice = model?.[key]
          if (!choice) return <div key={key} className="dt-incident-alt is-off"><span>{key.toUpperCase()}</span><b>EMPTY</b><em>not enough energy left for an extra ATTACK</em></div>
          return <div key={key} className={`dt-incident-alt ${(model.whatIfCall || model.call) === key.toUpperCase() ? 'picked' : ''}`}>
            <span>{key.toUpperCase()}{model.theyDid === key.toUpperCase() ? ' · WHAT THEY DID' : ''}</span>
            <b>{finishPos(choice.raceEnd ?? choice.endPosition)}</b>
            <em>{choice.extraPlaces ? `+${choice.extraPlaces} place${choice.extraPlaces === 1 ? '' : 's'} · ` : ''}{takeLine(choice.takes)} · {pct01(choice.pGain ?? choice.convertIn2)} this fight</em>
            {onOpenSimulation && <button type="button" className="dt-incident-alt-sim" onClick={() => onOpenSimulation(buildWhatIfRequest(sel, focus, key.toUpperCase()))}>SIM {key.toUpperCase()}</button>}
          </div>
        })}
      </div>
      {model?.resultIf && <p className="ov-notes">{model.resultIf}</p>}
      {model?.avoidedIf && <p className="ov-notes">{model.avoidedIf}</p>}
      <p className="ov-notes">A take only counts if the trained pass chance still beats the other car’s hold chance. SAVE now is for a later lap that still has energy; ATTACK now spends that energy here.</p>
      <div className="dt-incident-actions">
        <button type="button" className="dt-incident-load" onClick={loadTraces}>SHOW THIS LAP ON THE TRACE{focus.otherDriver ? ` · ${sel.driver} vs ${focus.otherDriver}` : ''}</button>
        {onOpenSimulation && <button type="button" className="dt-incident-load dt-incident-whatif" onClick={() => onOpenSimulation(buildWhatIfRequest(sel, focus))}>
          WHAT IF {interestingCall(model)} · SHOW ON TRACK
        </button>}
      </div>
    </div>}
  </section>
}

function CachedBatteryClip({ sel }) {
  const [clip, setClip] = useState({ loading: false })
  useEffect(() => {
    if (!sel?.year || !sel?.round) return undefined
    let live = true
    setClip({ loading: true })
    fetchBatteryClip({
      year: sel.year,
      round: sel.round,
      session: sel.session,
      driver: sel.driver,
    }).then((data) => { if (live) setClip({ data }) })
      .catch((error) => { if (live) setClip({ error: error.message }) })
    return () => { live = false }
  }, [sel.year, sel.round, sel.session, sel.driver])

  const data = clip.data
  const box = data?.box
  const actual = data?.actualRace
  return <section className="ov-panel dt-clip-panel">
    <div className="ov-panel-head">
      <span>C5.2 BATTERY CLIP / SELECTED RACE</span>
      <Badge tone="simulated">MODELLED ES · REAL LAP TIMES</Badge>
    </div>
    {clip.loading && <div className="lx-loading"><span className="lx-spinner" />CLIPPING CACHED LAPS…</div>}
    {clip.error && <p className="lx-empty">This race is not in the session cache yet, or the backend on :8787 is the old process. {clip.error}</p>}
    {data && !clip.loading && <>
      <div className="dt-clip-box">
        <div><span>USED</span><b>{box?.consumedMj ?? '—'} MJ</b></div>
        <div><span>LEFT</span><b>{box?.leftMj ?? '—'} MJ · {box?.leftPct ?? '—'}%</b></div>
        <div><span>PAIR</span><b>{actual?.driver} vs {actual?.defender}</b></div>
        <div><span>LAPS</span><b>{actual?.sharedTimedLaps ?? '—'}</b></div>
      </div>
      <p className="ov-notes">{data.call}</p>
    </>}
  </section>
}

// ─── Data hook: decision points + ML predictions + energy race ───────────────

export function useRaceEngine(sel, activeTab = 'STRATEGY') {
  const needsDecision = activeTab === 'STRATEGY' || activeTab === 'OVERTAKE'
  const needsEnergy = activeTab === 'STRATEGY' || activeTab === 'ENERGY'
  const needsEvents = needsDecision || needsEnergy || activeTab === 'TELEMETRY'
  const [decision, setDecision] = useState({ loading: true })
  const [energy, setEnergy] = useState({ loading: false })
  const [preds, setPreds] = useState(null)
  const [report, setReport] = useState(null)
  const [events, setEvents] = useState([])

  useEffect(() => {
    if (activeTab !== 'VALIDATION') return undefined
    let live = true
    fetchModelReport().then((r) => { if (live) setReport(r) }).catch(() => { if (live) setReport(null) })
    return () => { live = false }
  }, [activeTab])

  // Round -> race-name list for the season, read from the cache (instant).
  useEffect(() => {
    if (!needsEvents) {
      setEvents([])
      return undefined
    }
    let live = true
    setEvents([])
    Promise.all([
      fetchEvents(sel.year).catch(() => []),
      fetchCachedRaces(sel.year).catch(() => []),
    ]).then(([rounds, cached]) => {
      if (!live) return
      const byRound = new Map()
      for (const item of [...rounds, ...cached]) {
        const prev = byRound.get(Number(item.round)) || { round: Number(item.round) }
        byRound.set(Number(item.round), { ...prev, ...item })
      }
      setEvents([...byRound.values()].sort((a, b) => a.round - b.round))
    }).catch(() => { if (live) setEvents([]) })
    return () => { live = false }
  }, [sel.year, needsEvents])

  // Decision points (instant, cached) — also yields the driver list.
  useEffect(() => {
    if (!needsDecision) {
      setDecision({ loading: false, data: null })
      setPreds(null)
      return undefined
    }
    let live = true
    setDecision({ loading: true })
    setPreds(null)
    fetchDecisionPoints(sel.year, sel.round, sel.session)
      .then((dp) => { if (live) setDecision({ data: dp }) })
      .catch((e) => { if (live) setDecision({ error: e.message }) })
    return () => { live = false }
  }, [sel.year, sel.round, sel.session, needsDecision])

  // Score every observed close battle for inference. The extractor keeps the
  // clean label-eligible subset separately for training and validation, so an
  // ambiguous future outcome never becomes a training target merely because
  // it remains useful to inspect in Strategy.
  useEffect(() => {
    if (!needsDecision) return undefined
    const rows = decision.data?.analysisRows ?? decision.data?.rows
    if (!rows?.length) return
    let live = true
    setPreds({ loading: true })
    predictOvertake(rows, decision.data?.ruleContext)
      .then((p) => {
        if (!live) return
        const keyFor = (row) => [row.year, row.round, row.session, row.lap, row.driver, row.defender].join(':')
        const predictionByRow = new Map(rows.map((row, index) => [keyFor(row), p.predictions[index]]))
        const cleanRows = decision.data?.rows ?? []
        setPreds({
          analysis: p.predictions,
          clean: cleanRows.map((row) => predictionByRow.get(keyFor(row))),
        })
      })
      .catch((error) => {
        // Do not let a temporary scoring failure masquerade as an absence of
        // close-battle data.  Strategy can now show an honest retry state.
        if (live) setPreds({ loading: false, error: error.message })
      })
    return () => { live = false }
  }, [decision.data, needsDecision])

  // Energy race (slow on first fetch, then cached).
  useEffect(() => {
    if (!needsEnergy || !sel.driver) {
      setEnergy({ loading: false, data: null })
      return undefined
    }
    let live = true
    setEnergy({ loading: true })
    fetchEnergyRace(sel.year, sel.round, sel.session, sel.driver)
      .then((d) => { if (live) setEnergy({ data: d }) })
      .catch((e) => { if (live) setEnergy({ error: e.message }) })
    return () => { live = false }
  }, [sel.year, sel.round, sel.session, sel.driver, needsEnergy])

  const drivers = useMemo(() => {
    const rows = decision.data?.rows ?? []
    const set = new Set()
    // The participant roster contains every recorded entrant, including a
    // retirement with no classified position. Decision rows are only a
    // filtered subset and must never make other drivers vanish from the UI.
    ;(decision.data?.participants ?? []).forEach((driver) => set.add(driver))
    Object.keys(decision.data?.finishPositions ?? {}).forEach((driver) => set.add(driver))
    rows.forEach((r) => { if (r.driver) set.add(r.driver); if (r.defender) set.add(r.defender) })
    const cached = events.find((item) => Number(item.round) === Number(sel.round))
    ;(cached?.drivers ?? []).forEach((driver) => set.add(driver))
    return [...set].sort()
  }, [decision.data, events, sel.round])

  return { decision, energy, preds, drivers, report, events }
}

// ─── OVERTAKE tab ────────────────────────────────────────────────────────────

function observedOutcomeSummary(row) {
  if (row.observedPassLap != null) {
    const held = Number(row.observedLeadLaps ?? 0)
    return `PASSED L${row.observedPassLap} · HELD ${held} LAP${held === 1 ? '' : 'S'}`
  }
  return row.passedNow ? `PASSED L${Number(row.lap) + 1} · NOT HELD` : 'NO DURABLE PASS'
}

export function OvertakeTab({ sel, decision, preds }) {
  const dp = decision.data
  if (decision.loading) return <div className="lx-loading"><span className="lx-spinner" />LOADING DECISION POINTS</div>
  if (decision.error) return <p className="lx-empty">No decision points for {sel.year} R{sel.round} {sel.session}. {decision.error}</p>
  // Switching from Track/Telemetry clears the previous decision payload before
  // the new request effect runs. Keep the page alive during that one render
  // instead of dereferencing null and producing a blank React screen.
  if (!dp) return <div className="lx-loading"><span className="lx-spinner" />LOADING DECISION POINTS</div>

  const rows = dp.rows ?? []
  const ruleContext = dp.ruleContext ?? {}
  const eventRulesLoaded = Boolean(ruleContext.eventSpecificDataLoaded)
  const analysisWindowNote = eventRulesLoaded
    ? `FIA event appendix loaded: detection gap ${ruleContext.officialDetectionGapS ?? '—'}s; detection/activation line use still needs matching track-distance data.`
    : (ruleContext.status === 'EVENT_APPENDIX_REQUIRED'
      ? 'The 2026 FIA detection/activation appendix is not loaded here; this is a modelled analysis window, not an official event rule.'
      : 'Historical sessions use this derived battle window for comparable analysis; it is not a claim about the historical DRS rule.')
  const cleanPredictions = preds?.clean ?? preds
  const scored = rows.map((r, i) => ({ ...r, pred: cleanPredictions?.[i] })).filter((r) => r.pred)
  const agree = scored.filter((r) => r.pred.label === r.label).length
  const accuracy = scored.length ? Math.round((agree / scored.length) * 100) : null
  const lc = dp.labelCounts ?? {}
  const total = rows.length || 1


  return <div className="dt-overtake">
    <StrategyRaceStory sel={sel} />
    <section className="ov-panel">
      <div className="ov-panel-head"><span>DETECTED DECISION POINTS / {dp.eventName?.toUpperCase()}</span><span><Badge tone="derived">GAP + SPEED-TRAP</Badge> <Badge tone="real">{dp.lappingExcludedCount ?? 0} LAPPING EXCLUDED</Badge></span></div>
      <div className="dt-bignum"><strong>{rows.length}</strong><span>BATTLES DETECTED · {dp.totalLaps} LAPS · ANALYSIS WINDOW {dp.maxGapThresholdS}s</span></div>
      <div className="ov-factor dt-rule-disclosure"><span>Overtake window <Badge tone={eventRulesLoaded ? 'real' : 'simulated'}>{eventRulesLoaded ? 'EVENT APPENDIX' : 'MODELLED FILTER'}</Badge></span><b>{eventRulesLoaded ? 'EVENT-SPECIFIC CONTEXT' : (ruleContext.status === 'EVENT_APPENDIX_REQUIRED' ? 'NOT EVENT-SPECIFIC' : 'HISTORICAL PROXY')}</b><em>{analysisWindowNote}</em></div>
      <div className="dt-distbar">
        {STRATEGY_ORDER.map((k) => <i key={k} style={{ width: `${((lc[k] ?? 0) / total) * 100}%`, background: STRATEGY_COLORS[k] }} title={`${k}: ${lc[k] ?? 0}`} />)}
      </div>
      <div className="dt-distlegend">
        {STRATEGY_ORDER.map((k) => <span key={k}><i style={{ background: STRATEGY_COLORS[k] }} />{k} · {lc[k] ?? 0} ({Math.round(((lc[k] ?? 0) / total) * 100)}%)</span>)}
      </div>
      <div className="ov-panel-head second"><span>MODEL vs GROUND TRUTH (THIS RACE)</span><Badge tone="simulated">RANDOMFOREST</Badge></div>
      <div className="dt-accuracy">
        <b>{accuracy == null ? '—' : `${accuracy}%`}</b>
        <span>{agree} of {scored.length} decision points · predicted label matches the observed outcome</span>
      </div>
      <p className="ov-notes">Labels are modelled outcomes: ATTACK = pass made and held {dp.holdLaps} laps, DELAY = durable pass within {dp.holdLaps} laps, SAVE = no durable pass. Lapping/backmarker candidates are excluded before training ({dp.lappingExcludedCount ?? 0} in this race). This section describes the selected race only; global unseen-season validation is shown separately under VALIDATION.</p>
    </section>

    <section className="ov-panel dt-dp-panel">
      <div className="ov-panel-head"><span>DECISION POINTS · MODEL PROBABILITIES</span><b>{scored.length} ROWS</b></div>
      <div className="dt-dp-head"><span>LAP</span><span>ATTACKER → DEFENDER</span><span>GAP</span><span>Δ SPD</span><span>MODEL</span><span>TRUTH / OBSERVED</span></div>
      <div className="dt-dp-list">
        {scored.map((r, i) => {
          const hit = r.pred.label === r.label
          return <div className={`dt-dp-row ${hit ? 'hit' : 'miss'}`} key={`${r.lap}-${r.driver}-${r.defender}-${i}`}>
            <span className="dt-lap">L{r.lap}</span>
            <span className="dt-matchup"><b>{r.driver}</b>→{r.defender}<em>P{r.position}</em></span>
            <span className="dt-cell">{num(r.gapS, 2)}s</span>
            <span className="dt-cell">{r.speedDeltaKph == null ? '—' : `${r.speedDeltaKph > 0 ? '+' : ''}${r.speedDeltaKph} km/h`}</span>
            <span className="dt-model">
              <b style={{ color: STRATEGY_COLORS[r.pred.label] }}>{r.pred.label}</b>
              <ProbBar probabilities={r.pred.probabilities} height={5} />
            </span>
            <span className="dt-truth" style={{ color: STRATEGY_COLORS[r.label] }}><b>{r.label}</b><small>{observedOutcomeSummary(r)}</small></span>
          </div>
        })}
      </div>
    </section>
  </div>
}

// ─── VALIDATION tab: global held-out model and replay evaluation ─────────────

export function ValidationTab({ report }) {
  if (!report) return <div className="lx-loading"><span className="lx-spinner" />LOADING VALIDATION REPORT</div>
  const trainYears = report.temporalSplit?.trainYears ?? []
  const testYears = report.temporalSplit?.testYears ?? []
  const heldOutRaces = report.testByRace ?? []
  const heldOutDrivers = report.testByDriver ?? []
  const heldOutTracks = report.testByTrack ?? []
  const holdAcc = report.testAccuracy != null ? Math.round(report.testAccuracy * 100) : null
  const holdMacroF1 = report.testMacroF1 != null ? Math.round(report.testMacroF1 * 100) : null
  const alwaysSave = report.baselines?.alwaysSaveAccuracy != null ? Math.round(report.baselines.alwaysSaveAccuracy * 100) : null
  const gapOnly = report.baselines?.gapOnlyAccuracy != null ? Math.round(report.baselines.gapOnlyAccuracy * 100) : null
  const ci = report.testUncertainty
  const beatsBaseline = report.modelVsAlwaysSave?.beatsBaseline
  const modelComparison = Object.entries(report.modelComparison ?? {})
  const classMetrics = ['ATTACK', 'DELAY', 'SAVE'].map((label) => ({
    label,
    rows: report.classCounts?.test?.[label] ?? 0,
    metrics: report.testReport?.[label],
  })).filter((item) => item.metrics)
  const replayPersistence = report.replayPersistence
  const replayModels = Object.entries(replayPersistence?.models ?? {})
  const primaryReplay = replayPersistence?.models?.RandomForestClassifier ?? replayModels[0]?.[1]
  const durability = report.passDurabilityComponent
  const immediatePass = report.immediatePassComponent
  const rollingSelection = report.rollingSelection
  const labelAudit = report.decisionLabelAudit
  const actionPolicy = report.actionPolicy
  const rawArgmaxMetrics = report.rawArgmaxMetrics
  const rollingModels = Object.entries(rollingSelection?.aggregate ?? {})
  const selectedRollingCandidate = rollingSelection?.selectedCandidate ?? rollingModels.reduce(
    (best, [name, metrics]) => (!best || (metrics.weightedMacroF1 ?? -1) > (best.metrics.weightedMacroF1 ?? -1))
      ? { name, metrics } : best,
    null,
  )?.name

  return <div className="dt-validation-page">
    {labelAudit && !labelAudit.readyForTraining && <section className="ov-panel dt-validation-panel">
      <div className="ov-panel-head"><span>LABEL POLICY REFRESH REQUIRED</span><Badge tone="derived">TRAINING SAFEGUARD</Badge></div>
      <div className="dt-validation-verdict warning">
        CURRENT SCORES USE THE PREVIOUS LABEL CACHE
        <span>The durable v6 label policy is implemented, but only {labelAudit.summary?.currentSchemaFiles ?? 0} of {labelAudit.summary?.cachedRaceFiles ?? 0} cached race files have been rebuilt. Retraining is intentionally blocked until all cached race files use one policy.</span>
      </div>
    </section>}
    <section className="ov-panel dt-validation-panel">
      <div className="ov-panel-head"><span>HELD-OUT RACE VALIDATION / {testYears.join(', ')}</span><Badge tone="real">UNSEEN DATA</Badge></div>
      <p className="ov-notes">The selected race controls the Overtake page. This page reports the global frozen evaluation: the model trains on {trainYears[0]}–{trainYears[trainYears.length - 1]} and is tested only on completed {testYears.join(', ')} races.</p>
      <div className="dt-validation-summary">
        <div><b>{holdAcc == null ? '—' : `${holdAcc}%`}</b><span>MODEL ACCURACY</span></div>
        <div><b>{holdMacroF1 == null ? '—' : `${holdMacroF1}%`}</b><span>MACRO F1</span></div>
        <div><b>{alwaysSave == null ? '—' : `${alwaysSave}%`}</b><span>ALWAYS-SAVE BASELINE</span></div>
        <div><b>{gapOnly == null ? '—' : `${gapOnly}%`}</b><span>GAP-ONLY BASELINE</span></div>
        <div><b>{ci?.lower95 == null ? '—' : `${Math.round(ci.lower95 * 100)}–${Math.round(ci.upper95 * 100)}%`}</b><span>RACE-LEVEL 95% RANGE</span></div>
      </div>
      <div className={`dt-validation-verdict ${beatsBaseline ? 'positive' : 'warning'}`}>
        {beatsBaseline ? 'MODEL CURRENTLY BEATS ALWAYS-SAVE' : 'MODEL CURRENTLY DOES NOT BEAT ALWAYS-SAVE'}
        <span>{beatsBaseline ? 'The holdout result supports further comparison.' : 'Treat this as a diagnostic result; the transparent baseline remains stronger.'}</span>
      </div>
      <div className="dt-validation-head"><span>RACE</span><span>ROWS</span><span>ACCURACY</span><span>MACRO F1</span><span>ALWAYS SAVE</span><span>GAP ONLY</span></div>
      <div className="dt-validation-list">
        {heldOutRaces.map((race) => <div className="dt-validation-row" key={`${race.year}-${race.round}-${race.session}`}>
          <span>{race.year} {race.eventName || `Round ${race.round}`}</span>
          <span>{race.rows}</span>
          <b className={race.accuracy >= race.alwaysSaveAccuracy ? 'positive' : ''}>{Math.round(race.accuracy * 100)}%</b>
          <b>{Math.round(race.macroF1 * 100)}%</b>
          <span>{Math.round(race.alwaysSaveAccuracy * 100)}%</span>
          <span>{race.gapOnlyAccuracy == null ? '—' : `${Math.round(race.gapOnlyAccuracy * 100)}%`}</span>
        </div>)}
      </div>
      <p className="ov-notes">These are classification metrics for each unseen race. They do not claim that a retrospective replay changes the real race; persistence and counterfactual results are scored separately below.</p>
    </section>

    {heldOutDrivers.length > 0 && <section className="ov-panel dt-validation-panel">
      <div className="ov-panel-head"><span>HELD-OUT CLASSIFICATION / ATTACKING DRIVER</span><Badge tone="real">2026 ONLY</Badge></div>
      <p className="ov-notes">Each row contains only decision points where that driver was the car behind. Use this to spot driver-specific weaknesses; small samples should not be over-interpreted.</p>
      <div className="dt-validation-head"><span>ATTACKER</span><span>ROWS</span><span>ACCURACY</span><span>MACRO F1</span><span>ALWAYS SAVE</span><span>GAP ONLY</span></div>
      <div className="dt-validation-list">
        {heldOutDrivers.map((driver) => <div className="dt-validation-row" key={driver.driver}>
          <span>{driver.driver}</span><span>{driver.rows}</span>
          <b className={driver.accuracy >= driver.alwaysSaveAccuracy ? 'positive' : ''}>{Math.round(driver.accuracy * 100)}%</b>
          <b>{Math.round(driver.macroF1 * 100)}%</b>
          <span>{Math.round(driver.alwaysSaveAccuracy * 100)}%</span>
          <span>{Math.round(driver.gapOnlyAccuracy * 100)}%</span>
        </div>)}
      </div>
    </section>}

    {heldOutTracks.length > 0 && <section className="ov-panel dt-validation-panel">
      <div className="ov-panel-head"><span>HELD-OUT CLASSIFICATION / CIRCUIT</span><Badge tone="real">2026 ONLY</Badge></div>
      <p className="ov-notes">This exposes circuit context directly. It remains a classification slice, not proof that the model can issue a live legal Overtake Mode command at that circuit.</p>
      <div className="dt-validation-head"><span>CIRCUIT</span><span>ROWS</span><span>ACCURACY</span><span>MACRO F1</span><span>ALWAYS SAVE</span><span>GAP ONLY</span></div>
      <div className="dt-validation-list">
        {heldOutTracks.map((track) => <div className="dt-validation-row" key={track.eventName}>
          <span>{track.eventName}</span><span>{track.rows}</span>
          <b className={track.accuracy >= track.alwaysSaveAccuracy ? 'positive' : ''}>{Math.round(track.accuracy * 100)}%</b>
          <b>{Math.round(track.macroF1 * 100)}%</b>
          <span>{Math.round(track.alwaysSaveAccuracy * 100)}%</span>
          <span>{Math.round(track.gapOnlyAccuracy * 100)}%</span>
        </div>)}
      </div>
    </section>}

    {modelComparison.length > 0 && <section className="ov-panel dt-model-panel">
      <div className="ov-panel-head"><span>MODEL BENCHMARKS / SAME 2026 HOLDOUT</span><Badge tone="derived">FROZEN SPLIT</Badge></div>
      <div className="dt-model-head"><span>MODEL</span><span>ACCURACY</span><span>MACRO F1</span><span>STATUS</span></div>
      {modelComparison.map(([name, metrics]) => <div className="dt-model-row" key={name}>
        <span>{modelName(name)}</span>
        <b>{Math.round(metrics.accuracy * 100)}%</b>
        <b>{Math.round(metrics.macroF1 * 100)}%</b>
        <span className={metrics.production ? 'positive' : ''}>{metrics.status?.replaceAll('_', ' ') || (rollingModels.length ? 'FINAL HOLDOUT RESULT' : (metrics.production ? 'PRODUCTION CANDIDATE' : 'BENCHMARK ONLY'))}</span>
      </div>)}
      <p className="ov-notes">All candidates use the same frozen feature set, 2018–2025 training window, and unseen 2026 races. Production selection is not based on raw accuracy alone.</p>
    </section>}

    {actionPolicy && <section className="ov-panel dt-component-panel">
      <div className="ov-panel-head"><span>HISTORICAL ACTION POLICY / RANDOM FOREST</span><Badge tone="derived">2026 EXCLUDED</Badge></div>
      <p className="ov-notes">These are decision weights selected from {actionPolicy.selectionRows?.toLocaleString?.() ?? 'historical'} out-of-fold rows, not changed to fit 2026. A weight below 1 means the action needs stronger model evidence before it is recommended; probabilities themselves are unchanged.</p>
      <div className="dt-component-grid">
        <div><span>ATTACK WEIGHT</span><b>{actionPolicy.weights?.ATTACK == null ? '—' : `${actionPolicy.weights.ATTACK.toFixed(2)}×`}</b></div>
        <div><span>DELAY WEIGHT</span><b>{actionPolicy.weights?.DELAY == null ? '—' : `${actionPolicy.weights.DELAY.toFixed(2)}×`}</b></div>
        <div><span>HISTORICAL MACRO F1</span><b>{actionPolicy.weightedMacroF1 == null ? '—' : `${Math.round(actionPolicy.weightedMacroF1 * 100)}%`}</b></div>
        <div><span>RAW POLICY MACRO F1</span><b>{actionPolicy.rawArgmaxMacroF1 == null ? '—' : `${Math.round(actionPolicy.rawArgmaxMacroF1 * 100)}%`}</b></div>
      </div>
      {rawArgmaxMetrics && <p className="ov-notes">On the one-time 2026 holdout, this policy scored {holdMacroF1}% macro F1 and {holdAcc}% accuracy, compared with {Math.round((rawArgmaxMetrics.macroF1 ?? 0) * 100)}% macro F1 and {Math.round((rawArgmaxMetrics.accuracy ?? 0) * 100)}% accuracy from raw probability argmax.</p>}
    </section>}

    {classMetrics.length > 0 && <section className="ov-panel dt-class-panel">
      <div className="ov-panel-head"><span>HELD-OUT ACTION BALANCE / 2026</span><Badge tone="derived">OUTCOME LABELS</Badge></div>
      <p className="ov-notes">These are outcome-optimal labels, not claimed driver radio commands. The imbalance explains why predicting SAVE on every point can look accurate while missing useful ATTACK and DELAY cases.</p>
      <div className="dt-class-head"><span>LABEL</span><span>POINTS</span><span>PRECISION</span><span>RECALL</span><span>F1</span></div>
      {classMetrics.map(({ label, rows, metrics }) => <div className="dt-class-row" key={label}>
        <b className={`tree-${label.toLowerCase()}`}>{label}</b>
        <span>{rows}</span>
        <span>{Math.round((metrics.precision ?? 0) * 100)}%</span>
        <span>{Math.round((metrics.recall ?? 0) * 100)}%</span>
        <span>{Math.round((metrics['f1-score'] ?? 0) * 100)}%</span>
      </div>)}
      <p className="ov-notes">A future action policy must improve minority-action precision and recall as well as overall macro F1 before it can be considered stronger than the transparent baseline.</p>
    </section>}

    {immediatePass && <section className="ov-panel dt-component-panel">
      <div className="ov-panel-head"><span>IMMEDIATE-PASS COMPONENT / HELD-OUT 2026</span><Badge tone="derived">DESCRIPTIVE BENCHMARK</Badge></div>
      <p className="ov-notes">This estimates whether an immediate real on-track pass was observed from the causal battle context. It does not claim what would have happened if the driver had chosen a different action, so it is not yet used by the strategy tree.</p>
      <div className="dt-component-grid">
        <div><span>REAL PASS RATE</span><b>{immediatePass.testObservedPassRate == null ? '—' : `${Math.round(immediatePass.testObservedPassRate * 100)}%`}</b></div>
        <div><span>F1</span><b>{immediatePass.f1 == null ? '—' : `${Math.round(immediatePass.f1 * 100)}%`}</b></div>
        <div><span>BRIER</span><b>{immediatePass.brier == null ? '—' : immediatePass.brier.toFixed(3)}</b></div>
        <div><span>CAL. BRIER</span><b>{immediatePass.calibrationBenchmark?.status === 'BENCHMARK_ONLY' ? immediatePass.calibrationBenchmark.brier.toFixed(3) : '—'}</b></div>
        <div><span>REF. BRIER</span><b>{immediatePass.constantRateBrier == null ? '—' : immediatePass.constantRateBrier.toFixed(3)}</b></div>
        <div><span>AUC</span><b>{immediatePass.rocAuc == null ? '—' : immediatePass.rocAuc.toFixed(3)}</b></div>
      </div>
      <p className="ov-notes">Brier is probability error (lower is better); CAL. BRIER is a sigmoid map fitted through 2024 and calibrated only on 2025, never on the 2026 holdout. The reference is the historical pass rate. This remains observational and needs a causal action design before tree integration.</p>
    </section>}

    {rollingModels.length > 0 && <section className="ov-panel dt-model-panel">
      <div className="ov-panel-head"><span>MODEL SELECTION / ROLLING HISTORICAL VALIDATION</span><Badge tone="derived">2026 EXCLUDED</Badge></div>
      <p className="ov-notes">Each fold fits only earlier seasons, then validates the next full season. This evaluates whether a candidate keeps working across changing race contexts without using the final 2026 holdout.</p>
      <div className="dt-model-head"><span>MODEL</span><span>ACCURACY</span><span>MACRO F1</span><span>STATUS</span></div>
      {rollingModels.map(([name, metrics]) => <div className="dt-model-row" key={name}>
        <span>{modelName(name)}</span>
        <b>{Math.round(metrics.weightedAccuracy * 100)}%</b>
        <b>{Math.round(metrics.weightedMacroF1 * 100)}%</b>
        <span className={selectedRollingCandidate === name ? 'positive' : ''}>{selectedRollingCandidate === name ? 'SELECTION CANDIDATE' : `${metrics.macroF1Wins}/${metrics.folds} MACRO-F1 FOLDS`}</span>
      </div>)}
      <div className="dt-rolling-folds">FOLDS: {(rollingSelection.folds ?? []).map((fold) => `${fold.fitYears[0]}–${fold.fitYears.at(-1)} → ${fold.validationYear}`).join('  ·  ')}</div>
      <p className="ov-notes">Selection uses weighted macro F1 because the labels are imbalanced. {modelName(selectedRollingCandidate)} leads all historical folds, but neither it nor any candidate is a deployable policy while Always-SAVE remains stronger on raw accuracy. The final 2026 page remains a one-time evaluation after retraining on all 2018–2025.</p>
    </section>}

    {durability?.horizons?.length > 0 && <section className="ov-panel dt-durability-panel">
      <div className="ov-panel-head"><span>PASS-DURABILITY COMPONENT / HELD-OUT 2026</span><Badge tone="derived">BENCHMARK ONLY</Badge></div>
      <p className="ov-notes">Conditional on an immediate real on-track pass, this separate model estimates whether the gained position survives each horizon. It is not an input to the action classifier or tactical replay yet.</p>
      <div className="dt-durability-head"><span>HORIZON</span><span>TEST PASSES</span><span>REAL HOLD</span><span>F1</span><span>BRIER</span><span>CAL. BRIER</span><span>REF. BRIER</span><span>AUC</span></div>
      {durability.horizons.map((item) => <div className="dt-durability-row" key={item.horizonLaps}>
        <b>{item.horizonLaps} LAP{item.horizonLaps === 1 ? '' : 'S'}</b>
        <span>{item.testRows}</span>
        <span>{item.testObservedHoldRate == null ? '—' : `${Math.round(item.testObservedHoldRate * 100)}%`}</span>
        <span>{item.f1 == null ? '—' : `${Math.round(item.f1 * 100)}%`}</span>
        <span>{item.brier == null ? '—' : item.brier.toFixed(3)}</span>
        <span>{item.calibrationBenchmark?.status === 'BENCHMARK_ONLY' ? item.calibrationBenchmark.brier.toFixed(3) : '—'}</span>
        <span>{item.constantRateBrier == null ? '—' : item.constantRateBrier.toFixed(3)}</span>
        <span>{item.rocAuc == null ? '—' : item.rocAuc.toFixed(3)}</span>
      </div>)}
      <p className="ov-notes">CAL. BRIER is the sigmoid-calibrated observational benchmark fitted before 2026. REF. BRIER is the historical constant-rate reference; AUC measures whether durable and non-durable passes are ranked apart (0.5 is chance). The component remains diagnostic until it beats the reference and is integrated without leakage.</p>
    </section>}

    {replayModels.length > 0 && <section className="ov-panel dt-persistence-panel">
      <div className="ov-panel-head"><span>HELD-OUT REPLAY / POSITION DURABILITY</span><Badge tone="real">2026 · {replayPersistence.holdout?.treeHorizonLaps ?? 6}-LAP HORIZON</Badge></div>
      <p className="ov-notes">This evaluates completed immediate on-track passes only. Each model’s predicted first action is sent through the same two-car six-lap tactical replay and compared with the real race over the same capped horizon.</p>
      <div className="dt-model-head"><span>MODEL</span><span>REAL LAPS</span><span>EST. LAPS</span><span>DELTA</span><span>BETTER</span></div>
      {replayModels.map(([name, metrics]) => <div className="dt-model-row dt-persistence-row" key={name}>
        <span>{modelName(name)}</span>
        <b>{metrics.meanObservedLeadLaps == null ? '—' : metrics.meanObservedLeadLaps.toFixed(1)}</b>
        <b>{metrics.meanModelExpectedLeadLaps == null ? '—' : metrics.meanModelExpectedLeadLaps.toFixed(1)}</b>
        <b className={metrics.meanModelVsObservedDelta > 0 ? 'positive' : ''}>{metrics.meanModelVsObservedDelta == null ? '—' : `${metrics.meanModelVsObservedDelta > 0 ? '+' : ''}${metrics.meanModelVsObservedDelta.toFixed(1)}`}</b>
        <span>{metrics.modelEstimatedBetterRate == null ? '—' : `${Math.round(metrics.modelEstimatedBetterRate * 100)}%`}</span>
      </div>)}
      <div className="dt-replay-races">
        <div className="dt-replay-race-head"><span>RACE</span><span>REAL</span><span>EST.</span><span>DELTA</span></div>
        {(replayModels[0]?.[1]?.byRace ?? []).map((race) => <div className="dt-replay-race-row" key={race.round}>
          <span>2026 {race.eventName}</span><span>{race.meanObservedLeadLaps.toFixed(1)}</span><span>{race.meanModelExpectedLeadLaps.toFixed(1)}</span><span className={race.meanModelVsObservedDelta > 0 ? 'positive' : ''}>{race.meanModelVsObservedDelta > 0 ? '+' : ''}{race.meanModelVsObservedDelta.toFixed(1)}</span>
        </div>)}
      </div>
      {primaryReplay?.byDriver?.length > 0 && <div className="dt-replay-detail">
        <div className="dt-replay-detail-title">RANDOM FOREST / BY ATTACKING DRIVER</div>
        <div className="dt-replay-driver-head"><span>DRIVER</span><span>POINTS</span><span>REAL LAPS</span><span>EST. LAPS</span><span>DELTA</span></div>
        {primaryReplay.byDriver.map((driver) => <div className="dt-replay-driver-row" key={driver.driver}>
          <b>{driver.driver}</b><span>{driver.rows}</span><span>{driver.meanObservedLeadLaps.toFixed(1)}</span><span>{driver.meanModelExpectedLeadLaps.toFixed(1)}</span><span className={driver.meanModelVsObservedDelta > 0 ? 'positive' : ''}>{driver.meanModelVsObservedDelta > 0 ? '+' : ''}{driver.meanModelVsObservedDelta.toFixed(1)}</span>
        </div>)}
      </div>}
      {primaryReplay?.byHorizon?.length > 0 && <div className="dt-replay-detail">
        <div className="dt-replay-detail-title">RANDOM FOREST / DURABILITY HORIZON</div>
        <div className="dt-replay-horizon-head"><span>HORIZON</span><span>POINTS</span><span>REAL HOLD RATE</span><span>EST. PROBABILITY</span><span>DELTA</span></div>
        {primaryReplay.byHorizon.map((item) => <div className="dt-replay-horizon-row" key={item.horizon}>
          <b>{item.horizon} LAP{item.horizon === 1 ? '' : 'S'}</b><span>{item.rows}</span><span>{Math.round(item.observedHoldRate * 100)}%</span><span>{item.meanEstimatedProbability == null ? '—' : `${Math.round(item.meanEstimatedProbability * 100)}%`}</span><span className={item.probabilityDelta > 0 ? 'positive' : ''}>{item.probabilityDelta == null ? '—' : `${item.probabilityDelta > 0 ? '+' : ''}${Math.round(item.probabilityDelta * 100)}%`}</span>
        </div>)}
      </div>}
      <p className="ov-notes">Positive delta means the estimate is above the real capped persistence. It remains a modelled estimate, not proof that the real race would have changed.</p>
    </section>}
  </div>
}

// ─── ENERGY tab ──────────────────────────────────────────────────────────────

function chartX(index, count, width, pad) {
  return pad.x + (index / Math.max(count - 1, 1)) * (width - pad.x - pad.r)
}

function EnergyMarks({ series, width, height, pad, focusLap, onFocus, yFor }) {
  const count = Math.max(series.length, 1)
  return series.map((point, index) => {
    const x = chartX(index, count, width, pad)
    const overtake = point.event === 'OVERTAKE' || point.flags?.includes('OVERTAKE')
    const chased = point.chased || point.flags?.includes('CHASED')
    const missed = point.flags?.includes('MISSED_DRS')
    const spend = point.flags?.includes('HIGH_SPEND') || point.flags?.includes('SPEND_HERE')
    if (!overtake && !chased && !missed && !spend) return null
    const fill = overtake ? '#ff7043' : spend ? '#f3c85b' : missed ? '#9db7ff' : '#63e6be'
    const y = yFor ? yFor(point, index) : pad.y + 4
    return <circle
      key={`${point.lap}-${index}`}
      className={focusLap === point.lap ? 'dt-energy-dot is-focus' : 'dt-energy-dot'}
      cx={x}
      cy={y}
      r={focusLap === point.lap ? 5 : 3.4}
      fill={fill}
      role="button"
      tabIndex={0}
      onClick={() => onFocus?.(point.lap)}
    >
      <title>{`L${point.lap} ${point.flags?.join(' · ') || point.event}`}</title>
    </circle>
  })
}

function SocChart({ laps, series = [], focusLap, onFocus }) {
  const width = 760, height = 190, pad = { x: 40, y: 18, r: 16, b: 26 }
  const window_ = 4.0
  const pts = laps.map((l, i) => {
    const x = chartX(i, laps.length, width, pad)
    const y = pad.y + (1 - Math.min(l.socEndMj, window_) / window_) * (height - pad.y - pad.b)
    return `${x},${y}`
  })
  return <div className="ov-chart">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Battery state of charge across the race">
      <g className="chart-grid">
        <line x1={pad.x} y1={pad.y} x2={width - pad.r} y2={pad.y} />
        <line x1={pad.x} y1={(height - pad.b) / 2} x2={width - pad.r} y2={(height - pad.b) / 2} />
        <line x1={pad.x} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} />
      </g>
      <polyline className="chart-primary" points={pts.join(' ')} />
      <EnergyMarks series={series} width={width} height={height} pad={pad} focusLap={focusLap} onFocus={onFocus} />
    </svg>
    <div className="chart-axis"><span>{window_} MJ</span><span>2 MJ</span><span>0 MJ</span><b>BATTERY STATE OF CHARGE / LAP</b></div>
  </div>
}

function DeployHarvestChart({ laps, series = [], focusLap, onFocus }) {
  const width = 760, height = 170, pad = { x: 40, y: 16, r: 16, b: 24 }
  const maxV = Math.max(...laps.map((l) => Math.max(l.deployMj ?? 0, l.harvestMj ?? 0)), 1) * 1.1
  const bw = (width - pad.x - pad.r) / Math.max(laps.length, 1)
  return <div className="ov-chart">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Per-lap deployment vs harvest">
      {laps.map((l, i) => {
        const x = pad.x + i * bw
        const dh = ((l.deployMj ?? 0) / maxV) * (height - pad.y - pad.b)
        const hh = ((l.harvestMj ?? 0) / maxV) * (height - pad.y - pad.b)
        const base = height - pad.b
        return <g key={l.lap}>
          <rect x={x + bw * 0.12} y={base - dh} width={bw * 0.34} height={dh} fill="#ff7043" opacity="0.9" />
          <rect x={x + bw * 0.52} y={base - hh} width={bw * 0.34} height={hh} fill="#63e6be" opacity="0.85" />
        </g>
      })}
      <EnergyMarks series={series} width={width} height={height} pad={pad} focusLap={focusLap} onFocus={onFocus} />
    </svg>
    <div className="chart-axis"><span>{maxV.toFixed(1)} MJ</span><span>—</span><span>0</span><b>DEPLOY (ORANGE) vs HARVEST (TEAL) / LAP</b></div>
  </div>
}

function StandingChart({ series, focusLap, onFocus }) {
  if (!series?.length) return null
  const width = 760, height = 168, pad = { x: 40, y: 16, r: 16, b: 24 }
  const maxP = Math.max(...series.map((point) => point.position || 1), 1)
  const pts = series.map((point, index) => {
    const x = chartX(index, series.length, width, pad)
    const y = pad.y + ((point.position - 1) / Math.max(maxP - 1, 1)) * (height - pad.y - pad.b)
    return `${x},${y}`
  })
  return <div className="ov-chart">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Running position across the race">
      <g className="chart-grid">
        <line x1={pad.x} y1={pad.y} x2={width - pad.r} y2={pad.y} />
        <line x1={pad.x} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} />
      </g>
      <polyline className="chart-secondary" points={pts.join(' ')} />
      <EnergyMarks
        series={series}
        width={width}
        height={height}
        pad={pad}
        focusLap={focusLap}
        onFocus={onFocus}
        yFor={(point) => pad.y + ((point.position - 1) / Math.max(maxP - 1, 1)) * (height - pad.y - pad.b)}
      />
    </svg>
    <div className="chart-axis"><span>P1</span><span>P{Math.round(maxP / 2)}</span><span>P{maxP}</span><b>RUNNING ORDER / LAP · P1 AT TOP</b></div>
  </div>
}

function ModelledConsumeChart({ series, focusLap, onFocus }) {
  if (!series?.length) return null
  const width = 760, height = 168, pad = { x: 40, y: 16, r: 16, b: 24 }
  const maxV = Math.max(...series.map((point) => point.consumedMj ?? 0), 0.4) * 1.15
  const bw = (width - pad.x - pad.r) / series.length
  return <div className="ov-chart">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Modelled energy used per lap">
      {series.map((point, index) => {
        const x = pad.x + index * bw
        const h = ((point.consumedMj ?? 0) / maxV) * (height - pad.y - pad.b)
        const fill = point.event === 'OVERTAKE' ? '#ff7043' : point.chased ? '#63e6be' : point.flags?.includes('MISSED_DRS') ? '#9db7ff' : '#ffb36c'
        return <rect key={point.lap} x={x + bw * 0.18} y={height - pad.b - h} width={Math.max(bw * 0.64, 1)} height={h} fill={fill} opacity={focusLap === point.lap ? 1 : 0.82} onClick={() => onFocus?.(point.lap)} />
      })}
    </svg>
    <div className="chart-axis"><span>{maxV.toFixed(2)} MJ</span><span>—</span><span>0</span><b>MODELLED DEPLOY / LAP · ORANGE TAKE · TEAL CHASED · BLUE MISSED DRS</b></div>
  </div>
}

function markSeries(series, windows = []) {
  const missed = new Set((windows || []).filter((item) => !item.converted).map((item) => item.startLap))
  const spend = new Set((windows || []).filter((item) => item.counterfactual?.verdict === 'SPEND HERE').map((item) => item.startLap))
  return (series || []).map((point) => ({
    ...point,
    flags: [
      ...(point.flags || []),
      ...(missed.has(point.lap) ? ['MISSED_DRS'] : []),
      ...(spend.has(point.lap) ? ['SPEND_HERE'] : []),
    ],
  }))
}

function EnergyBattlePanel({ sel, trend, focusLap, onFocus, onOpenSimulation }) {
  if (trend.loading) return <section className="ov-panel dt-energy-story"><div className="lx-loading"><span className="lx-spinner" />JOINING ENERGY TO 2018–2025 OVERTAKE TREND FOR {sel.driver}…</div></section>
  if (trend.error) return <section className="ov-panel dt-energy-story"><p className="lx-empty">Energy trend unavailable. {trend.error}</p></section>
  const data = trend.data
  if (!data) return null
  const series = markSeries(data.series, data.windows)
  const hist = data.trend || {}
  const best = data.bestCounterfactual
  const pct01 = (value) => value == null ? '—' : `${Math.round(Number(value) * 100)}%`
  return <section className="ov-panel dt-energy-story">
    <div className="ov-panel-head">
      <span>ENERGY × OVERTAKE / {data.driver} · {String(data.event?.name || '').toUpperCase()}</span>
      <Badge tone="derived">2018–2025 TREND · NOT A DRAW</Badge>
    </div>
    <p className="ov-notes">Orange is a pass, teal is being hunted, blue is a 1.0s window they left; energy left is our 4 MJ store model, not the team battery.</p>
    <div className="dt-story-focus">
      <div><span>PASSES</span><b>{data.summary?.overtakes ?? 0}</b><em>places they actually gained</em></div>
      <div><span>1.0s WINDOWS</span><b>{data.summary?.drsConverted}/{data.summary?.drsWindows}</b><em>passed / times they sat in DRS range</em></div>
      <div><span>HUNTED LAPS</span><b>{data.summary?.chaseLaps ?? 0}</b><em>laps with someone ≤ 1.2s behind</em></div>
      <div><span>PEAK SPEND</span><b>{mj(data.summary?.peakConsumeMj)}</b><em>most modelled deploy, lap {data.summary?.peakConsumeLap ?? '—'}</em></div>
    </div>
    {data.whatIf && <div className="dt-incident-finish dt-whatif-board">
      <div><span>REAL FINISH</span><b>{finishPos(data.whatIf.observedFinish)}</b><em>classified result from the real race</em></div>
      <div><span>WHAT-IF FINISH IF {data.whatIf.bestCall || 'CALL'} L{data.whatIf.bestLap ?? '—'}</span><b className={data.whatIf.extraPlaces > 0 ? 'positive' : ''}>{finishPos(data.whatIf.endIfCall)}</b><em>{data.whatIf.keyTake ? `key lap L${data.whatIf.keyTake.lap} vs ${data.whatIf.keyTake.ahead}` : (data.whatIf.extraPlaces ? `+${data.whatIf.extraPlaces} whole place${data.whatIf.extraPlaces === 1 ? '' : 's'} to the flag` : 'same classified finish')}</em></div>
      <div><span>CHANCE OF BEING PASSED</span><b>{data.whatIf.avoidedLose == null || Math.abs(data.whatIf.avoidedLose) < 0.02 ? '—' : `${data.whatIf.avoidedLose >= 0 ? '−' : '+'}${Math.round(Math.abs(data.whatIf.avoidedLose) * 100)}pt`}</b><em>change in the next-2-lap chance of being passed</em></div>
    </div>}
    {!!data.whatIf?.takes?.length && <p className="dt-incident-takes">{takeLine(data.whatIf.takes)}</p>}
    {data.whatIf?.note && <p className="ov-notes">{data.whatIf.note}</p>}
    {onOpenSimulation && data.whatIf?.bestLap != null && <button type="button" className="dt-incident-load dt-incident-whatif" onClick={() => onOpenSimulation({
      year: sel.year, round: sel.round, session: sessionNameOf(sel), driver: data.driver,
      lap: data.whatIf.bestLap, otherDriver: (data.incidents || []).find((item) => item.id === data.whatIf.bestIncidentId)?.otherDriver,
      call: mapSimCall(data.whatIf.bestCall), kind: data.whatIf.bestKind, theyDid: data.whatIf.theyDid,
      problem: (data.incidents || []).find((item) => item.id === data.whatIf.bestIncidentId)?.model?.resultIf
        || (data.incidents || []).find((item) => item.id === data.whatIf.bestIncidentId)?.problem,
      leftPct: (data.incidents || []).find((item) => item.id === data.whatIf.bestIncidentId)?.live?.leftPct,
      observedFinish: data.whatIf.observedFinish, raceEnd: data.whatIf.endIfCall,
      takes: data.whatIf.takes, keyTake: data.whatIf.keyTake, opponent: data.whatIf.opponent, netPass: data.whatIf.netPass,
    })}>WHAT IF {mapSimCall(data.whatIf.bestCall)} L{data.whatIf.bestLap} · SHOW ON TRACK</button>}
    <div className="dt-energy-hist">
      <div><span>{data.driver} ENERGY WHEN THEY PASSED</span><b>{hist.medianLeftPctEarlyConvert == null ? '—' : `${hist.medianLeftPctEarlyConvert}%`}</b><em>median energy left on early laps (1–15) when they actually passed, {hist.years || '2018–2025'}</em></div>
      <div><span>{data.driver} ENERGY WHEN THEY MISSED</span><b>{hist.medianLeftPctEarlyMiss == null ? '—' : `${hist.medianLeftPctEarlyMiss}%`}</b><em>median energy left when they sat inside 1.0s early and did not pass</em></div>
      <div><span>PASS RATE WITH ENERGY LEFT</span><b>{pct01(hist.highLeftConvertRate)}</b><em>how often a 1.0s window with ≥70% energy left became a pass</em></div>
      <div><span>PASS RATE WHILE HUNTED</span><b>{pct01(hist.chaseConvertRate)}</b><em>how often they passed while also covering a car behind</em></div>
    </div>
    <div className="ov-panel-head second"><span>RUNNING ORDER</span><Badge tone="real">TIMING POSITION</Badge></div>
    <StandingChart series={series} focusLap={focusLap} onFocus={onFocus} />
    <div className="ov-panel-head second"><span>WHERE ENERGY WAS USED</span><Badge tone="simulated">MODELLED ES</Badge></div>
    <ModelledConsumeChart series={series} focusLap={focusLap} onFocus={onFocus} />
    <div className="dt-energy-legend">
      <span><i style={{ background: '#ff7043' }} />TAKE</span>
      <span><i style={{ background: '#63e6be' }} />CHASED</span>
      <span><i style={{ background: '#9db7ff' }} />MISSED DRS</span>
      <span><i style={{ background: '#f3c85b' }} />SPEND-HERE (FOREST)</span>
    </div>
    {data.hotspots?.length > 0 && <div className="dt-energy-hots">
      <span>HIGHEST MODELLED DEPLOY</span>
      {data.hotspots.map((item) => <button key={item.lap} type="button" className={focusLap === item.lap ? 'active' : ''} onClick={() => onFocus(item.lap)}>
        <b>L{item.lap}</b>
        <em>{mj(item.consumedMj)}</em>
        <small>{item.event}{item.chased ? ' · CHASED' : ''}{item.inDrs ? ' · DRS' : ''}</small>
      </button>)}
    </div>}
    {data.windows?.length > 0 && <div className="dt-energy-windows">
      <span>1.0s WINDOWS · PASSED OR LEFT · WHAT IF THEY SPENT HERE</span>
      {data.windows.map((item) => {
        const cf = item.counterfactual
        return <div key={`${item.startLap}-${item.ahead}`} role="button" tabIndex={0} className={`dt-energy-cf ${item.converted ? 'took' : 'miss'} ${focusLap === item.startLap ? 'active' : ''}`} onClick={() => onFocus(item.startLap)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') onFocus(item.startLap) }}>
          <strong>L{item.startLap} vs {item.ahead} · {item.converted ? 'TOOK' : 'MISSED'}</strong>
          <em>energy left {item.leftPct == null ? '—' : `${Math.round(item.leftPct)}%`} · deploy {mj(item.consumedMj)} · held {item.waitLaps} laps{item.chased ? ' · chased' : ''}</em>
          {cf && <b>{cf.verdict}</b>}
          {cf && <p>{cf.note}</p>}
          {cf && !item.converted && <small>
            IF ATTACK finish {finishPos(cf.raceEndIfAttack ?? cf.endIfAttack)}
            {' · '}IF HOLD finish {finishPos(cf.raceEndIfHold ?? cf.endIfHold)}
            {cf.takesIfAttack?.length ? ` · ${takeLine(cf.takesIfAttack)}` : ''}
          </small>}
          {onOpenSimulation && <button type="button" className="dt-incident-load dt-incident-whatif" onClick={(event) => {
            event.stopPropagation()
            onOpenSimulation({
              year: sel.year, round: sel.round, session: sessionNameOf(sel), driver: data.driver,
              lap: item.startLap, otherDriver: item.ahead,
              call: item.converted ? 'HOLD' : mapSimCall(cf?.call || 'ATTACK'),
              kind: item.converted ? 'TOOK' : 'MISS_ATTACK',
              problem: cf?.note,
              theyDid: cf?.actualAction,
              leftPct: item.leftPct,
              gapToAheadS: item.gapToAheadS,
            })
          }}>WHAT IF {item.converted ? 'HOLD' : mapSimCall(cf?.call || 'ATTACK')} · SHOW ON TRACK</button>}
        </div>
      })}
    </div>}
    {best?.counterfactual && <p className="ov-notes">Missed 1.0s windows × {data.driver}’s 2018–2025 pass rate ≈ {data.summary?.expectedExtraPlaces ?? '—'} extra places — a rate, not a rewritten result.</p>}
  </section>
}

export function EnergyTab({ sel, energy, onOpenSimulation }) {
  const [trend, setTrend] = useState({ loading: true })
  const [focusLap, setFocusLap] = useState(null)
  useEffect(() => {
    if (!sel?.year || !sel?.round || !sel?.driver) return undefined
    let live = true
    setTrend({ loading: true })
    setFocusLap(null)
    fetchEnergyTrend({ year: sel.year, round: sel.round, session: sel.session, driver: sel.driver })
      .then((data) => { if (live) setTrend(data?.error ? { error: data.error } : { data }) })
      .catch((error) => { if (live) setTrend({ error: error.message }) })
    return () => { live = false }
  }, [sel.year, sel.round, sel.session, sel.driver])

  const marked = markSeries(trend.data?.series, trend.data?.windows)
  const d = energy.data
  const laps = d?.laps ?? []
  const g = d?.gates ?? {}
  const za = g.zoneAlignment ?? {}, ce = g.ceilings ?? {}, ct = g.crossTrackConsistency ?? {}
  const rule = d?.regulation ?? {}
  const uses2026Defaults = rule.uses2026Defaults === true
  const startSoc = laps[0]?.socStartMj ?? null
  const finishSoc = laps.at(-1)?.socEndMj ?? null
  const lowestSocLap = laps.length
    ? laps.reduce((lowest, lap) => (lap.socEndMj ?? Infinity) < (lowest.socEndMj ?? Infinity) ? lap : lowest, laps[0])
    : null
  const highestSocLap = laps.length
    ? laps.reduce((highest, lap) => (lap.socEndMj ?? -Infinity) > (highest.socEndMj ?? -Infinity) ? lap : highest, laps[0])
    : null
  const overlay = marked.length
    ? marked
    : laps.map((lap) => ({ lap: lap.lap, flags: [] }))

  return <div className="dt-energy">
    <EnergyBattlePanel sel={sel} trend={trend} focusLap={focusLap} onFocus={setFocusLap} onOpenSimulation={onOpenSimulation} />
    <section className="ov-panel">
      <div className="ov-panel-head"><span>ENERGY PROJECTION / {d?.driver || sel.driver} · {d?.year || sel.year} R{d?.round || sel.round}</span><Badge tone="simulated">{d ? (uses2026Defaults ? 'MODELLED · FIA 2026 DEFAULTS' : 'MODELLED · HISTORICAL SURROGATE') : 'LOADING TELEMETRY TRACE'}</Badge></div>
      <div className="ov-energy-list">
        <div><b>{num(ct.meanDeployMjPerLap, 2)} MJ</b><span>DEPLOY / LAP</span><strong>ERS-K OUTPUT</strong></div>
        <div><b>{num(ct.meanHarvestMjPerLap, 2)} MJ</b><span>HARVEST / LAP</span><strong>REGEN UNDER BRAKING</strong></div>
        <div><b>{num(ct.meanFuelEnergyMjPerLap, 1)} MJ</b><span>FUEL ENERGY / LAP</span><strong>ICE BURN</strong></div>
      </div>
      <div className="ov-panel-head second"><span>BATTERY STATE OF CHARGE ACROSS THE RACE</span><Badge tone="simulated">4 MJ WINDOW</Badge></div>
      <p className="ov-notes">State of charge here is a 0–4 MJ energy-store model from public telemetry, not the team’s private battery.</p>
      <div className="dt-soc-summary" aria-label="Modelled state of charge summary">
        <div><span>START SOC</span><b>{mj(startSoc)}</b><em>LAP 1 OPENING</em></div>
        <div><span>FINISH SOC</span><b>{mj(finishSoc)}</b><em>LAP {laps.at(-1)?.lap ?? '—'} END</em></div>
        <div><span>LOWEST SOC</span><b>{mj(lowestSocLap?.socEndMj)}</b><em>LAP {lowestSocLap?.lap ?? '—'} END</em></div>
        <div><span>HIGHEST SOC</span><b>{mj(highestSocLap?.socEndMj)}</b><em>LAP {highestSocLap?.lap ?? '—'} END</em></div>
      </div>
      {energy.loading && !d && <div className="lx-loading"><span className="lx-spinner" />COMPUTING 2026-REG ENERGY PROJECTION FOR {sel.driver}…</div>}
      {energy.error && !d && <p className="lx-empty">Telemetry energy projection failed for {sel.driver}: {energy.error}. The overtake-energy story above still uses the cached race.</p>}
      {laps.length > 0 && <>
      <SocChart laps={laps} series={overlay} focusLap={focusLap} onFocus={setFocusLap} />
      <div className="ov-panel-head second"><span>PER-LAP DEPLOY vs HARVEST</span><Badge tone="simulated">MODELLED</Badge></div>
      <DeployHarvestChart laps={laps} series={overlay} focusLap={focusLap} onFocus={setFocusLap} />
      <p className="ov-notes">{uses2026Defaults
        ? 'Projected from real speed/throttle/brake telemetry under published global 2026 limits — never measured team data. Competition-specific Overtake, Recharge and power-curve settings remain unloaded.'
        : 'Projected from real speed/throttle/brake telemetry with a modelled surrogate. This historical view is not presented as 2026 FIA compliance.'} Total clipping this race: {num(ct.totalClipSeconds, 1)} s. Orange / teal / blue marks on these traces are the same take / chase / missed-DRS laps as the story above.</p>
      </>}
    </section>

    <section className="ov-panel dt-gates">
      <div className="ov-panel-head"><span>VALIDATION GATES</span><Badge tone="derived">PHYSICS SANITY</Badge></div>
      {!d && <p className="ov-notes">Gates appear after the telemetry energy projection finishes. The overtake-energy story does not wait on that pass.</p>}
      {d && <>
      <div className="dt-gatecard">
        <div className="dt-gatehead"><b>01 · ZONE ALIGNMENT</b><GatePill pass={za.pass} /></div>
        <div className="dt-gaterow"><span>Deploy at full throttle</span><strong>{pct(za.deployAtFullThrottlePct, 1)}</strong></div>
        <div className="dt-gaterow"><span>Harvest above {num(za.highSpeedKph, 0)} km/h</span><strong>{pct(za.harvestAtHighSpeedPct, 1)}</strong></div>
        <p>{za.description}</p>
      </div>

      <div className="dt-gatecard">
        <div className="dt-gatehead"><b>02 · GLOBAL FIA CEILINGS</b><GatePill pass={ce.pass} /></div>
        <div className="dt-gaterow"><span>Max harvest / lap (cap {ce.harvestCapMj} MJ)</span><strong>{mj(ce.maxHarvestPerLapMj)}</strong></div>
        <div className="dt-gaterow"><span>Max SoC swing (window {ce.socWindowMj} MJ)</span><strong>{mj(ce.maxSocSwingMj)}</strong></div>
        <p>{ce.description}</p>
      </div>

      <div className="dt-gatecard">
        <div className="dt-gatehead"><b>03 · CROSS-TRACK CONSISTENCY</b><Badge tone="derived">COMPARE ACROSS EVENTS</Badge></div>
        <div className="dt-gaterow"><span>Mean deploy / lap</span><strong>{mj(ct.meanDeployMjPerLap)}</strong></div>
        <div className="dt-gaterow"><span>Mean harvest / lap</span><strong>{mj(ct.meanHarvestMjPerLap)}</strong></div>
        <div className="dt-gaterow"><span>Mean fuel energy / lap</span><strong>{mj(ct.meanFuelEnergyMjPerLap, 1)}</strong></div>
        <p>{ct.description}</p>
      </div>

      <div className="ov-panel-head second"><span>REGULATION CITATIONS</span><Badge tone="real">{uses2026Defaults ? 'FIA 2026 · ISSUE 20' : 'MODELLED / ERA PENDING'}</Badge></div>
      {rule.note && <p className="ov-notes">{rule.note}</p>}
      {Object.entries(d?.citations ?? {}).map(([k, v]) => <div className="ov-factor" key={k}>
        <span>{k.replace(/([A-Z])/g, ' $1').replace(/^./, (c) => c.toUpperCase())}</span>
        <b className="positive">{v}</b>
      </div>)}
      </>}
    </section>
  </div>
}

// ─── STRATEGY tab (fusion of energy + overtake) ──────────────────────────────

export function StrategyTab({ sel, decision, preds, energy, onSelectDriver }) {
  const dp = decision.data
  const rows = dp?.analysisRows ?? dp?.rows ?? []
  const allScored = useMemo(
    () => rows.map((r, i) => ({ ...r, pred: (preds?.analysis ?? preds)?.[i] })).filter((r) => r.pred),
    [rows, preds],
  )
  const scored = useMemo(
    () => allScored.filter((r) => r.driver === sel.driver),
    [allScored, sel.driver],
  )
  const availableBattleDrivers = useMemo(
    () => [...new Set(allScored.map((row) => row.driver).filter(Boolean))].sort(),
    [allScored],
  )
  const selectedDriverHasSourceRows = useMemo(
    () => rows.some((row) => row.driver === sel.driver),
    [rows, sel.driver],
  )
  const observedPassEvents = useMemo(() => {
    const events = new Map()
    scored.forEach((row) => {
      if (row.observedPassLap == null) return
      const previous = events.get(Number(row.observedPassLap))
      if (!previous || Number(row.lap) > Number(previous.lap)) events.set(Number(row.observedPassLap), row)
    })
    return [...events.values()].sort((left, right) => Number(left.observedPassLap) - Number(right.observedPassLap))
  }, [scored])

  // Focus on the selected driver's most attack-favourable detected point.
  const [focusLap, setFocusLap] = useState(null)
  const focus = useMemo(() => {
    if (!scored.length) return null
    return scored.find((r) => r.lap === focusLap)
      ?? scored.reduce((best, r) => (r.pred.probabilities.ATTACK > best.pred.probabilities.ATTACK ? r : best), scored[0])
  }, [scored, focusLap])

  const eLaps = energy.data?.laps ?? []
  const eLap = focus ? eLaps.find((l) => l.lap === focus.lap) ?? eLaps[eLaps.length - 1] : null
  const socAvail = eLap?.socEndMj ?? null
  const deployHeadroom = socAvail == null ? null : Math.max(0, 4.0 - socAvail)
  const affordable = socAvail == null ? null : socAvail >= 1.0
  const replayKey = focus
    ? `${sel.year}-${sel.round}-${sel.session}-${focus.lap}-${focus.driver}-${focus.defender}`
    : null
  const [replay, setReplay] = useState({ loading: false, data: null, error: null, focusKey: null })
  const [defenderEnergy, setDefenderEnergy] = useState(null)

  useEffect(() => {
    if (!focus || !allScored.length) {
      setReplay({ loading: false, data: null, error: null, focusKey: null })
      return undefined
    }
    let cancelled = false
    setReplay({ loading: true, data: null, error: null, focusKey: replayKey })
    const defenderRequest = focus.defender && focus.defender !== sel.driver
      ? fetchEnergyRace(sel.year, sel.round, sel.session, focus.defender).catch(() => null)
      : Promise.resolve(null)
    defenderRequest.then((defenderPayload) => {
      if (cancelled) return
      setDefenderEnergy(defenderPayload)
      return fetchStrategyReplay({
        focus,
        rows: allScored,
        totalLaps: dp?.totalLaps ?? 0,
        holdLaps: dp?.holdLaps ?? 6,
        finishPositions: dp?.finishPositions,
        energyLaps: {
          [sel.driver]: eLaps,
          [focus.defender]: defenderPayload?.laps ?? [],
        },
        year: sel.year,
        regulationEra: Number(sel.year) >= 2026 ? '2026' : '2018_2025',
        ruleContext: dp?.ruleContext,
      })
    }).then((data) => {
      if (!cancelled && data) setReplay({ loading: false, data, error: null, focusKey: replayKey })
    }).catch((error) => {
      if (!cancelled) setReplay({ loading: false, data: null, error, focusKey: replayKey })
    })
    return () => { cancelled = true }
  }, [focus, replayKey, allScored, eLaps, dp?.totalLaps, dp?.holdLaps, dp?.finishPositions, sel.driver, sel.year, sel.round, sel.session])

  // Never show a tree for a different focus while the backend is recomputing.
  // The backend owns the replay, its regulation configuration, and the shared
  // energy transition; the browser never substitutes an alternative tree.
  const replayPending = Boolean(replayKey && replay.focusKey !== replayKey) || replay.loading
  const tree = replay.focusKey === replayKey
    ? replay.data
    : null

  if (decision.loading) return <div className="dt-strategy">
    <StrategyRaceStory sel={sel} onSelectDriver={onSelectDriver} />
    <div className="lx-loading"><span className="lx-spinner" />BUILDING FUSED DECISION…</div>
  </div>
  if (rows.length && (preds?.loading || !preds)) return <div className="dt-strategy">
    <StrategyRaceStory sel={sel} onSelectDriver={onSelectDriver} />
    <div className="lx-loading"><span className="lx-spinner" />SCORING DETECTED BATTLES</div>
  </div>
  if (!focus) return <div className="dt-strategy">
    <StrategyRaceStory sel={sel} onSelectDriver={onSelectDriver} />
    <CachedBatteryClip sel={sel} />
    {preds?.error && selectedDriverHasSourceRows
      ? <p className="lx-empty"><b>RACE TIMING IS LOADED; MODEL SCORING DID NOT COMPLETE.</b> {sel.driver} has extracted close-battle points in this race, but the classifier request failed: {preds.error}. Retry the page; this is not a “no battle” result.</p>
      : <p className="lx-empty"><b>RACE TIMING IS LOADED.</b> No close-battle model decision point was extracted for {sel.driver} in this race, so there is no ATTACK / SAVE / DELAY recommendation to display. The observed starting position, final classification, and position movements are shown below; these are timing movements, not confirmed overtakes.</p>}
    {!selectedDriverHasSourceRows && availableBattleDrivers.length > 0 && <section className="ov-panel">
      <div className="ov-panel-head"><span>MODEL-SUPPORTED BATTLES IN THIS RACE</span><b>{availableBattleDrivers.length} DRIVERS</b></div>
      <p className="ov-notes">{sel.driver} has no extracted close battle, but this race does. Select a driver below to inspect that driver’s separate model-supported decision points. This does not assign a recommendation to {sel.driver}.</p>
      <div className="dt-lapstrip">
        {availableBattleDrivers.map((driver) => <button key={driver} type="button" className="dt-lapchip" style={{ '--c': '#63e6be' }} onClick={() => onSelectDriver?.(driver)}>
          <i>{driver}</i><span>VIEW BATTLES</span>
        </button>)}
      </div>
    </section>}
     </div>

  const p = focus.pred.probabilities
  const rec = focus.pred.label
  const energyReady = energy.data && !energy.loading
  const trainingEligible = Boolean(focus.eligibleForTraining)
  const observedOutcomeNote = focus.outcomeLabelEligible
    ? `${focus.passedNow ? 'passed on track' : 'no immediate pass'}${focus.held ? ' · held' : ''}`
    : `not used for training: ${(focus.outcomeExclusionReasons ?? []).join(', ').replaceAll('_', ' ').toLowerCase() || 'ambiguous future outcome'}`
  const trainingExclusions = (focus.exclusionReasons ?? []).join(', ').replaceAll('_', ' ').toLowerCase()

  return <div className="dt-strategy">
    <StrategyRaceStory sel={sel} onSelectDriver={onSelectDriver} />
    <CachedBatteryClip sel={sel} />
    <div className="ov-alert" style={{ borderLeftColor: STRATEGY_COLORS[rec], background: `${STRATEGY_COLORS[rec]}14` }}>
      <span style={{ color: STRATEGY_COLORS[rec] }}>◆</span>
      <div>
        <b>LAP {focus.lap} · {focus.driver} ON {focus.defender} · GAP {num(focus.gapS, 2)}s</b>
        <p>{rec === 'ATTACK' ? 'The outcome-label model rates an immediate durable pass as the strongest observed-pattern class.'
          : rec === 'DELAY' ? 'The outcome-label model rates a later durable pass as the strongest observed-pattern class.'
          : 'The outcome-label model rates preserving energy as the strongest observed-pattern class.'}</p>
      </div>
      <strong style={{ color: STRATEGY_COLORS[rec] }}>ML {rec} RECOMMENDED</strong>
    </div>

    <div className="ov-main-grid">
      <section className="ov-panel">
        <div className="ov-panel-head"><span>OVERTAKE MODEL · CLASS PROBABILITIES</span><span><Badge tone="simulated">RANDOMFOREST</Badge> <Badge tone={focus.pred.inputState?.status === 'COMPLETE' ? 'real' : 'derived'}>{focus.pred.inputState?.status === 'COMPLETE' ? 'COMPLETE INPUT' : 'PARTIAL INPUT'}</Badge></span></div>
        <div className="dt-recbig" style={{ color: STRATEGY_COLORS[rec] }}>{rec}<em>{pct(p[rec] * 100, 0)}</em></div>
        <ProbBar probabilities={p} height={12} />
        <div className="dt-problegend">
          {STRATEGY_ORDER.map((k) => <span key={k}><i style={{ background: STRATEGY_COLORS[k] }} />{k} {pct(p[k] * 100, 0)}</span>)}
        </div>
        <div className="ov-panel-head second"><span>WHAT THE MODEL SEES</span><b>FEATURES</b></div>
        <div className="ov-factor"><span>Gap to car ahead <Badge tone="derived">DERIVED</Badge></span><b className={focus.gapS <= 1 ? 'positive' : 'negative'}>{num(focus.gapS, 3)} s</b><em>{focus.gapS <= dp.maxGapThresholdS ? 'inside analysis window' : 'outside analysis window'}</em></div>
        <div className="ov-factor"><span>Speed-trap delta <Badge tone="real">REAL</Badge></span><b className={focus.speedDeltaKph > 0 ? 'positive' : 'negative'}>{focus.speedDeltaKph == null ? '—' : `${focus.speedDeltaKph > 0 ? '+' : ''}${focus.speedDeltaKph} km/h`}</b><em>{focus.driver || 'attacker'} vs {focus.defender || 'defender'}</em></div>
        <div className="ov-factor"><span>Closing rate <Badge tone="derived">DERIVED</Badge></span><b>{focus.closingRateS == null ? '—' : `${focus.closingRateS > 0 ? '+' : ''}${num(focus.closingRateS, 2)} s/lap`}</b><em>positive = gap closing</em></div>
        <div className="ov-factor"><span>Tyre age differential <Badge tone="real">REAL</Badge></span><b>{num(focus.tyreAgeDiff, 0)} laps</b><em>{focus.attackerCompound} vs {focus.defenderCompound}</em></div>
        <div className="ov-factor"><span>Car mass <Badge tone="simulated">MODELLED</Badge></span><b>{focus.pred.mass ? `${focus.pred.mass.attackerKg} vs ${focus.pred.mass.defenderKg} kg` : '—'}</b><em>Δ {focus.pred.mass?.deltaKg ?? 0} kg · reg floor + tyres + 82 kg driver + fuel burn</em></div>
        {focus.pred.inputState?.usesNeutralFallback && <div className="ov-factor"><span>Inference completeness <Badge tone="derived">PARTIAL</Badge></span><b className="negative">CAUTION</b><em>neutral fallback for {focus.pred.inputState.missingImportantInputs.join(', ')}</em></div>}
        {focus.pred.liveCommandGate && <div className="ov-factor"><span>Live Overtake Mode gate <Badge tone={focus.pred.liveCommandGate.liveCommandEligible ? 'real' : 'derived'}>{focus.pred.liveCommandGate.liveCommandEligible ? 'READY' : 'ANALYSIS ONLY'}</Badge></span><b className={focus.pred.liveCommandGate.liveCommandEligible ? 'positive' : 'negative'}>{focus.pred.liveCommandGate.liveCommandEligible ? 'LIVE-COMMAND READY' : 'NOT A LIVE COMMAND'}</b><em>{focus.pred.liveCommandGate.liveCommandEligible ? focus.pred.liveCommandGate.note : focus.pred.liveCommandGate.blockedBy.join(' · ')}</em></div>}
        <div className="ov-factor"><span>Observed outcome <Badge tone={trainingEligible ? 'real' : 'derived'}>{trainingEligible ? 'TRAINING-ELIGIBLE' : 'OBSERVED / NOT TRAINED'}</Badge></span><b style={{ color: focus.outcomeLabelEligible ? STRATEGY_COLORS[focus.label] : '#ffb36c' }}>{focus.outcomeLabelEligible ? focus.label : 'NOT SCORED'}</b><em>{observedOutcomeNote}{!trainingEligible && focus.outcomeLabelEligible && trainingExclusions ? ` · input excluded: ${trainingExclusions}` : ''}</em></div>
      </section>

      <aside className="ov-panel">
        <div className="ov-panel-head"><span>ENERGY BUDGET AT LAP {focus.lap}</span><Badge tone="simulated">{energyReady ? 'MODELLED' : 'LOADING'}</Badge></div>
        {energyReady && eLap ? <>
          <div className="ov-big-metric">
            <span>BATTERY AVAILABLE</span>
            <strong style={{ color: affordable ? '#63e6be' : '#ff9b78' }}>{mj(socAvail)}</strong>
            <p>of a 4.0 MJ usable window · {eLap.compound ?? 'TYRE N/A'}{eLap.pit ? ' · PIT LAP' : ''}</p>
          </div>
          <div className="ov-energy-rows">
            <div><span>DEPLOY HEADROOM</span><b>{mj(deployHeadroom)}</b><em>BELOW WINDOW CAP</em></div>
            <div><span>THIS LAP DEPLOY</span><b>{mj(eLap.deployMj)}</b><em>ERS-K OUT</em></div>
            <div><span>THIS LAP HARVEST</span><b>{mj(eLap.harvestMj)}</b><em>REGEN IN</em></div>
            <div><span>CLIPPING</span><b>{num(eLap.clipSeconds, 1)} s</b><em>POWER LIMITED</em></div>
          </div>
          <div className={`ov-assumption ${affordable ? 'ok' : 'tight'}`}>
            {affordable ? 'ENERGY AFFORDABLE' : 'ENERGY TIGHT'}
            <p>{affordable
              ? `The battery holds ${mj(socAvail)} entering lap ${focus.lap}, enough to fund an ${rec} without breaching the 4 MJ window.`
              : `Only ${mj(socAvail)} available — an attack would clip. Favour SAVE/DELAY and harvest first.`}</p>
          </div>
        </> : <div className="lx-loading"><span className="lx-spinner" />LOADING ENERGY FOR {sel.driver}…</div>}
        <div className="ov-panel-head second"><span>FUSED VERDICT</span><Badge tone="derived">ENERGY × OVERTAKE</Badge></div>
        <p className="ov-notes">
          {rec === 'ATTACK' && affordable
            ? 'Outcome-pattern estimate: ATTACK is highest and the modelled energy state can fund it. This remains analysis unless the FIA live-command gate is READY.'
            : rec === 'ATTACK' && !affordable
            ? 'Outcome-pattern estimate: ATTACK is highest, but the modelled energy state is tight. This is not a command to deploy.'
            : rec === 'DELAY'
            ? 'Outcome-pattern estimate: DELAY is highest. It represents a later durable-pass pattern, not a known future instruction.'
            : 'Outcome-pattern estimate: SAVE is highest. It is not evidence that a real driver was instructed to save energy.'}
        </p>
        {defenderEnergy?.pitDoesNotRechargeEnergy === true && <p className="ov-notes">Defender energy is loaded from the same modelled race trace; pit context changes tyre/time state and does not reset SoC.</p>}
      </aside>
    </div>

   
    <div className="ov-strategy-row">
      <div className="ov-section-label"><span>{sel.driver} BATTLES + OBSERVED PASSES</span><b>SELECT A LAP TO INSPECT</b></div>
      <div className="dt-lapstrip">
        {scored.map((r) => <button
          key={`${r.lap}-${r.defender}`}
          className={`dt-lapchip ${focus.lap === r.lap ? 'active' : ''}`}
          style={{ '--c': STRATEGY_COLORS[r.pred.label] }}
          onClick={() => setFocusLap(r.lap)}
          title={`L${r.lap} vs ${r.defender} · ${r.pred.label} ${(r.pred.probabilities[r.pred.label] * 100).toFixed(0)}%${r.outcomeLabelEligible ? '' : ' · observed only, not used for training'}`}
        >L{r.lap}<i>{r.pred.label[0]}</i></button>)}
        {observedPassEvents.map((r) => <button
          key={`observed-pass-${r.observedPassLap}`}
          className={`dt-lapchip dt-lapchip-event ${focus.observedPassLap === r.observedPassLap ? 'active' : ''}`}
          style={{ '--c': '#63e6be' }}
          onClick={() => setFocusLap(r.lap)}
          title={`Observed pass on lap ${r.observedPassLap}; inspect decision from lap ${r.lap}`}
        >L{r.observedPassLap}<i>P</i></button>)}
      </div>
    </div>

    {replayPending && <div className="lx-loading dt-tree-loading"><span className="lx-spinner" />RECOMPUTING TREE FOR LAP {focus.lap}…</div>}
    {!replayPending && replay.focusKey === replayKey && replay.error && <div className="ov-assumption tight dt-tree-loading">
      REPLAY UNAVAILABLE
      <p>The browser will not substitute a local tactical calculation. Start the backend replay service and retry this decision point.</p>
    </div>}
    <StrategyTreePanel tree={tree ? { ...tree, classifierAction: rec } : tree} />
  </div>
}
