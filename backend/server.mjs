import http from 'node:http'
import { spawn } from 'node:child_process'
import { readFile, readdir, mkdir, writeFile, rename, rm } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { detectMoodFromText, detectMoodFromAudio } from './src/mood.mjs'
import { extractEngineerKeywords, primaryKeyword } from './src/keywords.mjs'
import { transcribeAudio } from './src/transcription.mjs'
import { buildTurnMarkers, resolveTrackContext } from './src/track-context.mjs'
import {
  createHistorySession,
  finishHistorySession,
  getHistorySession,
  historyStorageStatus,
  listHistorySessions,
  logHistoryEvent,
  saveHistoryTelemetry,
} from './src/database.mjs'

const root = path.dirname(fileURLToPath(import.meta.url))
const examples = JSON.parse(await readFile(path.join(root, 'data/hf-slice.json'), 'utf8'))
const nightRaceData = JSON.parse(await readFile(path.join(root, 'data/openf1-2023-night-races.json'), 'utf8'))
const PORT = Number(process.env.PORT || 8787)
const HF_MODEL = 'facebook/bart-large-mnli'

const driverLabels = ['rear slip', 'front grip loss', 'radio failure', 'rain report', 'race control', 'blue flag', 'pit request', 'other']
const engineerLabels = ['reduce curb', 'pit instruction', 'race control', 'blue flag', 'radio check', 'boost instruction', 'tyre management', 'other']

// ─── Helpers ──────────────────────────────────────────────────────────────────

function json(response, status, payload) {
  response.writeHead(status, {
    'content-type': 'application/json',
    'access-control-allow-origin': '*',
    'access-control-allow-headers': 'content-type, authorization',
    'access-control-allow-methods': 'POST, GET, OPTIONS',
  })
  response.end(JSON.stringify(payload))
}

function words(value) {
  return new Set(
    value.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter((word) => word.length > 2),
  )
}

function similarity(left, right) {
  const a = words(left)
  const b = words(right)
  const overlap = [...a].filter((word) => b.has(word)).length
  return overlap / Math.max(1, Math.sqrt(a.size * b.size))
}

function bestExample(message, direction) {
  return examples
    .filter((example) => example.direction === direction)
    .map((example) => ({ example, score: similarity(message, example.utterance) }))
    .sort((a, b) => b.score - a.score)[0]
}

function turnFromMessage(message) {
  const match = message.match(/turn\s*(\d{1,2})/i)
  return match ? `T${match[1]}` : ''
}

function accessTokenFrom(request) {
  const authorization = request.headers.authorization || ''
  const match = authorization.match(/^Bearer\s+(.+)$/i)
  if (!match) {
    const error = new Error('A signed-in Supabase user is required for history storage.')
    error.statusCode = 401
    throw error
  }
  return match[1]
}

// The Copilot must never pretend to make a race-strategy decision. It gives the
// driver a short, explainable acknowledgement / setup prompt and leaves a human
// engineer free to override it. The issue itself comes from HF retrieval or the
// zero-shot classifier above; this layer constrains that result to the small
// driver-display vocabulary used by the F1-style screen.
function autoEngineerResponse(issue, message, mood) {
  const turn = turnFromMessage(message)
  const atTurn = turn ? ` at ${turn}` : ''
  const displayTurn = turn ? ` ${turn}` : ''

  const responses = {
    'REAR SLIP': {
      reply: `Copy. Rear slip${atTurn}. Short-shift and reduce exit throttle.`,
      display: `SHORT SHIFT${displayTurn}`,
      action: 'Short-shift; smooth the throttle on exit.',
    },
    'FRONT GRIP': {
      reply: `Copy. Front grip loss${atTurn}. Avoid the kerb and manage the entry.`,
      display: `MANAGE ENTRY${displayTurn}`,
      action: 'Avoid the kerb; protect front grip into the corner.',
    },
    'TYRE / WHEEL': {
      reply: `Copy. Tyre or wheel concern${atTurn}. Confirm front or rear, then describe the grip change.`,
      display: `TYRE CHECK${displayTurn}`,
      action: 'Confirm whether the issue is at the front or rear before changing setup.',
    },
    'CAR BALANCE': {
      reply: `Copy. Balance issue${atTurn}. Confirm whether it is front or rear limited.`,
      display: `BALANCE CHECK${displayTurn}`,
      action: 'Confirm the affected axle and corner before a manual engineer response.',
    },
    'BRAKING': {
      reply: `Copy. Brake issue${atTurn}. Brake earlier and keep the pedal release smooth.`,
      display: `BRAKE EARLY${displayTurn}`,
      action: 'Brake earlier and release progressively.',
    },
    'GENERAL COMPLAINT': {
      reply: 'Copy. State the car system, the corner, and whether it is getting worse.',
      display: 'REPORT ISSUE',
      action: 'State the system, corner, and severity.',
    },
    'RADIO FAILURE': {
      reply: 'Copy. Radio check. Repeat only the critical car issue.',
      display: 'RADIO CHECK',
      action: 'Use short repeat-back messages until signal is clear.',
    },
    'RAIN REPORT': {
      reply: `Copy. Wet-condition report${atTurn}. Keep us updated on grip and standing water.`,
      display: `REPORT GRIP${displayTurn}`,
      action: 'Continue reporting grip changes and standing water.',
    },
    'RACE CONTROL': {
      reply: 'Copy. Race-control situation acknowledged. Follow the delta and wait for the next call.',
      display: 'HOLD DELTA',
      action: 'Follow the delta; await the next pit-wall instruction.',
    },
    'BLUE FLAG': {
      reply: 'Copy. Blue flag acknowledged. Give the car ahead a clean pass at the next safe point.',
      display: 'BLUE FLAG',
      action: 'Yield safely at the next appropriate point.',
    },
    'PIT REQUEST': {
      reply: 'Copy. Pit request received. We are checking the window; stay on the current plan.',
      display: 'STAY ON PLAN',
      action: 'Await manual pit-wall confirmation before changing strategy.',
    },
  }

  if (responses[issue]) return { ...responses[issue], source: 'issue-aware-auto-response' }

  if (mood === 'ANGRY') return { reply: 'Copy. We hear you. Give us the car issue and corner.', display: 'REPORT ISSUE', action: 'State the issue and the affected corner.', source: 'mood-safe-response' }
  if (mood === 'FRUSTRATED') return { reply: 'Copy. Keep the message short: issue, corner, then severity.', display: 'ISSUE / CORNER', action: 'Report the issue, corner, and severity.', source: 'mood-safe-response' }
  if (mood === 'URGENT') return { reply: 'Understood. Priority channel open — state the critical issue now.', display: 'PRIORITY RADIO', action: 'Use the radio for the critical issue only.', source: 'mood-safe-response' }
  return { reply: 'Copy. State the car issue and the affected corner.', display: 'REPORT ISSUE', action: 'State the issue and affected corner.', source: 'safe-default-response' }
}

function replayComparison(race) {
  const current = race.comparison.current
  const driverLaps = race.laps.filter((lap) => lap.driver_number === race.selected_driver.driver_number
    && Number.isFinite(lap.duration) && lap.duration > 50 && !lap.pit_out_lap
    && Number.isFinite(lap.sector_1) && Number.isFinite(lap.sector_2) && Number.isFinite(lap.sector_3))
  const reference = driverLaps
    .filter((lap) => lap.lap_number !== current.lap_number)
    .sort((left, right) => Math.abs(left.duration - current.duration) - Math.abs(right.duration - current.duration)
      || Math.abs(left.lap_number - current.lap_number) - Math.abs(right.lap_number - current.lap_number))[0]
  return {
    current,
    reference: reference || race.comparison.reference,
    delta_seconds: Number((current.duration - (reference || race.comparison.reference).duration).toFixed(3)),
    selection_rule: 'Fastest clean lap compared with the nearest valid lap-time reference.',
  }
}

// ─── Deterministic driver analysis ────────────────────────────────────────────

function deterministicDriverAnalysis(message) {
  const text = message.toLowerCase()
  const turn = turnFromMessage(message)
  if (/rear|slid|throttle|traction/.test(text)) return { issue: 'REAR SLIP', keyword: `REAR SLIP${turn ? ` ${turn}` : ''}`, confidence: 0.92 }
  if (/front|understeer/.test(text)) return { issue: 'FRONT GRIP', keyword: `FRONT GRIP${turn ? ` ${turn}` : ''}`, confidence: 0.88 }
  if (/wheel|tyre|tire/.test(text)) return { issue: 'TYRE / WHEEL', keyword: `TYRE CHECK${turn ? ` ${turn}` : ''}`, confidence: 0.76 }
  if (/car|balance|handling|unstable/.test(text)) return { issue: 'CAR BALANCE', keyword: `BALANCE CHECK${turn ? ` ${turn}` : ''}`, confidence: 0.66 }
  if (/hear|radio|mic|microphone/.test(text)) return { issue: 'RADIO FAILURE', keyword: 'RADIO FAIL', confidence: 0.96 }
  if (/safety car/.test(text)) return { issue: 'RACE CONTROL', keyword: 'SAFETY CAR', confidence: 0.97 }
  if (/blue flag/.test(text)) return { issue: 'BLUE FLAG', keyword: 'BLUE FLAG', confidence: 0.97 }
  if (/box|pit/.test(text)) return { issue: 'PIT REQUEST', keyword: 'BOX', confidence: 0.94 }
  if (/rain|wet|damp/.test(text)) return { issue: 'RAIN REPORT', keyword: `RAIN${turn ? ` ${turn}` : ''}`, confidence: 0.89 }
  return { issue: 'UNCLASSIFIED', keyword: 'REVIEW RADIO', confidence: 0.54 }
}

// ─── Hugging Face zero-shot classifier ────────────────────────────────────────

async function huggingFaceLabel(message, direction) {
  const token = process.env.HF_API_TOKEN
  if (!token) return null
  const labels = direction === 'driver_to_engineer' ? driverLabels : engineerLabels
  try {
    const response = await fetch(`https://api-inference.huggingface.co/models/${HF_MODEL}`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
      body: JSON.stringify({ inputs: message, parameters: { candidate_labels: labels, multi_label: false } }),
    })
    if (!response.ok) return null
    const result = await response.json()
    return result?.labels?.[0] || null
  } catch {
    return null
  }
}

// ─── Analyse: Driver → Engineer ───────────────────────────────────────────────

async function analyseDriver(message, team, audioFeatures, trackContext = null) {
  const match = bestExample(message, 'driver_to_engineer')
  const fallback = deterministicDriverAnalysis(message)
  const hfLabel = match?.score >= 0.28 ? null : await huggingFaceLabel(message, 'driver_to_engineer')
  const label = hfLabel || (match?.score >= 0.28 ? match.example.intent?.replaceAll('_', ' ') : null)

  // Mood detection — use audio features if provided, else text-only
  const moodResult = audioFeatures
    ? detectMoodFromAudio(message, audioFeatures)
    : detectMoodFromText(message)

  const result = {
    state: moodResult.mood,
    mood: moodResult.mood,
    moodConfidence: moodResult.moodConfidence,
    moodReason: moodResult.moodReason,
    ...fallback,
    direction: 'driver_to_engineer',
    team: team || null,
    original: message,
    trackContext,
    matchedExample: match?.example?.utterance || null,
    retrievalScore: Number((match?.score || 0).toFixed(2)),
    provider: hfLabel
      ? `huggingface:${HF_MODEL}`
      : match?.score >= 0.28
      ? 'hub-dataset-retrieval'
      : 'safe-local-fallback',
  }
  // Override issue/keyword if HF/retrieval gave us a strong label
  if (label && /rear slip/.test(label)) {
    const turn = turnFromMessage(message)
    result.issue = 'REAR SLIP'
    result.keyword = `REAR SLIP${turn ? ` ${turn}` : ''}`
  }
  if (label && /front grip/.test(label)) {
    const turn = turnFromMessage(message)
    result.issue = 'FRONT GRIP'
    result.keyword = `FRONT GRIP${turn ? ` ${turn}` : ''}`
  }

  const autoResponse = autoEngineerResponse(result.issue, message, result.mood)
  result.engineerReply = autoResponse.reply
  result.driverDisplay = autoResponse.display
  result.recommendedAction = autoResponse.action
  result.responseMode = 'AUTO COPILOT — HUMAN OVERRIDE AVAILABLE'
  result.responseSource = autoResponse.source

  return result
}

// ─── Analyse: Engineer → Driver ───────────────────────────────────────────────

async function analyseEngineer(message, team) {
  const match = bestExample(message, 'engineer_to_driver')
  const hfLabel = match?.score >= 0.28 ? null : await huggingFaceLabel(message, 'engineer_to_driver')

  // Multi-keyword extraction is the primary feature here
  const keywords = extractEngineerKeywords(message)
  const keyword = keywords[0] || primaryKeyword(message)

  return {
    state: 'INSTRUCTION',
    keyword,
    keywords,          // full array for sequential display on steering wheel
    direction: 'engineer_to_driver',
    team: team || null,
    original: message,
    matchedExample: match?.example?.utterance || null,
    retrievalScore: Number((match?.score || 0).toFixed(2)),
    provider: hfLabel
      ? `huggingface:${HF_MODEL}`
      : match?.score >= 0.28
      ? 'hub-dataset-retrieval'
      : 'safe-local-fallback',
  }
}

// ─── FastF1 bridge ────────────────────────────────────────────────────────────
// Session and telemetry payloads are produced by the Python scripts in
// backend/scripts (fastf1) and cached as JSON under data/f1-cache so repeated
// requests and the prefetch script share the same files.

const F1_CACHE_DIR = path.join(root, 'data', 'f1-cache')

function f1Slug(sessionName) {
  return sessionName.toLowerCase().replace(/\s+/g, '_')
}

function runPython(script, args, timeoutMs) {
  return new Promise((resolve, reject) => {
    const child = spawn('python', [path.join(root, 'scripts', script), ...args])
    let stdout = ''
    let stderr = ''
    const timer = setTimeout(() => {
      child.kill('SIGKILL')
      reject(new Error(`${script} timed out after ${Math.round(timeoutMs / 1000)}s`))
    }, timeoutMs)
    child.stdout.on('data', (chunk) => { stdout += chunk })
    child.stderr.on('data', (chunk) => { stderr += chunk })
    child.on('error', (error) => { clearTimeout(timer); reject(error) })
    child.on('close', (code) => {
      clearTimeout(timer)
      if (code === 0) return resolve(stdout)
      const detail = stderr.trim().split('\n').filter(Boolean).pop() || `exit ${code}`
      reject(new Error(`${script} failed: ${detail}`))
    })
  })
}

function runPythonWithInput(script, args, stdinData, timeoutMs) {
  return new Promise((resolve, reject) => {
    const child = spawn('python', [path.join(root, 'scripts', script), ...args])
    let stdout = ''
    let stderr = ''
    const timer = setTimeout(() => {
      child.kill('SIGKILL')
      reject(new Error(`${script} timed out after ${Math.round(timeoutMs / 1000)}s`))
    }, timeoutMs)
    child.stdout.on('data', (chunk) => { stdout += chunk })
    child.stderr.on('data', (chunk) => { stderr += chunk })
    child.on('error', (error) => { clearTimeout(timer); reject(error) })
    child.on('close', (code) => {
      clearTimeout(timer)
      if (code === 0) return resolve(stdout)
      const detail = stderr.trim().split('\n').filter(Boolean).pop() || `exit ${code}`
      reject(new Error(`${script} failed: ${detail}`))
    })
    child.stdin.on('error', () => {})
    child.stdin.end(stdinData)
  })
}

const ONDEMAND_LOCK = path.join(F1_CACHE_DIR, '.ondemand.lock')
let onDemandInFlight = 0
const f1FetchInFlight = new Map()

async function f1CachedOrFetch(cacheRel, script, args, timeoutMs) {
  const cachePath = path.join(F1_CACHE_DIR, cacheRel)
  try {
    return JSON.parse(await readFile(cachePath, 'utf8'))
  } catch {
    // cache miss: fetch below
  }
  const existing = f1FetchInFlight.get(cachePath)
  if (existing) return existing
  const pending = (async () => {
    onDemandInFlight += 1
    if (onDemandInFlight === 1) await writeFile(ONDEMAND_LOCK, String(process.pid)).catch(() => {})
    try {
      const payload = JSON.parse(await runPython(script, args, timeoutMs))
      await mkdir(path.dirname(cachePath), { recursive: true })
      const tmpPath = `${cachePath}.tmp`
      await writeFile(tmpPath, JSON.stringify(payload))
      await rename(tmpPath, cachePath)
      return payload
    } finally {
      onDemandInFlight -= 1
      if (onDemandInFlight === 0) await rm(ONDEMAND_LOCK, { force: true }).catch(() => {})
    }
  })()
  f1FetchInFlight.set(cachePath, pending)
  try {
    return await pending
  } finally {
    f1FetchInFlight.delete(cachePath)
  }
}

// ─── HTTP body helpers ─────────────────────────────────────────────────────────

async function readJsonBody(request) {
  let raw = ''
  for await (const chunk of request) raw += chunk
  return JSON.parse(raw || '{}')
}

async function readRawBody(request) {
  const chunks = []
  for await (const chunk of request) chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk)
  return Buffer.concat(chunks)
}

// ─── Server ───────────────────────────────────────────────────────────────────

export async function handler(request, response) {
  if (request.method === 'OPTIONS') return json(response, 204, {})

  // GET /api/health
  if (request.method === 'GET' && request.url === '/api/health') {
    return json(response, 200, {
      ok: true,
      examples: examples.length,
      model: HF_MODEL,
      whisperModel: 'openai/whisper-large-v3',
      features: ['mood-detection', 'multi-keyword', 'voice-transcription', 'auto-engineer-reply', 'session-history'],
      history: historyStorageStatus(),
    })
  }

  // GET /api/replay/circuits — available real historical replay sources.
  if (request.method === 'GET' && request.url === '/api/replay/circuits') {
    return json(response, 200, Object.entries(nightRaceData.races).map(([id, race]) => ({
      id,
      circuit: race.session.circuit_short_name,
      country: race.session.country_name,
      year: race.session.year,
    })))
  }

  // GET /api/replay?circuit=bahrain
  // Returns a compact payload for rendering real lap times and the circuit map.
  if (request.method === 'GET' && request.url?.startsWith('/api/replay')) {
    const circuit = new URL(request.url, 'http://localhost').searchParams.get('circuit') || 'bahrain'
    const race = nightRaceData.races[circuit.toLowerCase()]
    if (!race) return json(response, 404, { error: 'circuit not loaded', available: Object.keys(nightRaceData.races) })

    const trackPosition = Array.isArray(race.track_position)
      ? race.track_position
      : race.track_position.current
    const carData = Array.isArray(race.car_data)
      ? race.car_data
      : race.car_data.current

    return json(response, 200, {
      source: nightRaceData.source,
      session: race.session,
      selected_driver: race.selected_driver,
      comparison: replayComparison(race),
      track_position: trackPosition,
      car_data: carData,
      turn_markers: buildTurnMarkers(circuit.toLowerCase(), race),
      weather: race.weather,
      radio_clips: race.radio_clips,
      laps: race.laps.filter((lap) => lap.driver_number === race.selected_driver.driver_number),
    })
  }

  // POST /api/analyse/driver
  if (request.method === 'POST' && request.url === '/api/analyse/driver') {
    try {
      const input = await readJsonBody(request)
      if (!input.message || typeof input.message !== 'string') {
        return json(response, 400, { error: 'message is required' })
      }
      // Optional audio features (rms, pitch) sent from browser audio analysis
      const audioFeatures = input.audioFeatures || null
      const circuitId = String(input.circuit || '').toLowerCase()
      const race = nightRaceData.races[circuitId]
      const trackContext = race ? resolveTrackContext(circuitId, race, input.lapProgress) : null
      return json(response, 200, await analyseDriver(input.message.trim(), input.team, audioFeatures, trackContext))
    } catch (error) {
      return json(response, 400, { error: error.message || 'invalid request' })
    }
  }

  // POST /api/analyse/engineer
  if (request.method === 'POST' && request.url === '/api/analyse/engineer') {
    try {
      const input = await readJsonBody(request)
      if (!input.message || typeof input.message !== 'string') {
        return json(response, 400, { error: 'message is required' })
      }
      return json(response, 200, await analyseEngineer(input.message.trim(), input.team))
    } catch (error) {
      return json(response, 400, { error: error.message || 'invalid request' })
    }
  }

  // POST /api/transcribe
  // Accepts raw audio blob (multipart or raw binary) and returns transcription + mood
  if (request.method === 'POST' && request.url?.startsWith('/api/transcribe')) {
    try {
      const contentType = request.headers['content-type'] || 'audio/webm'
      const audioBuffer = await readRawBody(request)

      if (!audioBuffer.length) {
        return json(response, 400, { error: 'audio data is required' })
      }

      // Try Whisper transcription
      let transcription
      try {
        transcription = await transcribeAudio(audioBuffer, contentType.split(';')[0].trim())
      } catch (err) {
        return json(response, 503, { error: `Transcription failed: ${err.message}` })
      }

      // Optionally run analysis on the transcribed text
      const direction = request.url.includes('engineer') ? 'engineer_to_driver' : 'driver_to_engineer'
      const team = new URL(request.url, 'http://localhost').searchParams.get('team') || null

      let analysis = {}
      if (direction === 'driver_to_engineer') {
        analysis = await analyseDriver(transcription.text, team, null)
      } else {
        analysis = await analyseEngineer(transcription.text, team)
      }

      return json(response, 200, {
        transcription: transcription.text,
        whisperModel: transcription.model,
        ...analysis,
      })
    } catch (error) {
      return json(response, 400, { error: error.message || 'transcription request failed' })
    }
  }

  // ─── Persistent history ──────────────────────────────────────────────────
  // Every request carries the signed-in user's Supabase token. database.mjs
  // forwards that token to Postgres, so the team RLS rules apply to every query.
  if (request.method === 'POST' && request.url === '/api/history/start-session') {
    try {
      const input = await readJsonBody(request)
      const session = await createHistorySession(input, accessTokenFrom(request))
      return json(response, 201, { session })
    } catch (error) {
      return json(response, error.statusCode || 400, { error: error.message || 'could not start history session' })
    }
  }

  if (request.method === 'POST' && request.url === '/api/history/log-event') {
    try {
      const input = await readJsonBody(request)
      const event = await logHistoryEvent(input, accessTokenFrom(request))
      return json(response, 201, { event })
    } catch (error) {
      return json(response, error.statusCode || 400, { error: error.message || 'could not save history event' })
    }
  }

  if (request.method === 'POST' && request.url === '/api/history/log-telemetry') {
    try {
      const input = await readJsonBody(request)
      const telemetry = await saveHistoryTelemetry(input, accessTokenFrom(request))
      return json(response, 201, { telemetry })
    } catch (error) {
      return json(response, error.statusCode || 400, { error: error.message || 'could not save telemetry snapshot' })
    }
  }

  if (request.method === 'POST' && request.url === '/api/history/end-session') {
    try {
      const input = await readJsonBody(request)
      const session = await finishHistorySession(input.sessionId, input.status, accessTokenFrom(request))
      return json(response, 200, { session })
    } catch (error) {
      return json(response, error.statusCode || 400, { error: error.message || 'could not end history session' })
    }
  }

  if (request.method === 'GET' && request.url?.startsWith('/api/history/sessions/')) {
    try {
      const id = decodeURIComponent(new URL(request.url, 'http://localhost').pathname.split('/').pop())
      return json(response, 200, await getHistorySession(id, accessTokenFrom(request)))
    } catch (error) {
      return json(response, error.statusCode || 404, { error: error.message || 'history session not found' })
    }
  }

  if (request.method === 'GET' && request.url?.startsWith('/api/history/sessions')) {
    try {
      const team = new URL(request.url, 'http://localhost').searchParams.get('team')
      return json(response, 200, { sessions: await listHistorySessions(team, accessTokenFrom(request)) })
    } catch (error) {
      return json(response, error.statusCode || 400, { error: error.message || 'could not load history sessions' })
    }
  }

  // ─── FastF1 lap chart + telemetry data ──────────────────────────────────
  // GET /api/f1/events?year=2024 — rounds and available sessions for a year.
  if (request.method === 'GET' && request.url?.startsWith('/api/f1/events')) {
    const params = new URL(request.url, 'http://localhost').searchParams
    const year = params.get('year') || ''
    if (!/^\d{4}$/.test(year)) return json(response, 400, { error: 'a valid year is required' })
    try {
      return json(response, 200, await f1CachedOrFetch(`events/${year}.json`, 'fetch_f1_events.py', ['--year', year], 120000))
    } catch (error) {
      return json(response, 502, { error: error.message })
    }
  }

  // GET /api/f1/session?year&round&session — drivers + all laps for a session.
  if (request.method === 'GET' && request.url?.startsWith('/api/f1/session')) {
    const params = new URL(request.url, 'http://localhost').searchParams
    const year = params.get('year') || ''
    const round = params.get('round') || ''
    const session = params.get('session') || ''
    if (!/^\d{4}$/.test(year) || !/^\d{1,2}$/.test(round) || !session) {
      return json(response, 400, { error: 'year, round and session are required' })
    }
    try {
      const cacheRel = `sessions/${year}/${round}_${f1Slug(session)}.json`
      return json(response, 200, await f1CachedOrFetch(cacheRel, 'fetch_f1_session.py', ['--year', year, '--round', round, '--session', session], 240000))
    } catch (error) {
      return json(response, 502, { error: error.message })
    }
  }

  // GET /api/f1/telemetry?year&round&session&driver&lap — one lap's car data.
  if (request.method === 'GET' && request.url?.startsWith('/api/f1/telemetry')) {
    const params = new URL(request.url, 'http://localhost').searchParams
    const year = params.get('year') || ''
    const round = params.get('round') || ''
    const session = params.get('session') || ''
    const driver = (params.get('driver') || '').toUpperCase()
    const lap = params.get('lap') || ''
    if (!/^\d{4}$/.test(year) || !/^\d{1,2}$/.test(round) || !session || !/^[A-Z]{3}$/.test(driver) || !/^\d{1,3}$/.test(lap)) {
      return json(response, 400, { error: 'year, round, session, driver and lap are required' })
    }
    try {
      const cacheRel = `telemetry/${year}/${round}_${f1Slug(session)}/${driver}_${lap}.json`
      return json(response, 200, await f1CachedOrFetch(cacheRel, 'fetch_f1_telemetry.py', ['--year', year, '--round', round, '--session', session, '--driver', driver, '--lap', lap], 300000))
    } catch (error) {
      return json(response, 502, { error: error.message })
    }
  }

  // GET /api/f1/energy?year&round&session&driver&lap — physics energy trace
  // for one lap under 2026 reg ceilings (MODELLED, citations included).
  if (request.method === 'GET' && new URL(request.url, 'http://localhost').pathname === '/api/f1/energy') {
    const params = new URL(request.url, 'http://localhost').searchParams
    const year = params.get('year') || ''
    const round = params.get('round') || ''
    const session = params.get('session') || ''
    const driver = (params.get('driver') || '').toUpperCase()
    const lap = params.get('lap') || ''
    if (!/^\d{4}$/.test(year) || !/^\d{1,2}$/.test(round) || !session || !/^[A-Z]{3}$/.test(driver) || !/^\d{1,3}$/.test(lap)) {
      return json(response, 400, { error: 'year, round, session, driver and lap are required' })
    }
    try {
      const cacheRel = `energy/${year}/${round}_${f1Slug(session)}/${driver}_${lap}.json`
      return json(response, 200, await f1CachedOrFetch(cacheRel, 'fetch_f1_energy.py', ['--year', year, '--round', round, '--session', session, '--driver', driver, '--lap', lap], 300000))
    } catch (error) {
      return json(response, 502, { error: error.message })
    }
  }

  // GET /api/f1/energyrace?year&round&session&driver — whole-race battery
  // trace + validation gates (MODELLED). Slow on first fetch (~1-3 min).
  if (request.method === 'GET' && new URL(request.url, 'http://localhost').pathname === '/api/f1/energyrace') {
    const params = new URL(request.url, 'http://localhost').searchParams
    const year = params.get('year') || ''
    const round = params.get('round') || ''
    const session = params.get('session') || ''
    const driver = (params.get('driver') || '').toUpperCase()
    if (!/^\d{4}$/.test(year) || !/^\d{1,2}$/.test(round) || !session || !/^[A-Z]{3}$/.test(driver)) {
      return json(response, 400, { error: 'year, round, session and driver are required' })
    }
    try {
      // v2 removes the old pit-charge behaviour and uses the shared energy
      // transition metadata. Keep it in a new cache namespace so old traces
      // cannot be served as if they were current.
      const cacheRel = `energyrace/v2/${year}/${round}_${f1Slug(session)}/${driver}.json`
      return json(response, 200, await f1CachedOrFetch(cacheRel, 'fetch_f1_energy_race.py', ['--year', year, '--round', round, '--session', session, '--driver', driver], 600000))
    } catch (error) {
      return json(response, 502, { error: error.message })
    }
  }

  // POST /api/f1/overtake/predict — score decision points with the trained
  // RandomForest. Body: {features:{...}} | {rows:[{...}]} | [{...}].
  // Returns P(SAVE)/P(DELAY)/P(ATTACK) + recommended label per row.
  if (request.method === 'POST' && new URL(request.url, 'http://localhost').pathname === '/api/f1/overtake/predict') {
    let input
    try {
      input = await readJsonBody(request)
    } catch {
      return json(response, 400, { error: 'request body must be valid JSON' })
    }
    try {
      const out = await runPythonWithInput('predict_overtake.py', [], JSON.stringify(input), 60000)
      return json(response, 200, JSON.parse(out))
    } catch (error) {
      return json(response, 502, { error: error.message })
    }
  }

  // POST /api/f1/replay/strategy — evaluate a bounded ATTACK/SAVE/DELAY
  // tactical tree from one observed decision point.  The Python engine owns
  // the state transition so the UI cannot silently invent a separate energy
  // model for the same replay.
  if (request.method === 'POST' && new URL(request.url, 'http://localhost').pathname === '/api/f1/replay/strategy') {
    let input
    try {
      input = await readJsonBody(request)
    } catch {
      return json(response, 400, { error: 'request body must be valid JSON' })
    }
    if (!input || typeof input !== 'object' || !input.focus) {
      return json(response, 400, { error: 'focus and replay context are required' })
    }
    try {
      const out = await runPythonWithInput('replay_strategy.py', [], JSON.stringify(input), 60000)
      return json(response, 200, JSON.parse(out))
    } catch (error) {
      return json(response, 502, { error: error.message })
    }
  }

  // GET /api/f1/decisionpoints?year&round&session — pre-extracted overtake
  // decision points for a race, each labelled ATTACK/DELAY/SAVE. Served straight
  // from the batch-extracted cache (no fetch), so it is instant.
  if (request.method === 'GET' && new URL(request.url, 'http://localhost').pathname === '/api/f1/decisionpoints') {
    const params = new URL(request.url, 'http://localhost').searchParams
    const year = params.get('year') || ''
    const round = params.get('round') || ''
    const session = params.get('session') || ''
    if (!/^\d{4}$/.test(year) || !/^\d{1,2}$/.test(round) || !session) {
      return json(response, 400, { error: 'year, round and session are required' })
    }
    const cachePath = path.join(F1_CACHE_DIR, 'decision-points', year, `${round}_${f1Slug(session)}.json`)
    try {
      let payload
      try {
        payload = JSON.parse(await readFile(cachePath, 'utf8'))
      } catch {
        payload = null
      }
      // Older caches predate the current causal feature cutoff. Rebuild
      // only the requested race on demand so the frontend cannot silently
      // evaluate stale methodology.
      if (!payload || payload.schemaVersion !== 'decision-point.v5') {
        const out = await runPython('extract_decision_points.py', [
          '--year', year, '--round', round, '--session', session,
        ], 600000)
        payload = JSON.parse(out)
      }
      return json(response, 200, payload)
    } catch {
      return json(response, 404, { error: `no decision points extracted for ${year} round ${round} ${session}` })
    }
  }

  // GET /api/f1/modelreport — trained-model provenance (temporal split, holdout
  // accuracy, feature importances) so the UI never hardcodes training facts.
  if (request.method === 'GET' && new URL(request.url, 'http://localhost').pathname === '/api/f1/modelreport') {
    try {
      const reportPath = path.join(F1_CACHE_DIR, 'models', 'overtake_report.json')
      return json(response, 200, JSON.parse(await readFile(reportPath, 'utf8')))
    } catch {
      return json(response, 404, { error: 'model not trained yet' })
    }
  }

  // GET /api/f1/rounds?year — round numbers + race names already extracted for a
  // season, read straight from the decision-point cache so the STRATEGY/OVERTAKE/
  // ENERGY selector can show real event names instantly, without spawning FastF1
  // or spending rate-limit budget. Distinct path from the FastF1-backed
  // /api/f1/events (used by LapExplorer) which it must not shadow.
  if (request.method === 'GET' && new URL(request.url, 'http://localhost').pathname === '/api/f1/rounds') {
    const year = new URL(request.url, 'http://localhost').searchParams.get('year') || ''
    if (!/^\d{4}$/.test(year)) return json(response, 400, { error: 'year is required' })
    const dir = path.join(F1_CACHE_DIR, 'decision-points', year)
    const events = []
    try {
      for (const file of await readdir(dir)) {
        const match = file.match(/^(\d+)_[a-z0-9]+\.json$/)
        if (!match) continue
        try {
          const payload = JSON.parse(await readFile(path.join(dir, file), 'utf8'))
          const round = Number(match[1])
          if (payload.eventName && !events.some((e) => e.round === round)) {
            events.push({ round, name: payload.eventName })
          }
        } catch { /* skip unreadable cache entry */ }
      }
    } catch { /* no races extracted for this year yet */ }
    events.sort((a, b) => a.round - b.round)
    return json(response, 200, { year, events })
  }

  // GET /api/f1/trackmap?year&round&session — circuit outline + corners for a session.
  if (request.method === 'GET' && request.url?.startsWith('/api/f1/trackmap')) {
    const params = new URL(request.url, 'http://localhost').searchParams
    const year = params.get('year') || ''
    const round = params.get('round') || ''
    const session = params.get('session') || ''
    if (!/^\d{4}$/.test(year) || !/^\d{1,2}$/.test(round) || !session) {
      return json(response, 400, { error: 'year, round and session are required' })
    }
    try {
      const cacheRel = `trackmap/${year}/${round}_${f1Slug(session)}.json`
      return json(response, 200, await f1CachedOrFetch(cacheRel, 'fetch_f1_trackmap.py', ['--year', year, '--round', round, '--session', session], 240000))
    } catch (error) {
      return json(response, 502, { error: error.message })
    }
  }

  return json(response, 404, { error: 'not found' })
}

// Vercel imports the default handler. Keep the local Node server for development.
export default handler

if (!process.env.VERCEL) {
  const server = http.createServer(handler)
  server.listen(PORT, () => console.log(`Pitwall API listening on http://localhost:${PORT}`))
}
