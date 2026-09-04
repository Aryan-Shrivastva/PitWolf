// Thin clients for the PitWolf decision-engine backend endpoints.
//
// ENERGY  -> /api/f1/energyrace   (2026-reg physics projection + validation gates)
// OVERTAKE-> /api/f1/decisionpoints (batch-extracted, labelled ATTACK/DELAY/SAVE)
//          + /api/f1/overtake/predict (RandomForest class probabilities)
//          + /api/f1/modelreport (training provenance: split, holdout accuracy)
//          + /api/f1/rounds (per-season round numbers + race names from cache)
//
// Every energy figure is MODELLED from real telemetry under cited FIA limits; it
// is never measured team data. The overtake labels are modelled outcomes too.

import { fetchJson, fetchWithRetry } from '../components/LapExplorer'

export function fetchEnergyRace(year, round, session, driver) {
  const q = `year=${year}&round=${round}&session=${encodeURIComponent(session)}&driver=${encodeURIComponent(driver)}`
  // First fetch spawns a ~1-3 min physics pass, so retry transient 5xx.
  return fetchCached(`/api/f1/energyrace?${q}`, 6 * 60 * 60 * 1000)
}

export function fetchEnergyLap(year, round, session, driver, lap) {
  const q = `year=${year}&round=${round}&session=${encodeURIComponent(session)}&driver=${encodeURIComponent(driver)}&lap=${lap}`
  return fetchWithRetry(`/api/f1/energy?${q}`, 3)
}

export function fetchDecisionPoints(year, round, session) {
  const q = `year=${year}&round=${round}&session=${encodeURIComponent(session)}`
  return fetchCached(`/api/f1/decisionpoints?${q}`, 60 * 60 * 1000)
}

const memoryCache = new Map()
// Bump this when a model/schema artifact changes so an old report cannot be
// displayed from the browser after retraining.
const CACHE_PREFIX = 'pitwolf:api:v4:'

function storageGet(key) {
  try {
    if (typeof sessionStorage === 'undefined') return null
    const raw = sessionStorage.getItem(CACHE_PREFIX + key)
    if (!raw) return null
    const entry = JSON.parse(raw)
    return entry.expires > Date.now() ? entry.value : null
  } catch {
    return null
  }
}

function storageSet(key, value, ttl) {
  try {
    if (typeof sessionStorage === 'undefined') return
    sessionStorage.setItem(CACHE_PREFIX + key, JSON.stringify({ expires: Date.now() + ttl, value }))
  } catch {
    // Large race payloads may exceed browser storage; memory caching still works.
  }
}

function fetchCached(url, ttl) {
  const key = url
  const memory = memoryCache.get(key)
  if (memory?.value !== undefined && memory.expires > Date.now()) return Promise.resolve(memory.value)
  if (memory?.promise) return memory.promise
  const stored = storageGet(key)
  if (stored !== null) {
    memoryCache.set(key, { value: stored, expires: Date.now() + ttl })
    return Promise.resolve(stored)
  }
  const promise = fetchWithRetry(url, 3).then((value) => {
    memoryCache.set(key, { value, expires: Date.now() + ttl })
    storageSet(key, value, ttl)
    return value
  }).catch((error) => {
    memoryCache.delete(key)
    throw error
  })
  memoryCache.set(key, { promise, expires: Date.now() + ttl })
  return promise
}

export async function predictOvertake(rows) {
  const first = rows?.[0] ?? {}
  const last = rows?.[rows.length - 1] ?? {}
  const cacheKey = `predict:${first.year}:${first.round}:${first.session}:${rows?.length ?? 0}:${first.lap}:${last.lap}`
  const cached = storageGet(cacheKey)
  if (cached !== null) return cached
  const memory = memoryCache.get(cacheKey)
  if (memory?.value !== undefined && memory.expires > Date.now()) return memory.value
  if (memory?.promise) return memory.promise
  const promise = fetch('/api/f1/overtake/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rows }),
  }).then(async (response) => {
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) {
      const error = new Error(payload.error || `predict failed (${response.status})`)
      error.status = response.status
      throw error
    }
    memoryCache.set(cacheKey, { value: payload, expires: Date.now() + 60 * 60 * 1000 })
    storageSet(cacheKey, payload, 60 * 60 * 1000)
    return payload
  }).catch((error) => {
    memoryCache.delete(cacheKey)
    throw error
  })
  memoryCache.set(cacheKey, { promise, expires: Date.now() + 60 * 60 * 1000 })
  return promise
}

export async function fetchStrategyReplay(context) {
  const focus = context?.focus ?? {}
  const cacheKey = `replay:${focus.year}:${focus.round}:${focus.session}:${focus.lap}:${focus.driver}:${focus.defender}:${context?.regulationEra ?? ''}`
  const memory = memoryCache.get(cacheKey)
  if (memory?.value !== undefined && memory.expires > Date.now()) return memory.value
  if (memory?.promise) return memory.promise
  const promise = fetch('/api/f1/replay/strategy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(context),
  }).then(async (response) => {
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) {
      const error = new Error(payload.error || `strategy replay failed (${response.status})`)
      error.status = response.status
      throw error
    }
    memoryCache.set(cacheKey, { value: payload, expires: Date.now() + 60 * 60 * 1000 })
    return payload
  }).catch((error) => {
    memoryCache.delete(cacheKey)
    throw error
  })
  memoryCache.set(cacheKey, { promise, expires: Date.now() + 60 * 60 * 1000 })
  return promise
}

export function fetchModelReport() {
  return fetchCached('/api/f1/modelreport', 60 * 60 * 1000)
}

export async function fetchEvents(year) {
  const payload = await fetchCached(`/api/f1/rounds?year=${year}`, 6 * 60 * 60 * 1000)
  return payload.events ?? []
}

// Strategy accent colours, kept identical to the rule engine so the rebuilt
// pages preserve the PitWolf palette (ATTACK orange / SAVE teal / DELAY blue).
export const STRATEGY_COLORS = {
  ATTACK: '#ff7043',
  SAVE: '#63e6be',
  DELAY: '#a9bfff',
}

export const STRATEGY_ORDER = ['ATTACK', 'DELAY', 'SAVE']
