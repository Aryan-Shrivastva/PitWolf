"""DRS-to-overtake recipes from cached race laps.

A window starts when a driver first sits within 1.0s of the car ahead
(FIA-style DRS range from timing, not team DRS telemetry). It ends on an
overtake, a pit, the car ahead changing, or the gap opening back out.

Wait times and a 0–1 efficiency are computed only from 2018–2025 races
passed in by the trainer. Playback of a later year does not add windows.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

DRS_GAP_S = 1.0
DRS_BREAK_S = 1.5
MAX_WAIT = 8
HORIZON = 3
HISTORY_YEARS = range(2018, 2026)


def in_drs(step: dict, gap=DRS_GAP_S) -> bool:
    if not step or step.get('isPitLap') or not step.get('ahead'):
        return False
    try:
        return float(step.get('gapToAheadS') or 99) <= gap
    except (TypeError, ValueError):
        return False


def laps_already_in_drs(laps: list[dict], current_lap: int) -> int:
    past = [step for step in laps if int(step.get('lap') or 0) <= int(current_lap)]
    if not past:
        return 0
    now = past[-1]
    if not in_drs(now):
        return 0
    held = 0
    ahead = now.get('ahead')
    for step in reversed(past):
        if step.get('isPitLap') or step.get('ahead') != ahead or not in_drs(step):
            break
        held += 1
    return held


def extract_windows(field: dict, event: dict) -> list[dict]:
    year = int(event.get('year') or 0)
    location = str(event.get('location') or event.get('name') or 'unknown')
    out = []
    for driver_row in field.get('drivers') or []:
        laps = driver_row.get('laps') or []
        index = 0
        while index < len(laps):
            step = laps[index]
            prev = laps[index - 1] if index else None
            fresh = in_drs(step) and (prev is None or not in_drs(prev) or prev.get('ahead') != step.get('ahead'))
            if not fresh:
                index += 1
                continue
            ahead = step.get('ahead')
            wait_s = 0.0
            converted = False
            wait_laps = 0
            duration = 0
            for later in laps[index:index + MAX_WAIT]:
                duration += 1
                wait_s += float(later.get('lapTimeS') or 0.0)
                if later.get('isPitLap'):
                    break
                if later.get('ahead') != ahead and later.get('posChange', 0) <= 0:
                    break
                if later is not step and later.get('posChange', 0) > 0:
                    converted = True
                    wait_laps = duration
                    break
                if float(later.get('gapToAheadS') or 0) > DRS_BREAK_S:
                    break
            else:
                wait_laps = duration
            if not converted:
                wait_laps = duration
            last_lap = max(1, max((item.get('lap') or 1) for item in laps))
            steps = []
            for held, later in enumerate(laps[index:index + duration], start=1):
                modelled = later.get('modelled') or {}
                steps.append({
                    'lap': later.get('lap'),
                    'gapS': float(later.get('gapToAheadS') or 0.0),
                    'closingRateS': float(later.get('closingRateS') or 0.0),
                    'position': float(later.get('timingPosition') or 10),
                    'lapFraction': float(later.get('lap') or 1) / last_lap,
                    'deltaToPrevS': float(later.get('deltaToPrevS') or 0.0),
                    'modelledLeftPct': float(modelled.get('endPct') or 100.0),
                    'modelledUsedMj': float(modelled.get('usedMj') or 0.0),
                    'lapsInDrs': float(held),
                })
            out.append({
                'year': year,
                'location': location,
                'driver': driver_row['driver'],
                'ahead': ahead,
                'startLap': step['lap'],
                'converted': int(converted),
                'waitLaps': wait_laps,
                'waitS': round(wait_s, 3),
                'duration': duration,
                'steps': steps,
            })
            index += max(1, duration)
    return out


DRS_FEATURES = [
    'gapS', 'closingRateS', 'position', 'lapFraction', 'deltaToPrevS',
    'modelledLeftPct', 'modelledUsedMj', 'lapsInDrs',
    'driverEfficiency', 'typicalWaitLaps', 'trackOvertakeRate',
]


def expand_drs_samples(windows: list[dict], recipes: dict | None = None) -> list[dict]:
    recipe_drivers = (recipes or {}).get('drivers') or {}
    recipe_tracks = (recipes or {}).get('tracks') or {}
    global_eff = float((recipes or {}).get('efficiency') or 0.35)
    rows = []
    for window in windows:
        recipe = recipe_drivers.get(window['driver']) or {}
        track = recipe_tracks.get(window['location']) or {}
        efficiency = float(recipe.get('efficiency') or global_eff)
        typical = float(recipe.get('waitLapsP50') or 2.0)
        track_rate = float(track.get('overtakeSoonRate') or track.get('efficiency') or global_eff)
        for step in window.get('steps') or []:
            held = int(step.get('lapsInDrs') or 1)
            converted = bool(window['converted'])
            wait = int(window['waitLaps'] or 0)
            remaining = (wait - held) if converted else None
            rows.append({
                **step,
                'driver': window['driver'],
                'location': window['location'],
                'year': window['year'],
                'driverEfficiency': efficiency,
                'typicalWaitLaps': typical,
                'trackOvertakeRate': track_rate,
                'convertWithin1': int(converted and wait <= held + 1),
                'convertWithin2': int(converted and wait <= held + 2),
                'convertWithin3': int(converted and wait <= held + 3),
                'remainingWait': remaining if remaining is not None and remaining > 0 else None,
            })
    return rows


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {'p25': None, 'p50': None, 'p75': None}
    arr = np.array(values, dtype=float)
    return {
        'p25': round(float(np.percentile(arr, 25)), 2),
        'p50': round(float(np.percentile(arr, 50)), 2),
        'p75': round(float(np.percentile(arr, 75)), 2),
    }


def _residual(windows: list[dict]) -> dict:
    residual = {}
    for held in range(0, MAX_WAIT):
        survived = [row for row in windows if row['duration'] >= held]
        if len(survived) < 8:
            continue
        residual[str(held)] = []
        for extra in range(1, HORIZON + 1):
            took = sum(
                1 for row in survived
                if row['converted'] and row['waitLaps'] <= held + extra
            )
            residual[str(held)].append(round(took / len(survived), 3))
    return residual


def pack_recipes(windows: list[dict]) -> dict:
    by_driver = defaultdict(list)
    by_track = defaultdict(list)
    for row in windows:
        by_driver[row['driver']].append(row)
        by_track[row['location']].append(row)

    def pack(groups, min_windows=6):
        out = {}
        for key, rows in groups.items():
            if len(rows) < min_windows:
                continue
            converts = [row for row in rows if row['converted']]
            waits = [row['waitLaps'] for row in converts]
            wait_s = [row['waitS'] for row in converts]
            efficiency = len(converts) / len(rows)
            out[key] = {
                'windows': len(rows),
                'converts': len(converts),
                'efficiency': round(efficiency, 3),
                'waitLaps': waits[:400],
                'waitLapsHist': {str(k): waits.count(k) for k in range(1, MAX_WAIT + 1)},
                **{f'waitLaps{name.upper()}': value for name, value in _percentiles(waits).items()},
                **{f'waitS{name.upper()}': value for name, value in _percentiles(wait_s).items()},
                'residual': _residual(rows),
            }
        return out

    global_rows = windows
    converts = [row for row in global_rows if row['converted']]
    return {
        'gapS': DRS_GAP_S,
        'years': f'{min(HISTORY_YEARS)}-{max(HISTORY_YEARS)}',
        'windows': len(global_rows),
        'converts': len(converts),
        'efficiency': round(len(converts) / max(1, len(global_rows)), 3),
        'drivers': pack(by_driver),
        'tracks': pack(by_track, min_windows=12),
        'note': 'Timing gap ≤ 1.0s is used as DRS range. Not official DRS activation telemetry.',
    }


def recipe_for(recipes: dict, driver: str, location: str, held: int, seed: int = 0) -> dict:
    global_eff = float((recipes or {}).get('efficiency') or 0.35)
    driver_row = ((recipes or {}).get('drivers') or {}).get(driver) or {}
    track_row = ((recipes or {}).get('tracks') or {}).get(location) or {}
    efficiency = float(driver_row.get('efficiency') or global_eff)
    typical = driver_row.get('waitLapsP50')
    residual = (driver_row.get('residual') or {}).get(str(min(max(held, 0), MAX_WAIT - 1)))
    if not residual:
        residual = (driver_row.get('residual') or {}).get('0') or [0.2, 0.35, 0.45]
    return {
        'inDrsHistory': True,
        'source': 'empirical',
        'efficiency': round(efficiency, 3),
        'windows': driver_row.get('windows') or 0,
        'converts': driver_row.get('converts') or 0,
        'typicalWaitLaps': typical,
        'waitLapsP25': driver_row.get('waitLapsP25'),
        'waitLapsP75': driver_row.get('waitLapsP75'),
        'waitSP50': driver_row.get('waitSP50'),
        'trackEfficiency': track_row.get('efficiency'),
        'trackWindows': track_row.get('windows'),
        'heldLaps': held,
        'pNext1': residual[0] if len(residual) > 0 else None,
        'pNext2': residual[1] if len(residual) > 1 else None,
        'pNext3': residual[2] if len(residual) > 2 else None,
        'predictedMoreLaps': typical,
        'years': (recipes or {}).get('years'),
    }


def _positive_proba(model, vector) -> float:
    proba = model.predict_proba(vector)
    classes = list(model.classes_)
    positive = classes.index(1) if 1 in classes else 0
    return float(proba[0, positive] if proba.shape[1] > 1 else proba[0, 0])


def score_drs_model(drs_models: dict, features: dict) -> dict:
    if not drs_models:
        return {}
    names = drs_models.get('features') or DRS_FEATURES
    vector = np.array([[float(features.get(name) or 0.0) for name in names]], dtype=float)
    out = {'source': 'random_forest'}
    classifiers = drs_models.get('classifiers') or {}
    for key, target in (('pNext1', 'convertWithin1'), ('pNext2', 'convertWithin2'), ('pNext3', 'convertWithin3')):
        model = classifiers.get(target)
        if model is None:
            continue
        out[key] = round(_positive_proba(model, vector), 3)
    if 'pNext2' in out and 'pNext1' in out:
        out['pNext2'] = max(out['pNext2'], out['pNext1'])
    if 'pNext3' in out and 'pNext2' in out:
        out['pNext3'] = max(out['pNext3'], out['pNext2'])
    regressor = drs_models.get('waitRegressor')
    if regressor is not None:
        wait = float(regressor.predict(vector)[0])
        out['predictedMoreLaps'] = int(max(1, min(MAX_WAIT, round(wait))))
        out['predictedMoreLapsRaw'] = round(wait, 2)
    return out
