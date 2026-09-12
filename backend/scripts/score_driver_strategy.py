"""Retrospective race story for the Strategy dashboard.

Observed start/finish and place changes come from the cached session.
Missed DRS windows are judged with the 2018-2025 recipe, not with later
laps used as a spoiler during live playback.
"""

from __future__ import annotations

import json
import sys

from clip_cached_race import build_field_story, clip_cached, load_session, session_path
from drs_recipe import DRS_GAP_S, extract_windows, in_drs, recipe_for
from score_recommend import MODELS, score

import joblib


def _classified(item: dict, fallback):
    raw = item.get('classifiedPosition') or item.get('position') or fallback
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


def _driver_story(row: dict, windows: list[dict], recipes: dict, location: str) -> dict:
    laps = row.get('laps') or []
    start = laps[0]['timingPosition'] if laps else None
    finish_running = laps[-1]['timingPosition'] if laps else None
    finish = row.get('classifiedPosition') or finish_running
    net = None if start is None or finish is None else start - finish
    overtakes = [step for step in laps if step.get('event') == 'OVERTAKE']
    lost = [
        step for step in laps
        if (step.get('posChange') or 0) < 0 and not step.get('isPitLap')
    ]
    mine = [item for item in windows if item['driver'] == row['driver']]
    converted = [item for item in mine if item['converted']]
    missed = [item for item in mine if not item['converted']]
    recipe = recipe_for(recipes, row['driver'], location, 0, seed=abs(hash(row['driver'])) % (2**32))
    expected_extra = round(float(recipe.get('efficiency') or 0) * len(missed), 2)
    potential = finish
    if finish is not None and expected_extra >= 0.5:
        potential = max(1, int(finish) - int(round(expected_extra)))
    changes = []
    for step in laps:
        delta = step.get('posChange') or 0
        if delta == 0:
            continue
        changes.append({
            'lap': step['lap'],
            'fromPosition': step['timingPosition'] + delta,
            'toPosition': step['timingPosition'],
            'places': delta,
            'event': step.get('event'),
            'ahead': step.get('ahead'),
            'gapS': step.get('gapToAheadS'),
            'pit': bool(step.get('isPitLap')),
        })
    return {
        'driver': row['driver'],
        'name': row.get('name'),
        'startPosition': start,
        'finishPosition': finish,
        'finishRunning': finish_running,
        'netPlaces': net,
        'overtakes': len(overtakes),
        'overtakeLaps': [step['lap'] for step in overtakes],
        'placesLost': abs(sum(step.get('posChange') or 0 for step in lost)),
        'drsWindows': len(mine),
        'drsConverted': len(converted),
        'drsMissed': len(missed),
        'efficiency': recipe.get('efficiency'),
        'typicalWaitLaps': recipe.get('typicalWaitLaps'),
        'expectedExtraPlaces': expected_extra,
        'potentialFinish': potential,
        'couldFinishBetter': bool(potential is not None and finish is not None and potential < finish),
        'changes': changes[:24],
        'missedWindows': [
            {'startLap': item['startLap'], 'ahead': item['ahead'], 'waitLaps': item['waitLaps']}
            for item in missed[:8]
        ],
        'convertedWindows': [
            {'startLap': item['startLap'], 'ahead': item['ahead'], 'waitLaps': item['waitLaps']}
            for item in converted[:8]
        ],
    }


def _battle_payload(row: dict, event: dict, step: dict) -> dict:
    last = max(1, step.get('lap') or 1)
    modelled = step.get('modelled') or {}
    return {
        'year': event.get('year'),
        'location': event.get('location'),
        'driver': row['driver'],
        'ahead': step.get('ahead'),
        'behind': step.get('behind'),
        'lap': step.get('lap'),
        'gapS': step.get('gapToAheadS'),
        'closingRateS': step.get('closingRateS'),
        'position': step.get('timingPosition'),
        'lapFraction': (step.get('lap') or 1) / last,
        'deltaToPrevS': step.get('deltaToPrevS'),
        'modelledLeftPct': modelled.get('endPct'),
        'modelledUsedMj': modelled.get('usedMj'),
        'lapsInDrs': 1 if in_drs(step) else 0,
    }


def pick_battle_steps(laps: list[dict], limit=3) -> list[dict]:
    close = [step for step in laps if in_drs(step)]
    if not close:
        close = [step for step in laps if (step.get('gapToAheadS') or 9) <= 1.5 and step.get('ahead')]
    if not close:
        return laps[:1]
    picked = [close[0]]
    if len(close) > 2:
        picked.append(close[len(close) // 2])
    if close[-1] is not picked[0]:
        picked.append(close[-1])
    uniq = []
    seen = set()
    for step in picked:
        if step['lap'] in seen:
            continue
        seen.add(step['lap'])
        uniq.append(step)
    return uniq[:limit]


def build_strategy_story(year, round_number, session_name, driver: str | None) -> dict:
    if not year or not round_number:
        first = clip_cached(None, None, 'Race', driver, None, 'AUTO', 4.0)
        if first.get('error'):
            return first
        event = (first.get('actualRace') or {}).get('event') or {}
        year = event.get('year')
        round_number = event.get('round')
        session_name = 'Race'
    path = session_path(int(year), int(round_number), session_name or 'Race')
    if not path.exists():
        return {'error': f'no cached race for {year} R{round_number}'}
    payload = load_session(path)
    event = payload.get('event') or {}
    field = build_field_story(payload)
    location = str(event.get('location') or '')
    windows = extract_windows(field, event)
    recipes = {}
    if MODELS.exists():
        recipes = (joblib.load(MODELS) or {}).get('drsRecipes') or {}
    stories = [_driver_story(row, windows, recipes, location) for row in field.get('drivers') or []]
    selected_code = (driver or '').upper() or (stories[0]['driver'] if stories else None)
    selected = next((item for item in stories if item['driver'] == selected_code), stories[0] if stories else None)
    recs = []
    selected_row = next((row for row in field.get('drivers') or [] if row['driver'] == selected_code), None)
    if selected_row and MODELS.exists():
        for step in pick_battle_steps(selected_row.get('laps') or []):
            scored = score(_battle_payload(selected_row, event, step))
            scored['lap'] = step.get('lap')
            recs.append(scored)
    return {
        'ok': True,
        'schema': 'driver-strategy-story.v1',
        'event': event,
        'driver': selected_code,
        'drsGapS': DRS_GAP_S,
        'selected': selected,
        'drivers': stories,
        'recommendations': recs,
        'provenance': [
            'REAL first-lap running order and classified finish from the cached session',
            'REAL timing movements labelled OVERTAKE only when position improved on a non-pit lap',
            'MODELLED extra places = missed DRS windows × 2018-2025 driver efficiency',
            'Recommendations use driver-recommend.v2. This is a finished-race review, not a live spoiler.',
        ],
    }


if __name__ == '__main__':
    payload = json.loads(sys.stdin.read() or '{}')
    print(json.dumps(build_strategy_story(
        payload.get('year'),
        payload.get('round'),
        payload.get('session') or 'Race',
        payload.get('driver'),
    )))
