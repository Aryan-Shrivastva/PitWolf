"""Score a current-lap recommendation from the trained historical model.

The payload is the state known now. It must not include later laps of the
race being watched. Driver and track rates subtract this Grand Prix if it
was in the training set. DRS recipes come from 2018-2025 timing gaps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

from drs_recipe import DRS_GAP_S, recipe_for, score_drs_model
from train_driver_recommend import FEATURES

MODELS = Path(__file__).resolve().parents[1] / 'data' / 'f1-cache' / 'models' / 'recommend_rf.joblib'
REPORT = Path(__file__).resolve().parents[1] / 'data' / 'f1-cache' / 'models' / 'recommend_report.json'


def _num(value, default=0.0) -> float:
    try:
        number = float(value)
        return number if number == number else default
    except (TypeError, ValueError):
        return default


def _rate(count: float, denom: float, fallback: float) -> float:
    if denom <= 0:
        return fallback
    return float(count) / float(denom)


def exclude_current_race(prior: dict, year, location: str, recover=False) -> dict:
    if not prior:
        return {}
    cleaned = {key: value for key, value in prior.items() if key != 'byRace'}
    if year in (None, '') or not location:
        return cleaned
    race = (prior.get('byRace') or {}).get(f'{int(year)}|{location}')
    if not race:
        return cleaned
    laps = max(0, int(prior.get('laps') or 0) - int(race.get('laps') or 0))
    overtakes = max(0, int(prior.get('overtakes') or 0) - int(race.get('overtakes') or 0))
    races = max(0, int(prior.get('races') or 0) - 1)
    cleaned['laps'] = laps
    cleaned['overtakes'] = overtakes
    cleaned['races'] = races
    cleaned['overtakeSoonRate'] = round(_rate(overtakes, laps, cleaned.get('overtakeSoonRate') or 0), 4)
    if recover:
        lose = max(0, int(prior.get('loseWindows') or 0) - int(race.get('loseWindows') or 0))
        recoveries = max(0, int(prior.get('recoveries') or 0) - int(race.get('recoveries') or 0))
        cleaned['loseWindows'] = lose
        cleaned['recoveries'] = recoveries
        if lose:
            cleaned['recoverIfLostRate'] = round(_rate(recoveries, lose, 0.25), 4)
        else:
            cleaned.pop('recoverIfLostRate', None)
    return cleaned


def _pct(value) -> str:
    if value is None:
        return '—'
    return f'{int(round(float(value) * 100))}%'


def score(payload: dict, artifact=None) -> dict:
    if artifact is None:
        if not MODELS.exists():
            return {'error': 'recommend model missing — run train_driver_recommend.py'}
        artifact = joblib.load(MODELS)
    priors = artifact.get('priors') or {}
    recipes = artifact.get('drsRecipes') or {}
    driver = str(payload.get('driver') or '').upper()
    location = str(payload.get('location') or payload.get('track') or '')
    year = payload.get('year')
    driver_prior = exclude_current_race((priors.get('drivers') or {}).get(driver) or {}, year, location, recover=True)
    track_prior = exclude_current_race((priors.get('tracks') or {}).get(location) or {}, year, location)
    global_rates = artifact.get('global') or {}
    gap = _num(payload.get('gapS'))
    held = int(_num(payload.get('lapsInDrs')))
    behind_close = bool(payload.get('behindClose'))
    close = gap <= 1.2
    inside_drs = gap <= DRS_GAP_S
    drs = recipe_for(recipes, driver, location, held if inside_drs else 0)
    row = {
        'gapS': gap,
        'closingRateS': _num(payload.get('closingRateS')),
        'position': _num(payload.get('position'), 10),
        'lapFraction': _num(payload.get('lapFraction'), 0.1),
        'deltaToPrevS': _num(payload.get('deltaToPrevS')),
        'modelledLeftPct': _num(payload.get('modelledLeftPct'), 100),
        'modelledUsedMj': _num(payload.get('modelledUsedMj')),
        'closeBattle': 1.0 if close else 0.0,
        'driverOvertakeRate': _num(driver_prior.get('overtakeSoonRate'), global_rates.get('overtakeSoonRate', 0.15)),
        'driverRecoverRate': _num(driver_prior.get('recoverIfLostRate'), global_rates.get('recoverIfLostRate', 0.25)),
        'trackOvertakeRate': _num(track_prior.get('overtakeSoonRate'), global_rates.get('overtakeSoonRate', 0.15)),
        'inDrs': 1.0 if inside_drs else 0.0,
        'lapsInDrs': float(held if inside_drs else 0),
        'driverEfficiency': _num(drs.get('efficiency'), global_rates.get('drsEfficiency', 0.35)),
        'typicalWaitLaps': _num(drs.get('typicalWaitLaps'), 2.0),
    }
    modelled = score_drs_model(artifact.get('drsModels') or {}, row)
    if modelled:
        drs.update(modelled)
        drs['source'] = 'random_forest'
    vector = np.array([[row[name] for name in FEATURES]], dtype=float)
    probs = {}
    for target, model in (artifact.get('models') or {}).items():
        proba = model.predict_proba(vector)
        classes = list(model.classes_)
        positive = classes.index(1) if 1 in classes else 0
        probs[target] = round(float(proba[0, positive] if proba.shape[1] > 1 else proba[0, 0]), 3)

    outcome = {}
    outcome_models = artifact.get('outcomeModels') or {}
    places_model = outcome_models.get('placesToFlag')
    if places_model is not None:
        places = float(places_model.predict(vector)[0])
        now = row['position']
        expected = max(1, min(20, int(round(now - places))))
        outcome['placesToFlag'] = round(places, 2)
        outcome['expectedFinish'] = expected
        outcome['runningPosition'] = int(round(now))
    lose_model = outcome_models.get('loseSoon')
    if lose_model is not None:
        proba = lose_model.predict_proba(vector)
        classes = list(lose_model.classes_)
        positive = classes.index(1) if 1 in classes else 0
        outcome['pLoseSoon'] = round(float(proba[0, positive] if proba.shape[1] > 1 else proba[0, 0]), 3)

    push = probs.get('pushHelps', 0)
    recover = probs.get('recoverIfLost', 0)
    soon = probs.get('overtakeSoon', 0)
    ahead = payload.get('ahead')
    behind = payload.get('behind')
    selected = driver or 'the selected driver'
    years = drs.get('years') or '2018-2025'
    lines = []
    if gap > 0 and ahead:
        zone = 'inside DRS range' if inside_drs else 'outside DRS range'
        lines.append(f'{selected} is {gap:.2f}s behind {ahead} right now ({zone}, 1.0s timing).')
    elif ahead:
        lines.append(f'{selected} is effectively on {ahead}.')
    if drs.get('windows'):
        typical = drs.get('typicalWaitLaps')
        wait_bit = f' When it converts, typical wait is {typical:g} laps after entering DRS (p25–p75 {drs.get("waitLapsP25")}–{drs.get("waitLapsP75")}).' if typical else ''
        lines.append(
            f'{selected} DRS efficiency {drs["efficiency"]:.2f} from {years} ({drs["converts"]}/{drs["windows"]} windows).{wait_bit}'
        )
    if inside_drs and ahead:
        more = drs.get('predictedMoreLaps')
        wait = f' Modelled wait: {more} more lap{"s" if more != 1 else ""} if the gap stays in DRS.' if more else ''
        lines.append(
            f'In DRS of {ahead} for {max(1, held)} lap{"s" if max(1, held) != 1 else ""}. '
            f'RandomForest: {_pct(drs.get("pNext1"))} within 1 more lap, {_pct(drs.get("pNext2"))} within 2, {_pct(drs.get("pNext3"))} within 3.{wait}'
        )
    elif ahead and gap > DRS_GAP_S:
        lines.append(f'Not in DRS yet. Historical efficiency still {drs["efficiency"]:.2f}; the wait clock starts after the gap is ≤ 1.0s.')
    if close and push >= 0.55:
        lines.append(f'IF the gap stays this small: PUSH. Historical net-gain within 5 laps is {_pct(push)}.')
        action = 'PUSH'
    elif push <= 0.4:
        lines.append(f'Do not push now. Historical net-gain chance from this state is {_pct(push)}. HOLD and wait.')
        action = 'HOLD'
    else:
        lines.append(f'IF {ahead or "the car ahead"} starts to lose time, a push can help ({_pct(push)} historical net-gain). Otherwise HOLD.')
        action = 'HOLD'
    if close or behind_close:
        target = ahead if close else behind
        lines.append(f'IF {selected} loses this place now{f" to {target}" if target else ""}, take-back inside 5 laps has been {_pct(recover)} in training races.')
    if soon >= 0.4 and not inside_drs:
        lines.append(f'IF this battle stays on: overtake chance in the next few laps is {_pct(soon)} from prior races, not from later laps of this one.')

    holdout = {}
    if REPORT.exists():
        try:
            report = json.loads(REPORT.read_text(encoding='utf8'))
            holdout = {
                'racesTest': report.get('racesTest'),
                'racesTrain': report.get('racesTrain'),
                'years': report.get('years'),
                'targets': {
                    name: {
                        'testAuc': (report.get('targets') or {}).get(name, {}).get('testAuc'),
                        'testBrier': (report.get('targets') or {}).get(name, {}).get('testBrier'),
                    }
                    for name in ('overtakeSoon', 'recoverIfLost', 'pushHelps')
                },
            }
        except Exception:
            holdout = {}

    return {
        'ok': True,
        'schema': artifact.get('schema'),
        'driver': driver,
        'ahead': ahead,
        'behind': behind,
        'probabilities': probs,
        'outcome': outcome,
        'action': action,
        'lines': lines[:7],
        'drs': drs,
        'priors': {
            'driver': {key: driver_prior.get(key) for key in ('laps', 'races', 'overtakeSoonRate', 'recoverIfLostRate', 'loseWindows')},
            'track': {key: track_prior.get(key) for key in ('laps', 'races', 'overtakeSoonRate')},
        },
        'holdout': holdout,
        'excludedRace': f'{year}|{location}' if year not in (None, '') and location else None,
        'provenance': 'DRS recipe and models trained on 2018-2025 cached races. Uses this lap only. Later laps of the race you are watching are not read.',
    }


if __name__ == '__main__':
    payload = json.loads(sys.stdin.read() or '{}')
    print(json.dumps(score(payload)))
