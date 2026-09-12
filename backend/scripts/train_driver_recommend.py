"""Train causal push / recover recommendations from cached race laps.

Uses every locally cached *_race.json. Features are known at the start of the
labelled window (gap, pace, position, modelled leftover so far). Labels look
forward inside that historical race only, and never into a race held out for
test.

Races are split at random (not by lap) so the test set is a fresh set of
Grands Prix. The current playback race is not required at train time.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

from clip_cached_race import SESSIONS, build_field_story
from drs_recipe import (
    DRS_FEATURES, HISTORY_YEARS, expand_drs_samples, extract_windows,
    in_drs, laps_already_in_drs, pack_recipes,
)

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'f1-cache'
MODELS = ROOT / 'models'
FEATURES = [
    'gapS', 'closingRateS', 'position', 'lapFraction', 'deltaToPrevS',
    'modelledLeftPct', 'modelledUsedMj', 'closeBattle',
    'driverOvertakeRate', 'driverRecoverRate', 'trackOvertakeRate',
    'inDrs', 'lapsInDrs', 'driverEfficiency', 'typicalWaitLaps',
]
TARGETS = ('overtakeSoon', 'recoverIfLost', 'pushHelps')
SEED = 42
TEST_RACE_FRACTION = 0.2


def race_key(event: dict, path: Path) -> tuple:
    year = int(event.get('year') or path.parent.name)
    round_number = int(event.get('round') or path.name.split('_')[0])
    return year, round_number


def iter_races() -> list[tuple[tuple, dict]]:
    out = []
    for path in sorted(SESSIONS.glob('*/*_race.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        event = payload.get('event') or {}
        if not payload.get('laps'):
            continue
        out.append((race_key(event, path), payload))
    return out


def samples_from_story(field: dict, event: dict) -> list[dict]:
    location = str(event.get('location') or event.get('name') or 'unknown')
    year = int(event.get('year') or 0)
    rows = []
    for driver_row in field.get('drivers') or []:
        laps = driver_row.get('laps') or []
        if len(laps) < 4:
            continue
        last_lap = max(step['lap'] for step in laps)
        for index, step in enumerate(laps[:-1]):
            future = laps[index + 1:index + 6]
            later_pos = [item['timingPosition'] for item in future]
            overtake_soon = any(item.get('posChange', 0) > 0 for item in future)
            lose_soon = any(item.get('posChange', 0) < 0 for item in future[:2])
            recover = False
            if lose_soon:
                lost_at = next((j for j, item in enumerate(future) if item.get('posChange', 0) < 0), None)
                if lost_at is not None:
                    recover = any(item.get('posChange', 0) > 0 for item in future[lost_at + 1:])
            net = (later_pos[-1] - step['timingPosition']) if later_pos else 0
            finish = driver_row.get('classifiedPosition') or laps[-1]['timingPosition']
            try:
                finish = int(finish)
            except (TypeError, ValueError):
                finish = laps[-1]['timingPosition']
            places_to_flag = float(step['timingPosition'] - finish)
            modelled = step.get('modelled') or {}
            held = laps_already_in_drs(laps, step['lap'])
            rows.append({
                'year': year,
                'location': location,
                'driver': driver_row['driver'],
                'lap': step['lap'],
                'gapS': float(step.get('gapToAheadS') or 0.0),
                'closingRateS': float(step.get('closingRateS') or 0.0),
                'position': float(step.get('timingPosition') or 10),
                'lapFraction': step['lap'] / max(1, last_lap),
                'deltaToPrevS': float(step.get('deltaToPrevS') or 0.0),
                'modelledLeftPct': float(modelled.get('endPct') or 100.0),
                'modelledUsedMj': float(modelled.get('usedMj') or 0.0),
                'closeBattle': 1.0 if (step.get('gapToAheadS') or 9) <= 1.2 else 0.0,
                'inDrs': 1.0 if in_drs(step) else 0.0,
                'lapsInDrs': float(held),
                'overtakeSoon': int(overtake_soon),
                'recoverIfLost': int(recover) if lose_soon else None,
                'loseSoon': int(lose_soon),
                'pushHelps': int(net < 0),
                'placesToFlag': places_to_flag,
                'finishPosition': float(finish),
            })
    return rows


def attach_priors(rows: list[dict], prior_rows: list[dict], recipes: dict | None = None) -> None:
    driver_ot = defaultdict(list)
    driver_rec = defaultdict(list)
    track_ot = defaultdict(list)
    for row in prior_rows:
        driver_ot[row['driver']].append(row['overtakeSoon'])
        track_ot[row['location']].append(row['overtakeSoon'])
        if row['recoverIfLost'] is not None:
            driver_rec[row['driver']].append(row['recoverIfLost'])
    global_ot = float(np.mean([row['overtakeSoon'] for row in prior_rows]) or 0.15)
    rec_values = [row['recoverIfLost'] for row in prior_rows if row['recoverIfLost'] is not None]
    global_rec = float(np.mean(rec_values) if rec_values else 0.25)
    recipe_drivers = (recipes or {}).get('drivers') or {}
    global_eff = float((recipes or {}).get('efficiency') or 0.35)
    global_wait = 2.0
    for row in rows:
        ot = driver_ot.get(row['driver']) or [global_ot]
        rec = driver_rec.get(row['driver']) or [global_rec]
        tr = track_ot.get(row['location']) or [global_ot]
        recipe = recipe_drivers.get(row['driver']) or {}
        row['driverOvertakeRate'] = float(np.mean(ot))
        row['driverRecoverRate'] = float(np.mean(rec))
        row['trackOvertakeRate'] = float(np.mean(tr))
        row['driverEfficiency'] = float(recipe.get('efficiency') or global_eff)
        row['typicalWaitLaps'] = float(recipe.get('waitLapsP50') or global_wait)


def matrix(rows: list[dict], target: str) -> tuple[np.ndarray, np.ndarray]:
    usable = [row for row in rows if row.get(target) is not None]
    x = np.array([[float(row[name]) for name in FEATURES] for row in usable], dtype=float)
    y = np.array([int(row[target]) for row in usable], dtype=int)
    return x, y


def fit_target(train: list[dict], test: list[dict], target: str) -> dict:
    x_train, y_train = matrix(train, target)
    x_test, y_test = matrix(test, target)
    model = RandomForestClassifier(
        n_estimators=220,
        min_samples_leaf=12,
        max_depth=10,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)
    positive = list(model.classes_).index(1) if 1 in model.classes_ else 0
    scores = proba[:, positive] if proba.shape[1] > 1 else proba[:, 0]
    report = {
        'samplesTrain': int(len(y_train)),
        'samplesTest': int(len(y_test)),
        'positiveRateTrain': round(float(y_train.mean()), 4) if len(y_train) else 0,
        'positiveRateTest': round(float(y_test.mean()), 4) if len(y_test) else 0,
        'testAuc': None,
        'testBrier': None,
        'importances': dict(zip(FEATURES, [round(float(v), 4) for v in model.feature_importances_])),
    }
    if len(set(y_test)) > 1:
        report['testAuc'] = round(float(roc_auc_score(y_test, scores)), 4)
        report['testBrier'] = round(float(brier_score_loss(y_test, scores)), 4)
    return model, report


def fit_places_regressor(train: list[dict], test: list[dict]):
    usable_train = [row for row in train if row.get('placesToFlag') is not None]
    usable_test = [row for row in test if row.get('placesToFlag') is not None]
    x_train = np.array([[float(row[name]) for name in FEATURES] for row in usable_train], dtype=float)
    y_train = np.array([float(row['placesToFlag']) for row in usable_train], dtype=float)
    x_test = np.array([[float(row[name]) for name in FEATURES] for row in usable_test], dtype=float)
    y_test = np.array([float(row['placesToFlag']) for row in usable_test], dtype=float)
    model = RandomForestRegressor(
        n_estimators=220,
        min_samples_leaf=14,
        max_depth=10,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test) if len(y_test) else np.array([])
    report = {
        'samplesTrain': int(len(y_train)),
        'samplesTest': int(len(y_test)),
        'testMaePlaces': None if not len(y_test) else round(float(mean_absolute_error(y_test, pred)), 3),
        'trainMeanPlaces': round(float(y_train.mean()), 3) if len(y_train) else 0,
        'importances': dict(zip(FEATURES, [round(float(v), 4) for v in model.feature_importances_])),
    }
    return model, report


def _race_bucket():
    return {'laps': 0, 'overtakes': 0, 'recoveries': 0, 'loseWindows': 0}


def prior_table(train: list[dict]) -> dict:
    drivers = defaultdict(lambda: {'overtakes': 0, 'recoveries': 0, 'loseWindows': 0, 'laps': 0, 'races': set(), 'byRace': defaultdict(_race_bucket)})
    tracks = defaultdict(lambda: {'overtakes': 0, 'laps': 0, 'races': set(), 'byRace': defaultdict(_race_bucket)})
    for row in train:
        race_id = f"{row['year']}|{row['location']}"
        driver_row = drivers[row['driver']]
        track_row = tracks[row['location']]
        driver_row['laps'] += 1
        driver_row['overtakes'] += row['overtakeSoon']
        driver_row['races'].add((row['year'], row['location']))
        driver_row['byRace'][race_id]['laps'] += 1
        driver_row['byRace'][race_id]['overtakes'] += row['overtakeSoon']
        track_row['laps'] += 1
        track_row['overtakes'] += row['overtakeSoon']
        track_row['races'].add((row['year'], row['location']))
        track_row['byRace'][race_id]['laps'] += 1
        track_row['byRace'][race_id]['overtakes'] += row['overtakeSoon']
        if row['recoverIfLost'] is not None:
            driver_row['loseWindows'] += 1
            driver_row['recoveries'] += row['recoverIfLost']
            driver_row['byRace'][race_id]['loseWindows'] += 1
            driver_row['byRace'][race_id]['recoveries'] += row['recoverIfLost']
    def pack(store, include_recover=False):
        out = {}
        for key, item in store.items():
            laps = max(1, item['laps'])
            packed = {
                'laps': item['laps'],
                'races': len(item['races']),
                'overtakes': item['overtakes'],
                'overtakeSoonRate': round(item['overtakes'] / laps, 4),
                'byRace': dict(item['byRace']),
            }
            if include_recover and item.get('loseWindows'):
                packed['recoveries'] = item['recoveries']
                packed['loseWindows'] = item['loseWindows']
                packed['recoverIfLostRate'] = round(item['recoveries'] / item['loseWindows'], 4)
            out[key] = packed
        return out
    return {'drivers': pack(drivers, include_recover=True), 'tracks': pack(tracks)}


def fit_drs_models(train_windows: list[dict], test_windows: list[dict], recipes: dict) -> tuple[dict, dict]:
    train = expand_drs_samples(train_windows, recipes)
    test = expand_drs_samples(test_windows, recipes)
    classifiers = {}
    reports = {}
    for target in ('convertWithin1', 'convertWithin2', 'convertWithin3'):
        x_train = np.array([[float(row[name]) for name in DRS_FEATURES] for row in train], dtype=float)
        y_train = np.array([int(row[target]) for row in train], dtype=int)
        x_test = np.array([[float(row[name]) for name in DRS_FEATURES] for row in test], dtype=float)
        y_test = np.array([int(row[target]) for row in test], dtype=int)
        model = RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=10,
            max_depth=10,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        classifiers[target] = model
        report = {
            'samplesTrain': int(len(y_train)),
            'samplesTest': int(len(y_test)),
            'positiveRateTrain': round(float(y_train.mean()), 4) if len(y_train) else 0,
            'positiveRateTest': round(float(y_test.mean()), 4) if len(y_test) else 0,
            'testAuc': None,
            'testBrier': None,
        }
        if len(y_test) and len(set(y_test)) > 1:
            proba = model.predict_proba(x_test)
            positive = list(model.classes_).index(1) if 1 in model.classes_ else 0
            scores = proba[:, positive] if proba.shape[1] > 1 else proba[:, 0]
            report['testAuc'] = round(float(roc_auc_score(y_test, scores)), 4)
            report['testBrier'] = round(float(brier_score_loss(y_test, scores)), 4)
        reports[target] = report
    wait_train = [row for row in train if row.get('remainingWait') is not None]
    wait_test = [row for row in test if row.get('remainingWait') is not None]
    wait_model = None
    wait_report = {'samplesTrain': len(wait_train), 'samplesTest': len(wait_test), 'testMaeLaps': None}
    if len(wait_train) >= 40:
        wait_model = RandomForestRegressor(
            n_estimators=220,
            min_samples_leaf=8,
            max_depth=10,
            random_state=SEED,
            n_jobs=-1,
        )
        xw = np.array([[float(row[name]) for name in DRS_FEATURES] for row in wait_train], dtype=float)
        yw = np.array([float(row['remainingWait']) for row in wait_train], dtype=float)
        wait_model.fit(xw, yw)
        if wait_test:
            xt = np.array([[float(row[name]) for name in DRS_FEATURES] for row in wait_test], dtype=float)
            yt = np.array([float(row['remainingWait']) for row in wait_test], dtype=float)
            wait_report['testMaeLaps'] = round(float(mean_absolute_error(yt, wait_model.predict(xt))), 3)
    reports['remainingWait'] = wait_report
    return {
        'schema': 'drs-wait.v1',
        'features': DRS_FEATURES,
        'classifiers': classifiers,
        'waitRegressor': wait_model,
    }, reports


def main() -> None:
    random.seed(SEED)
    races = [(key, payload) for key, payload in iter_races() if key[0] in HISTORY_YEARS]
    if len(races) < 8:
        raise SystemExit(json.dumps({'error': f'need more 2018-2025 cached races, found {len(races)}'}))
    keys = [key for key, _ in races]
    test_n = max(2, int(len(keys) * TEST_RACE_FRACTION))
    test_keys = set(random.sample(keys, test_n))
    train_rows, test_rows = [], []
    train_windows, test_windows = [], []
    for index, (key, payload) in enumerate(races, start=1):
        print(f'building samples {index}/{len(races)} {key}', file=sys.stderr, flush=True)
        field = build_field_story(payload)
        event = payload.get('event') or {}
        rows = samples_from_story(field, event)
        windows = extract_windows(field, event)
        if key in test_keys:
            test_rows.extend(rows)
            test_windows.extend(windows)
        else:
            train_rows.extend(rows)
            train_windows.extend(windows)
    recipes = pack_recipes(train_windows)
    attach_priors(train_rows, train_rows, recipes)
    attach_priors(test_rows, train_rows, recipes)
    models = {}
    reports = {}
    for target in TARGETS:
        models[target], reports[target] = fit_target(train_rows, test_rows, target)
    drs_models, drs_reports = fit_drs_models(train_windows, test_windows, recipes)
    places_model, places_report = fit_places_regressor(train_rows, test_rows)
    lose_model, lose_report = fit_target(train_rows, test_rows, 'loseSoon')
    MODELS.mkdir(parents=True, exist_ok=True)
    artifact = {
        'schema': 'driver-recommend.v4',
        'features': FEATURES,
        'targets': list(TARGETS),
        'models': models,
        'outcomeModels': {
            'placesToFlag': places_model,
            'loseSoon': lose_model,
        },
        'priors': prior_table(train_rows),
        'drsRecipes': recipes,
        'drsModels': drs_models,
        'global': {
            'overtakeSoonRate': reports['overtakeSoon']['positiveRateTrain'],
            'recoverIfLostRate': reports['recoverIfLost']['positiveRateTrain'],
            'pushHelpsRate': reports['pushHelps']['positiveRateTrain'],
            'drsEfficiency': recipes.get('efficiency'),
        },
    }
    joblib.dump(artifact, MODELS / 'recommend_rf.joblib')
    report = {
        'schema': 'driver-recommend.v4',
        'years': f'{min(HISTORY_YEARS)}-{max(HISTORY_YEARS)}',
        'racesTotal': len(keys),
        'racesTrain': len(keys) - len(test_keys),
        'racesTest': len(test_keys),
        'testRaces': sorted(f'{year}-R{round_}' for year, round_ in test_keys),
        'samplesTrain': len(train_rows),
        'samplesTest': len(test_rows),
        'drs': {
            'windows': recipes.get('windows'),
            'converts': recipes.get('converts'),
            'efficiency': recipes.get('efficiency'),
            'drivers': {
                code: {
                    'windows': item.get('windows'),
                    'efficiency': item.get('efficiency'),
                    'waitLapsP50': item.get('waitLapsP50'),
                    'waitLapsP25': item.get('waitLapsP25'),
                    'waitLapsP75': item.get('waitLapsP75'),
                }
                for code, item in (recipes.get('drivers') or {}).items()
            },
        },
        'targets': reports,
        'drsModel': drs_reports,
        'outcome': {
            'placesToFlag': places_report,
            'loseSoon': lose_report,
        },
        'note': 'Trained on 2018-2025 cached races. DRS wait is a RandomForest. placesToFlag regresses current running order minus classified finish. loseSoon is a place-loss in the next 2 laps. Not a rewritten result. Race-level holdout.',
    }
    (MODELS / 'recommend_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
