"""Evaluate downstream position durability on the frozen 2026 holdout.

Classification accuracy answers whether a label matches the derived outcome
label.  This report answers the separate product question: when a model makes
an action recommendation at an observed immediate on-track pass, how many
laps ahead does the deterministic tactical replay estimate for that action,
and how does that compare with the real observed persistence?

Only completed 2026 rows are used here.  They are never used to fit or tune a
model.  The replay remains a modelled counterfactual: it does not claim that
the actual race would have changed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fetch_f1_session import CACHE_DIR
from replay_strategy import build_tree
from train_overtake_model import FEATURES, DECISION_SCHEMA_VERSION, load_rows, to_frame


ROOT = pathlib.Path(CACHE_DIR).parent
DECISION_DIR = ROOT / 'decision-points'
MODELS = ROOT / 'models'
REPORT_PATH = MODELS / 'overtake_report.json'
LABELS = ('SAVE', 'DELAY', 'ATTACK')


def clean(value: Any) -> Any:
    """Convert numpy scalars into JSON-safe values for the replay payload."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def metadata_by_race() -> dict[tuple[int, int, str], dict[str, Any]]:
    metadata = {}
    for path in sorted(DECISION_DIR.glob('2026/*.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get('schemaVersion') != DECISION_SCHEMA_VERSION:
            continue
        rows = payload.get('rows') or []
        sample = rows[0] if rows else {}
        key = (int(sample.get('year', 2026)), int(sample.get('round', path.stem.split('_')[0])),
               str(sample.get('session', 'R')))
        metadata[key] = {
            'eventName': payload.get('eventName') or f'Round {key[1]}',
            'totalLaps': int(payload.get('totalLaps') or 0),
            'holdLaps': int(payload.get('holdLaps') or 6),
            'finishPositions': payload.get('finishPositions') or {},
        }
    return metadata


def energy_laps(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build a compact pre-decision SoC trace from both-car surrogate fields."""
    values: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        lap = int(float(row.get('lap') or 0))
        driver = str(row.get('driver') or '')
        defender = str(row.get('defender') or '')
        if driver and row.get('attackerSoCMj') is not None:
            values[driver][lap] = float(row['attackerSoCMj'])
        if defender and row.get('defenderSoCMj') is not None:
            values[defender][lap] = float(row['defenderSoCMj'])
    return {
        driver: [{'lap': lap, 'socEndMj': soc} for lap, soc in sorted(laps.items())]
        for driver, laps in values.items()
    }


def pair_key(driver: Any, defender: Any) -> tuple[str, str]:
    """Use one key whether a pair is attacking or defending."""
    return tuple(sorted((str(driver or ''), str(defender or ''))))


def prepare_replay_inputs(race_frames: dict[tuple[int, int, str], pd.DataFrame]) -> dict[tuple[int, int, str], dict[str, Any]]:
    """Prepare model-independent race state once for all benchmark models."""
    prepared = {}
    for key, race_df in race_frames.items():
        records = []
        pair_rows: dict = defaultdict(list)
        for index, series in race_df.iterrows():
            row = clean(series.to_dict())
            records.append(row)
            pair_rows[pair_key(row.get('driver'), row.get('defender'))].append((index, row))
        prepared[key] = {
            'energyLaps': energy_laps(records),
            'pairRows': pair_rows,
        }
    return prepared


def branch_expected_laps(child: dict[str, Any]) -> float:
    """Follow the best continuation after a forced root action."""
    node = child.get('next') or {}
    expected = float(child.get('leadLaps') or 0.0)
    while node.get('children'):
        next_child = next((item for item in node['children'] if item.get('best')), node['children'][0])
        expected = float(next_child.get('leadLaps') or expected)
        node = next_child.get('next') or {}
    return expected


def branch_durability(child: dict[str, Any], observed_laps: float,
                      horizons: tuple[int, ...] = (1, 2, 3, 5, 6)) -> list[dict[str, Any]]:
    """Read durability probabilities from a forced first action then best continuation."""
    steps = {1: child}
    cursor = child.get('next') or {}
    step_number = 1
    while cursor.get('children'):
        step_number += 1
        selected = next((item for item in cursor['children'] if item.get('best')),
                        cursor['children'][0])
        steps[step_number] = selected
        cursor = selected.get('next') or {}
    return [{
        'horizon': horizon,
        'estimatedProbability': round(float(steps[horizon]['aheadProbability']), 4)
        if horizon in steps else None,
        'observedAhead': bool(observed_laps >= horizon),
    } for horizon in horizons]


def bootstrap_mean(values_by_race: dict[str, list[float]], samples: int = 1000) -> dict[str, Any]:
    groups = [np.asarray(values, dtype=float) for values in values_by_race.values() if values]
    if not groups:
        return {'method': 'race bootstrap percentile interval', 'samples': 0,
                'races': 0, 'lower95': None, 'upper95': None}
    rng = np.random.default_rng(29)
    scores = []
    for _ in range(samples):
        selected = rng.integers(0, len(groups), size=len(groups))
        values = np.concatenate([groups[index] for index in selected])
        scores.append(float(values.mean()))
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return {
        'method': 'race bootstrap percentile interval',
        'samples': samples,
        'races': len(groups),
        'lower95': round(float(lower), 4),
        'upper95': round(float(upper), 4),
    }


def summarize_model(name: str, model: Any, test_df: pd.DataFrame,
                    replay_inputs: dict[tuple[int, int, str], dict[str, Any]],
                    metadata: dict[tuple[int, int, str], dict[str, Any]]) -> dict[str, Any]:
    x_test = test_df[FEATURES].to_numpy()
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    classes = list(model.classes_) if hasattr(model, 'classes_') else list(model.named_steps['classifier'].classes_)
    class_index = {label: index for index, label in enumerate(classes)}

    # Keep model predictions attached by the original dataframe index so the
    # same prediction is used at the focus row and at later replay rows.
    prediction_by_index = {}
    for index, label, row_proba in zip(test_df.index, predictions, probabilities):
        prediction_by_index[index] = {
            'label': str(label),
            'probabilities': {label_name: float(row_proba[class_index[label_name]])
                              for label_name in classes},
        }

    focus_df = test_df[test_df['passedNow'].fillna(False).astype(bool)].copy()
    race_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    driver_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    horizon_results: dict[int, list[dict[str, Any]]] = defaultdict(list)
    action_counts = Counter()
    scored = []
    for index, focus_series in focus_df.iterrows():
        key = (int(focus_series['year']), int(focus_series['round']), str(focus_series['session']))
        race_meta = metadata.get(key)
        replay_input = replay_inputs.get(key)
        if not race_meta or replay_input is None:
            continue
        focus = clean(focus_series.to_dict())
        focus['pred'] = prediction_by_index[index]
        pair_rows = [
            {**row, 'pred': prediction_by_index.get(pair_index)}
            for pair_index, row in replay_input['pairRows'].get(
                pair_key(focus['driver'], focus['defender']), [])
            if float(row.get('lap') or 0.0) >= float(focus['lap'])
        ]
        if not pair_rows:
            pair_rows = [focus]
        payload = {
            'focus': focus,
            'rows': pair_rows,
            'totalLaps': race_meta['totalLaps'],
            'holdLaps': race_meta['holdLaps'],
            'finishPositions': race_meta['finishPositions'],
            'energyLaps': replay_input['energyLaps'],
            'year': key[0],
            'regulationEra': '2026',
            # The evaluation needs only the base tree. Interactive requests
            # include SoC sensitivity; omitting it here keeps the frozen report
            # quick and does not change any score.
            'includeSensitivity': False,
        }
        tree = build_tree(payload)
        model_action = str(prediction_by_index[index]['label'])
        root_children = tree.get('tree', {}).get('children') or []
        selected_child = next((child for child in root_children if child.get('action') == model_action), None)
        if selected_child is None:
            continue
        observed_raw = float(focus.get('observedLeadLaps') or 0.0)
        observed = min(observed_raw, float(tree.get('horizon') or race_meta['holdLaps']))
        model_expected = branch_expected_laps(selected_child)
        best_expected = float(tree.get('expectedLeadLaps') or 0.0)
        record = {
            'round': key[1],
            'eventName': race_meta['eventName'],
            'lap': int(float(focus['lap'])),
            'driver': focus['driver'],
            'defender': focus['defender'],
            'modelAction': model_action,
            'treeBestAction': (tree.get('path') or [{}])[0].get('action'),
            'observedLeadLaps': round(observed, 3),
            'observedLeadLapsRaw': round(observed_raw, 3),
            'modelExpectedLeadLaps': round(model_expected, 3),
            'bestTreeExpectedLeadLaps': round(best_expected, 3),
            'modelVsObservedDelta': round(model_expected - observed, 3),
            'bestTreeVsObservedDelta': round(best_expected - observed, 3),
            'durabilityByHorizon': branch_durability(selected_child, observed),
        }
        action_counts[model_action] += 1
        scored.append(record)
        race_results[race_meta['eventName']].append(record)
        driver_results[str(record['driver'])].append(record)
        for horizon_result in record['durabilityByHorizon']:
            horizon_results[int(horizon_result['horizon'])].append(horizon_result)

    deltas = [item['modelVsObservedDelta'] for item in scored]
    best_deltas = [item['bestTreeVsObservedDelta'] for item in scored]
    result = {
        'modelType': name,
        'rows': len(scored),
        'actionCounts': dict(action_counts),
        'meanObservedLeadLaps': round(float(np.mean([item['observedLeadLaps'] for item in scored])), 4) if scored else None,
        'meanModelExpectedLeadLaps': round(float(np.mean([item['modelExpectedLeadLaps'] for item in scored])), 4) if scored else None,
        'meanModelVsObservedDelta': round(float(np.mean(deltas)), 4) if deltas else None,
        'modelEstimatedBetterRate': round(float(np.mean([delta > 0 for delta in deltas])), 4) if deltas else None,
        'meanBestTreeExpectedLeadLaps': round(float(np.mean([item['bestTreeExpectedLeadLaps'] for item in scored])), 4) if scored else None,
        'meanBestTreeVsObservedDelta': round(float(np.mean(best_deltas)), 4) if best_deltas else None,
        'modelDeltaRace95': bootstrap_mean({race: [item['modelVsObservedDelta'] for item in items]
                                             for race, items in race_results.items()}),
        'byRace': [],
        'byDriver': [],
        'byHorizon': [],
    }
    for event_name, items in race_results.items():
        result['byRace'].append({
            'eventName': event_name,
            'round': items[0]['round'],
            'rows': len(items),
            'meanObservedLeadLaps': round(float(np.mean([item['observedLeadLaps'] for item in items])), 4),
            'meanModelExpectedLeadLaps': round(float(np.mean([item['modelExpectedLeadLaps'] for item in items])), 4),
            'meanModelVsObservedDelta': round(float(np.mean([item['modelVsObservedDelta'] for item in items])), 4),
            'modelEstimatedBetterRate': round(float(np.mean([item['modelVsObservedDelta'] > 0 for item in items])), 4),
        })
    result['byRace'].sort(key=lambda item: item['round'])
    for driver, items in driver_results.items():
        result['byDriver'].append({
            'driver': driver,
            'rows': len(items),
            'meanObservedLeadLaps': round(float(np.mean([item['observedLeadLaps'] for item in items])), 4),
            'meanModelExpectedLeadLaps': round(float(np.mean([item['modelExpectedLeadLaps'] for item in items])), 4),
            'meanModelVsObservedDelta': round(float(np.mean([item['modelVsObservedDelta'] for item in items])), 4),
        })
    result['byDriver'].sort(key=lambda item: (-item['rows'], item['driver']))
    for horizon, items in horizon_results.items():
        estimates = [item['estimatedProbability'] for item in items if item['estimatedProbability'] is not None]
        observed_rate = float(np.mean([item['observedAhead'] for item in items])) if items else None
        estimate_rate = float(np.mean(estimates)) if estimates else None
        result['byHorizon'].append({
            'horizon': horizon,
            'rows': len(items),
            'observedHoldRate': round(observed_rate, 4) if observed_rate is not None else None,
            'meanEstimatedProbability': round(estimate_rate, 4) if estimate_rate is not None else None,
            'probabilityDelta': round(estimate_rate - observed_rate, 4)
            if estimate_rate is not None and observed_rate is not None else None,
        })
    result['byHorizon'].sort(key=lambda item: item['horizon'])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2026)
    args = parser.parse_args()
    if args.year != 2026:
        raise SystemExit('This evaluator is intentionally frozen to the unseen 2026 holdout.')

    records = load_rows()
    frame = to_frame(records)
    test_df = frame[frame['year'] == args.year].copy()
    if test_df.empty:
        raise SystemExit('no completed 2026 decision points available')
    metadata = metadata_by_race()
    race_frames = {
        key: group.copy() for key, group in test_df.groupby(['year', 'round', 'session'], sort=True)
    }
    replay_inputs = prepare_replay_inputs(race_frames)

    artifact_names = {
        'RandomForestClassifier': 'overtake_rf.joblib',
        'RandomForestSigmoidCalibrated': 'overtake_rf_sigmoid.joblib',
        'LogisticRegression': 'overtake_logistic.joblib',
        'GradientBoostingClassifier': 'overtake_gradientboost.joblib',
    }
    models = {}
    for name, filename in artifact_names.items():
        path = MODELS / filename
        if not path.exists():
            continue
        artifact = joblib.load(path)
        models[name] = artifact['model']
    if not models:
        raise SystemExit('no trained model artifacts available')

    persistence = {
        'schemaVersion': 'replay-evaluation.v3',
        'evaluationUnit': 'completed 2026 immediate on-track pass decision points',
        'holdout': {
            'year': args.year,
            'trainingYears': 'earlier seasons only',
            'trainingOrTuningOnHoldout': False,
            'treeHorizonLaps': 6,
            'rowsAvailable': int(len(test_df)),
        },
        'interpretation': 'Expected laps are deterministic model estimates under the tactical replay assumptions; they are not measured alternate-race outcomes.',
        'models': {},
    }
    for name, model in models.items():
        persistence['models'][name] = summarize_model(name, model, test_df, replay_inputs, metadata)

    report = json.loads(REPORT_PATH.read_text(encoding='utf-8')) if REPORT_PATH.exists() else {}
    report['replayPersistence'] = persistence
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({
        'schemaVersion': persistence['schemaVersion'],
        'holdout': persistence['holdout'],
        'models': {name: {
            'rows': result['rows'],
            'meanObservedLeadLaps': result['meanObservedLeadLaps'],
            'meanModelExpectedLeadLaps': result['meanModelExpectedLeadLaps'],
            'meanModelVsObservedDelta': result['meanModelVsObservedDelta'],
            'modelEstimatedBetterRate': result['modelEstimatedBetterRate'],
        } for name, result in persistence['models'].items()},
    }, indent=2))


if __name__ == '__main__':
    main()
