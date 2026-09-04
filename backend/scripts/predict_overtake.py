"""Serve PitWolf overtake-feasibility predictions.

Loads the trained RandomForest from data/f1-cache/models/overtake_rf.joblib and
scores one or more decision points, returning P(SAVE)/P(DELAY)/P(ATTACK) plus
the recommended label. Reads a JSON document on stdin and prints JSON on stdout.

Input shapes accepted:
  {"features": {gapS, closingRateS, speedDeltaKph, tyreAgeDiff, lapFraction,
                position, raceMeanSpeedKph, attackerCompound, defenderCompound}}
  {"rows": [ ...same feature objects... ]}
  [ ...same feature objects... ]

Missing numeric features fall back to 0 (matching training-time fills). Compound
strings are mapped to the same ordinal used in training; drsEligible is derived
from gapS <= 1.0 exactly as in train_overtake_model.py.
"""

import json
import pathlib
import sys

import joblib
import numpy as np

from car_mass import car_mass_kg
from fetch_f1_session import CACHE_DIR

ROOT = pathlib.Path(CACHE_DIR).parent
MODEL_PATH = ROOT / 'models' / 'overtake_rf.joblib'

COMPOUND_ORDINAL = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}


def _num(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value == value else default


def build_feature_vector(row, features):
    gap = _num(row.get('gapS'))
    vector = {
    'gapS': gap,
        'closingRateS': _num(row.get('closingRateS')),
        'speedDeltaKph': _num(row.get('speedDeltaKph')),
        'tyreAgeDiff': _num(row.get('tyreAgeDiff')),
        'lapFraction': _num(row.get('lapFraction'), 0.5),
        'position': _num(row.get('position')),
        'raceMeanSpeedKph': _num(row.get('raceMeanSpeedKph')),
        'attackerCompoundOrd': COMPOUND_ORDINAL.get(str(row.get('attackerCompound')).upper(), 1.0),
        'defenderCompoundOrd': COMPOUND_ORDINAL.get(str(row.get('defenderCompound')).upper(), 1.0),
        'drsEligible': 1.0 if gap <= 1.0 else 0.0,
        'trafficAheadCount': _num(row.get('trafficAheadCount')),
        'trafficBehindCount': _num(row.get('trafficBehindCount')),
        'packDensity': _num(row.get('packDensity')),
        'slipstreamProxy': _num(row.get('slipstreamProxy')),
        'dirtyAirRisk': _num(row.get('dirtyAirRisk')),
        'attackerTyreDegProxy': _num(row.get('attackerTyreDegProxy')),
        'defenderTyreDegProxy': _num(row.get('defenderTyreDegProxy')),
    }
    # These fields are supplied by the historical extractor after applying the
    # cheap lap-time SoC surrogate.  Keep the neutral 70% state for older cache
    # rows so prediction remains backwards compatible during re-extraction.
    vector['attackerSoCMj'] = _num(row.get('attackerSoCMj'), 2.8)
    vector['defenderSoCMj'] = _num(row.get('defenderSoCMj'), 2.8)
    vector['energyDeltaMj'] = _num(row.get('energyDeltaMj'), vector['attackerSoCMj'] - vector['defenderSoCMj'])
    year = int(_num(row.get('year'), 2025))
    frac = _num(row.get('lapFraction'), 0.5)
    attacker_mass = car_mass_kg(year, row.get('driver'), frac)
    vector['attackerMassKg'] = attacker_mass
    vector['massDeltaKg'] = attacker_mass - car_mass_kg(year, row.get('defender'), frac)
    return np.array([[vector[name] for name in features]], dtype=float)


def main():
    try:
        artifact = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(json.dumps({'error': 'model not trained yet', 'modelPath': str(MODEL_PATH)}))
        return

    model, features, classes = artifact['model'], artifact['features'], artifact['classes']

    payload = json.loads(sys.stdin.read() or '{}')
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and 'rows' in payload:
        rows = payload['rows']
    elif isinstance(payload, dict) and 'features' in payload:
        rows = [payload['features']]
    else:
        rows = [payload]
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        print(json.dumps({'error': 'no feature rows supplied'}))
        return

    predictions = []
    for row in rows:
        x = build_feature_vector(row, features)
        proba = model.predict_proba(x)[0]
        probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        label = classes[int(np.argmax(proba))]
        year = int(_num(row.get('year'), 2025))
        frac = _num(row.get('lapFraction'), 0.5)
        a = car_mass_kg(year, row.get('driver'), frac)
        d = car_mass_kg(year, row.get('defender'), frac)
        predictions.append({'label': label, 'probabilities': probs,
                            'mass': {'attackerKg': round(a, 1),
                                     'defenderKg': round(d, 1),
                                     'deltaKg': round(a - d, 1)}})

    print(json.dumps({'modelType': 'RandomForestClassifier', 'classes': classes,
                      'predictions': predictions, 'rows': len(predictions)}))


if __name__ == '__main__':
    main()
