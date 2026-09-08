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
from overtake_feature_schema import COMPOUND_ORDINAL, FEATURE_SCHEMA_VERSION

ROOT = pathlib.Path(CACHE_DIR).parent
MODEL_PATH = ROOT / 'models' / 'overtake_rf.joblib'

IMPORTANT_INPUTS = (
    'gapS', 'closingRateS', 'speedDeltaKph', 'tyreAgeDiff', 'position',
    'attackerSoCMj', 'defenderSoCMj', 'trafficAheadCount',
    'trafficBehindCount', 'slipstreamProxy', 'dirtyAirRisk',
)


def _num(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value == value else default


def _weather(row, field, default=0.0):
    weather = row.get('weather')
    return _num(weather.get(field), default) if isinstance(weather, dict) else default


def input_state(row):
    """Report input completeness without refusing a useful observed battle.

    The fitted pipeline has neutral defaults for incomplete public data. Those
    defaults are retained for backwards-compatible inference, but the caller
    must be able to distinguish a full decision-time state from a partial one.
    """
    missing = []
    for field in IMPORTANT_INPUTS:
        value = row.get(field)
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            missing.append(field)
    return {
        'status': 'COMPLETE' if not missing else 'PARTIAL',
        'missingImportantInputs': missing,
        'usesNeutralFallback': bool(missing),
    }


def live_command_gate(row, completeness):
    """Keep retrospective classification distinct from a live FIA command.

    The historical model can score a causal race-state record, but a 2026 live
    Overtake Mode recommendation additionally needs an exact track-zone
    identity and the cited event appendix. Neither may be inferred from the
    generic close-battle filter. This gate does not change probabilities.
    """
    year = int(_num(row.get('year'), 0))
    if year < 2026:
        return {
            'status': 'HISTORICAL_REPLAY_ONLY',
            'liveCommandEligible': False,
            'blockedBy': ['2026 Overtake Mode does not apply to this historical regulation era'],
            'note': 'Valid for retrospective analysis only; not a current-regulation race command.',
        }
    rule_context = row.get('ruleContext') if isinstance(row.get('ruleContext'), dict) else {}
    blocked = []
    if not row.get('zoneId'):
        blocked.append('exact track-zone identity is not loaded')
    if not rule_context.get('eventSpecificDataLoaded'):
        blocked.append('cited FIA event Overtake Mode appendix is not loaded')
    if completeness.get('status') != 'COMPLETE':
        blocked.append('complete two-car decision-state input is not available')
    if bool(row.get('pitDistorted')):
        blocked.append('pit-cycle context is active')
    if row.get('attackerTrackStatus') not in (None, '1') or row.get('defenderTrackStatus') not in (None, '1'):
        blocked.append('non-green race-control state is active')
    return {
        'status': 'LIVE_COMMAND_READY' if not blocked else 'ANALYSIS_ONLY',
        'liveCommandEligible': not blocked,
        'blockedBy': blocked,
        'note': ('All configured live-command gates are satisfied.' if not blocked else
                 'Prediction remains an analysis/replay estimate; it is not a live Overtake Mode command.'),
    }


def choose_action(probabilities, classes, action_policy):
    """Apply the historically selected decision weights, if present."""
    weights = (action_policy or {}).get('weights') or {}
    factors = np.asarray([float(weights.get(label, 1.0)) for label in classes])
    return classes[int(np.argmax(probabilities * factors))]


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
        'airTempC': _weather(row, 'airTempC'),
        'trackTempC': _weather(row, 'trackTempC'),
        'humidityPct': _weather(row, 'humidityPct'),
        'windSpeedMps': _weather(row, 'windSpeedMps'),
        'rainfall': _weather(row, 'rainfall'),
        'weatherMissing': 0.0 if isinstance(row.get('weather'), dict) else 1.0,
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
    return np.array([[vector[name] for name in features]], dtype=float), input_state(row)


def main():
    try:
        artifact = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(json.dumps({'error': 'model not trained yet', 'modelPath': str(MODEL_PATH)}))
        return

    if artifact.get('featureSchemaVersion') != FEATURE_SCHEMA_VERSION:
        print(json.dumps({
            'error': 'trained model feature schema is stale',
            'expectedFeatureSchemaVersion': FEATURE_SCHEMA_VERSION,
            'artifactFeatureSchemaVersion': artifact.get('featureSchemaVersion'),
            'nextStep': 'retrain with train_overtake_model.py after cache schema audit',
        }))
        return
    model, features, classes = artifact['model'], artifact['features'], artifact['classes']
    action_policy = artifact.get('actionPolicy')

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
        x, completeness = build_feature_vector(row, features)
        proba = model.predict_proba(x)[0]
        probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        label = choose_action(proba, classes, action_policy)
        year = int(_num(row.get('year'), 2025))
        frac = _num(row.get('lapFraction'), 0.5)
        a = car_mass_kg(year, row.get('driver'), frac)
        d = car_mass_kg(year, row.get('defender'), frac)
        predictions.append({'label': label, 'probabilities': probs,
                            'inputState': completeness,
                            'liveCommandGate': live_command_gate(row, completeness),
                            'mass': {'attackerKg': round(a, 1),
                                     'defenderKg': round(d, 1),
                                     'deltaKg': round(a - d, 1)}})

    print(json.dumps({'modelType': 'RandomForestClassifier', 'featureSchemaVersion': FEATURE_SCHEMA_VERSION, 'classes': classes,
                      'actionPolicy': action_policy,
                      'predictions': predictions, 'rows': len(predictions)}))


if __name__ == '__main__':
    main()
