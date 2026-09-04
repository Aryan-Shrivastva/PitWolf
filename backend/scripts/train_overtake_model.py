"""Train the PitWolf overtake-feasibility classifier.

Reads decision-point JSONs from data/f1-cache/decision-points/, keeps rows
that represent genuine on-track battles, engineers features, applies a strict
temporal split (train on earlier seasons, test on later ones — 2026 is held
out entirely per the competition methodology), and fits a RandomForest.

Artifacts written under data/f1-cache/models/:
- overtake_rf.joblib      (fitted model + feature names)
- overtake_report.json    (metrics, feature importances, class priors —
                           consumed by the dashboard)
"""

import argparse
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from car_mass import car_mass_kg
from energy_surrogate import add_surrogate_energy
from fetch_f1_session import CACHE_DIR

ROOT = pathlib.Path(CACHE_DIR).parent
MODELS = ROOT / 'models'

FEATURES = [
    'gapS', 'closingRateS', 'speedDeltaKph', 'tyreAgeDiff', 'lapFraction',
    'position', 'raceMeanSpeedKph', 'attackerCompoundOrd', 'defenderCompoundOrd',
    'drsEligible', 'attackerMassKg', 'massDeltaKg', 'attackerSoCMj',
    'defenderSoCMj', 'energyDeltaMj', 'trafficAheadCount',
    'trafficBehindCount', 'packDensity', 'slipstreamProxy', 'dirtyAirRisk',
    'attackerTyreDegProxy', 'defenderTyreDegProxy',
]
LABELS = ['SAVE', 'DELAY', 'ATTACK']
DECISION_SCHEMA_VERSION = 'decision-point.v5'
GAP_ATTACK_THRESHOLD_S = 0.70
GAP_DELAY_THRESHOLD_S = 1.20

COMPOUND_ORDINAL = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}


def load_rows():
    records = []
    for path in sorted(ROOT.glob('decision-points/*/*.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get('schemaVersion') != DECISION_SCHEMA_VERSION:
            continue
        event_name = payload.get('eventName')
        for row in payload.get('rows', []):
            record = dict(row)
            # Keep the cache's human-readable event name alongside each row so
            # aggregate holdout metrics can be reported by race, not only by R#.
            record['_eventName'] = event_name
            records.append(record)
    return records


def to_frame(records):
    df = pd.DataFrame(records)
    if df.empty:
        return df
    keep = (~df['pitDistorted']) & df['defenderActive'] & df['gapS'].notna()
    if 'eligibleForTraining' in df:
        keep &= df['eligibleForTraining'].fillna(False).astype(bool)
    df = df[keep].copy()
    df['closingRateS'] = df['closingRateS'].fillna(0.0)
    df['speedDeltaKph'] = df['speedDeltaKph'].fillna(0.0)
    df['tyreAgeDiff'] = df['tyreAgeDiff'].fillna(0.0)
    for column in ('trafficAheadCount', 'trafficBehindCount', 'packDensity',
                   'slipstreamProxy', 'dirtyAirRisk', 'attackerTyreDegProxy',
                   'defenderTyreDegProxy'):
        if column not in df:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0.0)
    # Missing values occur only when no completed prior lap was available.
    # Use a fixed sentinel rather than the dataset mean, which would make a
    # future season influence an earlier row during preprocessing.
    df['raceMeanSpeedKph'] = df['raceMeanSpeedKph'].fillna(0.0)
    df['attackerCompoundOrd'] = df['attackerCompound'].map(COMPOUND_ORDINAL).fillna(1.0)
    df['defenderCompoundOrd'] = df['defenderCompound'].map(COMPOUND_ORDINAL).fillna(1.0)
    df['drsEligible'] = (df['gapS'] <= 1.0).astype(float)
    df['attackerMassKg'] = [
        car_mass_kg(y, drv, lf) for y, drv, lf in
        zip(df['year'], df['driver'], df['lapFraction'])]
    df['massDeltaKg'] = [
        car_mass_kg(y, drv, lf) - car_mass_kg(y, dfd, lf)
        for y, drv, dfd, lf in
        zip(df['year'], df['driver'], df['defender'], df['lapFraction'])]
    return add_surrogate_energy(df)


def gap_only_actions(frame):
    """Transparent baseline using only the observed gap at the cutoff."""
    gap = frame['gapS'].to_numpy()
    return np.select(
        [gap <= GAP_ATTACK_THRESHOLD_S, gap <= GAP_DELAY_THRESHOLD_S],
        ['ATTACK', 'DELAY'],
        default='SAVE',
    )


def calibration_metrics(y_true, pred, proba, classes):
    """Return compact, deterministic multiclass calibration diagnostics."""
    class_to_index = {name: i for i, name in enumerate(classes)}
    truth = np.zeros_like(proba, dtype=float)
    for i, label in enumerate(y_true):
        if label in class_to_index:
            truth[i, class_to_index[label]] = 1.0
    confidence = np.max(proba, axis=1)
    correct = (pred == y_true).astype(float)
    brier = float(np.mean(np.sum((proba - truth) ** 2, axis=1)))
    ece = 0.0
    bins = []
    for lower in np.linspace(0.0, 1.0, 11)[:-1]:
        upper = min(lower + 0.1, 1.0)
        mask = (confidence >= lower) & ((confidence < upper) if upper < 1.0 else (confidence <= upper))
        if not np.any(mask):
            continue
        count = int(mask.sum())
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        ece += (count / len(y_true)) * abs(accuracy - mean_confidence)
        bins.append({
            'range': f'{lower:.1f}-{upper:.1f}',
            'rows': count,
            'accuracy': round(accuracy, 4),
            'confidence': round(mean_confidence, 4),
        })
    return {
        'multiclassBrier': round(brier, 4),
        'expectedCalibrationError': round(float(ece), 4),
        'bins': bins,
    }


def race_bootstrap_accuracy(test_df, pred, n_boot=1000):
    """Race-level bootstrap interval to avoid treating laps as independent."""
    groups = []
    for _, race_df in test_df.groupby(['year', 'round', 'session'], sort=True):
        groups.append(race_df.index.to_numpy())
    if not groups:
        return {'method': 'race bootstrap', 'samples': 0, 'lower95': None, 'upper95': None}
    pred_by_index = pd.Series(pred, index=test_df.index)
    rng = np.random.default_rng(17)
    scores = []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indexes = np.concatenate([groups[i] for i in sampled])
        scores.append(float((pred_by_index.loc[indexes].to_numpy() == test_df.loc[indexes, 'label'].to_numpy()).mean()))
    low, high = np.percentile(scores, [2.5, 97.5])
    return {
        'method': 'race bootstrap percentile interval',
        'samples': n_boot,
        'races': len(groups),
        'lower95': round(float(low), 4),
        'upper95': round(float(high), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-from', type=int, default=2025,
                        help='first year held out as test (strict temporal split)')
    parser.add_argument('--max-train-year', type=int, default=None)
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='RandomForest workers; 1 is portable on restricted Windows hosts')
    args = parser.parse_args()

    df = to_frame(load_rows())
    if len(df) < 50:
        raise SystemExit(json.dumps({'error': 'not enough decision points yet', 'rows': len(df)}))

    train_df = df[df['year'] < args.test_from]
    test_df = df[df['year'] >= args.test_from]
    if args.max_train_year is not None:
        train_df = train_df[train_df['year'] <= args.max_train_year]
    if train_df.empty or test_df.empty:
        raise SystemExit(json.dumps({
            'error': 'temporal split produced an empty side',
            'trainRows': len(train_df), 'testRows': len(test_df),
            'years': sorted(df['year'].unique().tolist()),
        }))

    X_train, y_train = train_df[FEATURES].to_numpy(), train_df['label'].to_numpy()
    X_test, y_test = test_df[FEATURES].to_numpy(), test_df['label'].to_numpy()

    model = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=20,
        class_weight='balanced', random_state=7, n_jobs=args.n_jobs)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    attack_idx = list(model.classes_).index('ATTACK') if 'ATTACK' in model.classes_ else None
    always_save = np.full(len(y_test), 'SAVE')
    gap_only = gap_only_actions(test_df)
    test_by_race = []
    for (year, round_number, session), race_df in test_df.groupby(
            ['year', 'round', 'session'], sort=True):
        race_pred = model.predict(race_df[FEATURES].to_numpy())
        race_truth = race_df['label'].to_numpy()
        race_baseline = np.full(len(race_truth), 'SAVE')
        race_gap_only = gap_only_actions(race_df)
        event_names = race_df['_eventName'].dropna().astype(str).unique().tolist()
        test_by_race.append({
            'year': int(year),
            'round': int(round_number),
            'session': str(session),
            'eventName': event_names[0] if event_names else f'Round {int(round_number)}',
            'rows': int(len(race_df)),
            'accuracy': round(float(accuracy_score(race_truth, race_pred)), 4),
            'macroF1': round(float(f1_score(race_truth, race_pred, labels=LABELS,
                                             average='macro', zero_division=0)), 4),
            'alwaysSaveAccuracy': round(float(accuracy_score(race_truth, race_baseline)), 4),
            'gapOnlyAccuracy': round(float(accuracy_score(race_truth, race_gap_only)), 4),
            'gapOnlyMacroF1': round(float(f1_score(race_truth, race_gap_only, labels=LABELS,
                                                   average='macro', zero_division=0)), 4),
            'classCounts': race_df['label'].value_counts().to_dict(),
        })

    report = {
        'modelType': 'RandomForestClassifier',
        'temporalSplit': {'trainYears': sorted(train_df['year'].unique().tolist()),
                          'testYears': sorted(test_df['year'].unique().tolist()),
                          'strategy': 'STRICT_TEMPORAL_BY_SEASON',
                          'groupBoundary': 'RACE'},
        'rows': {'train': len(train_df), 'test': len(test_df)},
        'classCounts': {'train': train_df['label'].value_counts().to_dict(),
                        'test': test_df['label'].value_counts().to_dict()},
        'testAccuracy': round(float(accuracy_score(y_test, pred)), 4),
        'testMacroF1': round(float(f1_score(y_test, pred, average='macro', zero_division=0)), 4),
        'testUncertainty': race_bootstrap_accuracy(test_df, pred),
        'testCalibration': calibration_metrics(y_test, pred, proba, model.classes_),
        'testRaceCount': len(test_by_race),
        'testByRace': test_by_race,
        'baselines': {
            'alwaysSaveAccuracy': round(float(accuracy_score(y_test, always_save)), 4),
            'alwaysSaveMacroF1': round(float(f1_score(y_test, always_save, average='macro', zero_division=0)), 4),
            'gapOnlyAccuracy': round(float(accuracy_score(y_test, gap_only)), 4),
            'gapOnlyMacroF1': round(float(f1_score(y_test, gap_only, labels=LABELS,
                                                   average='macro', zero_division=0)), 4),
            'gapOnlyRule': f'ATTACK if gap <= {GAP_ATTACK_THRESHOLD_S:.2f}s; DELAY if gap <= {GAP_DELAY_THRESHOLD_S:.2f}s; otherwise SAVE',
        },
        'testReport': classification_report(y_test, pred, zero_division=0, output_dict=True),
        'featureImportances': {name: round(float(v), 4) for name, v in
                               zip(FEATURES, model.feature_importances_)},
        'features': FEATURES,
        'labels': LABELS,
        'featurePolicy': {
            'decisionSchemaVersion': DECISION_SCHEMA_VERSION,
            'cutoff': 'LAP_START',
            'futureOutcomeFieldsAreTargetsOnly': True,
            'raceMeanSpeedMissingValue': 0.0,
            'energyProvenance': 'MODELLED_SURROGATE',
        },
        'note': 'Outcome labels: ATTACK = pass made and held 6 laps; DELAY = durable pass within 6 laps; SAVE = none. SoC features are lap-time battle surrogates, modelled rather than measured.',
    }
    report['modelVsAlwaysSave'] = {
        'accuracyDelta': round(report['testAccuracy'] - report['baselines']['alwaysSaveAccuracy'], 4),
        'beatsBaseline': report['testAccuracy'] > report['baselines']['alwaysSaveAccuracy'],
    }

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'features': FEATURES, 'classes': list(model.classes_)},
                MODELS / 'overtake_rf.joblib')
    (MODELS / 'overtake_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('rows', 'temporalSplit', 'testAccuracy', 'featureImportances')}, indent=2))


if __name__ == '__main__':
    main()
