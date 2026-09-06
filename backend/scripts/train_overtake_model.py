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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# These are real decision-time values and remain available for controlled
# experiments. They are not in the production feature set unless they improve
# the frozen holdout; the first ablation did not.
WEATHER_FEATURES = [
    'airTempC', 'trackTempC', 'humidityPct', 'windSpeedMps', 'rainfall',
    'weatherMissing',
]
LABELS = ['SAVE', 'DELAY', 'ATTACK']
PERSISTENCE_HORIZONS = (1, 2, 3, 5, 6)
DECISION_SCHEMA_VERSION = 'decision-point.v5'
GAP_ATTACK_THRESHOLD_S = 0.70
GAP_DELAY_THRESHOLD_S = 1.20

COMPOUND_ORDINAL = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}


def weather_value(value, field, default=0.0):
    """Extract a causal decision-time weather value from a cache row."""
    if not isinstance(value, dict):
        return default
    try:
        number = float(value.get(field))
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


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
    weather = df['weather'] if 'weather' in df else pd.Series([None] * len(df), index=df.index)
    df['weatherMissing'] = weather.map(lambda value: 0.0 if isinstance(value, dict) else 1.0)
    for feature, source in (
        ('airTempC', 'airTempC'), ('trackTempC', 'trackTempC'),
        ('humidityPct', 'humidityPct'), ('windSpeedMps', 'windSpeedMps'),
        ('rainfall', 'rainfall'),
    ):
        df[feature] = weather.map(lambda value, key=source: weather_value(value, key))
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


def candidate_metrics(name, estimator, X_train, y_train, X_test, y_test, test_df, fit_kwargs=None):
    """Fit one comparable candidate using the frozen split and report it."""
    estimator.fit(X_train, y_train, **(fit_kwargs or {}))
    pred = estimator.predict(X_test)
    proba = estimator.predict_proba(X_test)
    classes = estimator.classes_ if hasattr(estimator, 'classes_') else estimator.named_steps['classifier'].classes_
    return {
        'modelType': name,
        'accuracy': round(float(accuracy_score(y_test, pred)), 4),
        'macroF1': round(float(f1_score(y_test, pred, labels=LABELS, average='macro', zero_division=0)), 4),
        'uncertainty': race_bootstrap_accuracy(test_df, pred),
        'calibration': calibration_metrics(y_test, pred, proba, classes),
    }, estimator


def train_pass_durability_component(train_df, test_df, features, n_jobs):
    """Benchmark P(hold >= H | an immediate observed on-track pass).

    This deliberately does not enter the tactical tree yet. It is a separate,
    strictly held-out component evaluation: training only sees historical pass
    states and the test season is never used to fit its parameters. The target
    is conditional on a real immediate pass, so it cannot by itself support a
    counterfactual claim for ATTACK/SAVE/DELAY.
    """
    train_passes = train_df[train_df['passedNow'].fillna(False)].copy()
    test_passes = test_df[test_df['passedNow'].fillna(False)].copy()
    summaries = []
    artifacts = {}
    for horizon in PERSISTENCE_HORIZONS:
        y_train = (train_passes['observedLeadLaps'].fillna(0).astype(float) >= horizon).astype(int)
        y_test = (test_passes['observedLeadLaps'].fillna(0).astype(float) >= horizon).astype(int)
        summary = {
            'horizonLaps': horizon,
            'trainRows': int(len(train_passes)),
            'testRows': int(len(test_passes)),
            'trainObservedHoldRate': round(float(y_train.mean()), 4) if len(y_train) else None,
            'testObservedHoldRate': round(float(y_test.mean()), 4) if len(y_test) else None,
            'target': f'P(hold position for at least {horizon} lap(s) | immediate observed on-track pass)',
        }
        if len(train_passes) < 30 or len(test_passes) == 0 or y_train.nunique() < 2:
            summary['status'] = 'INSUFFICIENT_CLASS_VARIATION'
            summaries.append(summary)
            continue
        model = RandomForestClassifier(
            n_estimators=300, max_depth=7, min_samples_leaf=12,
            class_weight='balanced', random_state=100 + horizon, n_jobs=n_jobs)
        model.fit(train_passes[features].to_numpy(), y_train.to_numpy())
        positive_index = list(model.classes_).index(1)
        probability = model.predict_proba(test_passes[features].to_numpy())[:, positive_index]
        predicted = (probability >= 0.5).astype(int)
        summary.update({
            'status': 'BENCHMARK_ONLY',
            'accuracy': round(float(accuracy_score(y_test, predicted)), 4),
            'f1': round(float(f1_score(y_test, predicted, zero_division=0)), 4),
            'brier': round(float(brier_score_loss(y_test, probability)), 4),
            'rocAuc': round(float(roc_auc_score(y_test, probability)), 4) if y_test.nunique() > 1 else None,
            'constantRateBrier': round(float(brier_score_loss(
                y_test, np.full(len(y_test), float(y_train.mean())))), 4),
        })
        artifacts[str(horizon)] = {'model': model, 'classes': list(model.classes_)}
        summaries.append(summary)
    return {
        'schemaVersion': 'pass-durability.v1',
        'status': 'BENCHMARK_NOT_REPLAY_INPUT',
        'definition': 'Conditional durability model evaluated only at real immediate on-track pass states.',
        'features': features,
        'trainPassRows': int(len(train_passes)),
        'testPassRows': int(len(test_passes)),
        'horizons': summaries,
        'note': 'This component does not infer private battery state and is not yet used to score counterfactual tree branches.',
    }, {'schemaVersion': 'pass-durability.v1', 'features': features, 'models': artifacts}


def development_selection(df, features, development_from, development_to, n_jobs):
    """Compare candidates on an earlier development window, never 2026.

    The final holdout remains a report-only check. This avoids silently using
    a 2026 score to choose features or a model family.
    """
    fit_df = df[df['year'] < development_from]
    development_df = df[(df['year'] >= development_from) &
                        (df['year'] <= development_to)]
    if fit_df.empty or development_df.empty:
        return {
            'status': 'UNAVAILABLE',
            'reason': 'The requested fit/development temporal split is empty.',
        }
    X_fit, y_fit = fit_df[features].to_numpy(), fit_df['label'].to_numpy()
    X_dev, y_dev = development_df[features].to_numpy(), development_df['label'].to_numpy()
    random_forest = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=20,
        class_weight='balanced', random_state=7, n_jobs=n_jobs)
    rf_summary, _ = candidate_metrics(
        'RandomForestClassifier', random_forest, X_fit, y_fit, X_dev, y_dev, development_df)
    logistic = Pipeline([
        ('scale', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=7)),
    ])
    logistic_summary, _ = candidate_metrics(
        'LogisticRegression', logistic, X_fit, y_fit, X_dev, y_dev, development_df)
    class_counts = pd.Series(y_fit).value_counts()
    class_weights = np.asarray([
        len(y_fit) / (len(LABELS) * class_counts[label]) for label in y_fit
    ])
    boosted = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.06, max_depth=3,
        min_samples_leaf=20, random_state=7)
    boosted_summary, _ = candidate_metrics(
        'GradientBoostingClassifier', boosted, X_fit, y_fit, X_dev, y_dev,
        development_df, {'sample_weight': class_weights})
    always_save = np.full(len(y_dev), 'SAVE')
    gap_only = gap_only_actions(development_df)
    return {
        'status': 'SELECTION_WINDOW_ONLY',
        'temporalSplit': {
            'fitYears': sorted(fit_df['year'].unique().tolist()),
            'developmentYears': sorted(development_df['year'].unique().tolist()),
            'finalHoldoutExcluded': True,
            'strategy': 'STRICT_TEMPORAL_DEVELOPMENT',
        },
        'rows': {'fit': int(len(fit_df)), 'development': int(len(development_df))},
        'models': {
            'RandomForestClassifier': rf_summary,
            'LogisticRegression': logistic_summary,
            'GradientBoostingClassifier': boosted_summary,
        },
        'baselines': {
            'alwaysSaveAccuracy': round(float(accuracy_score(y_dev, always_save)), 4),
            'alwaysSaveMacroF1': round(float(f1_score(
                y_dev, always_save, labels=LABELS, average='macro', zero_division=0)), 4),
            'gapOnlyAccuracy': round(float(accuracy_score(y_dev, gap_only)), 4),
            'gapOnlyMacroF1': round(float(f1_score(
                y_dev, gap_only, labels=LABELS, average='macro', zero_division=0)), 4),
        },
        'note': 'Use this window for candidate/model decisions. Do not select a model based on the final 2026 report.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-from', type=int, default=2025,
                        help='first year held out as test (strict temporal split)')
    parser.add_argument('--max-train-year', type=int, default=None)
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='RandomForest workers; 1 is portable on restricted Windows hosts')
    parser.add_argument('--include-weather', action='store_true',
                        help='evaluate decision-time weather as an experimental feature group')
    parser.add_argument('--development-from', type=int, default=2024,
                        help='first year of the pre-holdout development window')
    parser.add_argument('--development-to', type=int, default=2025,
                        help='last year of the pre-holdout development window')
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

    active_features = FEATURES + WEATHER_FEATURES if args.include_weather else FEATURES
    if args.development_from >= args.test_from or args.development_to >= args.test_from:
        raise SystemExit('development window must end before the final holdout')
    if args.development_from > args.development_to:
        raise SystemExit('development-from must not be after development-to')
    X_train, y_train = train_df[active_features].to_numpy(), train_df['label'].to_numpy()
    X_test, y_test = test_df[active_features].to_numpy(), test_df['label'].to_numpy()

    model = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=20,
        class_weight='balanced', random_state=7, n_jobs=args.n_jobs)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    logistic = Pipeline([
        ('scale', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=7)),
    ])
    logistic_summary, logistic = candidate_metrics(
        'LogisticRegression', logistic, X_train, y_train, X_test, y_test, test_df)
    class_counts = pd.Series(y_train).value_counts()
    class_weights = np.asarray([
        len(y_train) / (len(LABELS) * class_counts[label]) for label in y_train
    ])
    boosted = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.06, max_depth=3,
        min_samples_leaf=20, random_state=7)
    boosted_summary, boosted = candidate_metrics(
        'GradientBoostingClassifier', boosted, X_train, y_train,
        X_test, y_test, test_df, {'sample_weight': class_weights})
    # Calibration must never learn from the held-out season. Fit a separate
    # Random Forest through the season before the final training year, then
    # learn its sigmoid mapping from that final historical year only.
    calibration_year = int(train_df['year'].max())
    calibration_df = train_df[train_df['year'] == calibration_year]
    calibration_fit_df = train_df[train_df['year'] < calibration_year]
    calibrated_summary = None
    calibrated_rf = None
    if not calibration_df.empty and not calibration_fit_df.empty:
        calibration_base = RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=20,
            class_weight='balanced', random_state=7, n_jobs=args.n_jobs)
        calibration_base.fit(calibration_fit_df[active_features].to_numpy(),
                             calibration_fit_df['label'].to_numpy())
        calibrated_rf = CalibratedClassifierCV(
            FrozenEstimator(calibration_base), method='sigmoid')
        calibrated_rf.fit(calibration_df[active_features].to_numpy(),
                          calibration_df['label'].to_numpy())
        calibrated_pred = calibrated_rf.predict(X_test)
        calibrated_proba = calibrated_rf.predict_proba(X_test)
        calibrated_summary = {
            'modelType': 'RandomForestSigmoidCalibrated',
            'accuracy': round(float(accuracy_score(y_test, calibrated_pred)), 4),
            'macroF1': round(float(f1_score(y_test, calibrated_pred, labels=LABELS,
                                             average='macro', zero_division=0)), 4),
            'uncertainty': race_bootstrap_accuracy(test_df, calibrated_pred),
            'calibration': calibration_metrics(y_test, calibrated_pred,
                                               calibrated_proba, calibrated_rf.classes_),
            'calibrationSplit': {
                'fitYears': sorted(calibration_fit_df['year'].unique().tolist()),
                'calibrationYear': calibration_year,
                'heldOutYears': sorted(test_df['year'].unique().tolist()),
                'method': 'sigmoid',
            },
            'production': False,
        }
    # This is deliberately evaluated independently of the three-action model:
    # its target exists only after an observed immediate pass and cannot be
    # used as a causal action label without a further counterfactual design.
    durability_component, durability_artifact = train_pass_durability_component(
        train_df, test_df, active_features, args.n_jobs)
    development_report = development_selection(
        df, active_features, args.development_from, args.development_to, args.n_jobs)
    attack_idx = list(model.classes_).index('ATTACK') if 'ATTACK' in model.classes_ else None
    always_save = np.full(len(y_test), 'SAVE')
    gap_only = gap_only_actions(test_df)
    test_by_race = []
    for (year, round_number, session), race_df in test_df.groupby(
            ['year', 'round', 'session'], sort=True):
        race_pred = model.predict(race_df[active_features].to_numpy())
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
        'modelComparison': {
            'RandomForestClassifier': {
                'modelType': 'RandomForestClassifier',
                'accuracy': round(float(accuracy_score(y_test, pred)), 4),
                'macroF1': round(float(f1_score(y_test, pred, average='macro', zero_division=0)), 4),
                'uncertainty': race_bootstrap_accuracy(test_df, pred),
                'calibration': calibration_metrics(y_test, pred, proba, model.classes_),
                'production': True,
            },
            'LogisticRegression': logistic_summary,
            'GradientBoostingClassifier': boosted_summary,
            **({'RandomForestSigmoidCalibrated': calibrated_summary}
               if calibrated_summary is not None else {}),
        },
        'passDurabilityComponent': durability_component,
        'developmentSelection': development_report,
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
                               zip(active_features, model.feature_importances_)},
        'features': active_features,
        'labels': LABELS,
        'featurePolicy': {
            'decisionSchemaVersion': DECISION_SCHEMA_VERSION,
            'cutoff': 'LAP_START',
            'futureOutcomeFieldsAreTargetsOnly': True,
            'raceMeanSpeedMissingValue': 0.0,
            'energyProvenance': 'MODELLED_SURROGATE',
            'weatherPolicy': 'Decision-time FastF1 weather snapshot; missing values use zero plus weatherMissing=1.',
            'weatherFeaturesIncluded': args.include_weather,
            'raceControlPolicy': 'Non-green and pit-distorted rows are excluded before fitting; race control remains a replay gate, not a learned action feature.',
        },
        'note': 'Outcome labels: ATTACK = pass made and held 6 laps; DELAY = durable pass within 6 laps; SAVE = none. SoC features are lap-time battle surrogates, modelled rather than measured.',
    }
    report['modelVsAlwaysSave'] = {
        'accuracyDelta': round(report['testAccuracy'] - report['baselines']['alwaysSaveAccuracy'], 4),
        'beatsBaseline': report['testAccuracy'] > report['baselines']['alwaysSaveAccuracy'],
    }

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'features': active_features, 'classes': list(model.classes_)},
                MODELS / 'overtake_rf.joblib')
    joblib.dump({'model': logistic, 'features': active_features,
                 'classes': list(logistic.named_steps['classifier'].classes_)},
                MODELS / 'overtake_logistic.joblib')
    joblib.dump({'model': boosted, 'features': active_features, 'classes': list(boosted.classes_)},
                MODELS / 'overtake_gradientboost.joblib')
    if calibrated_rf is not None:
        joblib.dump({'model': calibrated_rf, 'features': active_features,
                     'classes': list(calibrated_rf.classes_)},
                    MODELS / 'overtake_rf_sigmoid.joblib')
    joblib.dump(durability_artifact, MODELS / 'pass_durability_rf.joblib')
    (MODELS / 'overtake_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('rows', 'temporalSplit', 'testAccuracy', 'featureImportances')}, indent=2))


if __name__ == '__main__':
    main()
