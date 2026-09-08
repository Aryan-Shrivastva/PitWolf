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
import itertools
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
from overtake_feature_schema import (COMPOUND_ORDINAL, FEATURES,
                                     FEATURE_SCHEMA_VERSION, WEATHER_FEATURES)

ROOT = pathlib.Path(CACHE_DIR).parent
MODELS = ROOT / 'models'

LABELS = ['SAVE', 'DELAY', 'ATTACK']
PERSISTENCE_HORIZONS = (1, 2, 3, 5, 6)
DECISION_SCHEMA_VERSION = 'decision-point.v6'
GAP_ATTACK_THRESHOLD_S = 0.70
GAP_DELAY_THRESHOLD_S = 1.20

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
    for path in sorted(ROOT.glob('decision-points/*/*_r.json')):
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


def decision_cache_schema_status():
    """Return the local cache files that cannot safely join a retraining run."""
    stale = []
    invalid = []
    for path in sorted(ROOT.glob('decision-points/*/*_r.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            invalid.append(path)
            continue
        if (payload.get('schemaVersion') != DECISION_SCHEMA_VERSION
                or payload.get('featureSchemaVersion') != FEATURE_SCHEMA_VERSION):
            stale.append(path)
    return stale, invalid


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


def apply_action_policy(probabilities, classes, action_weights=None):
    """Choose an action from class probabilities using fixed policy weights.

    The weights are selected only from historical out-of-fold predictions. They
    are decision thresholds in multiplicative form: values below one require
    stronger evidence before choosing an action, while SAVE remains the neutral
    reference. They never alter the fitted probability model itself.
    """
    labels = np.asarray(classes)
    weights = action_weights or {}
    factors = np.asarray([float(weights.get(label, 1.0)) for label in labels])
    return labels[np.argmax(probabilities * factors, axis=1)]


def select_action_policy(out_of_fold_probabilities, out_of_fold_truth, classes):
    """Select RF action weights using historical expanding-fold predictions only."""
    if not out_of_fold_probabilities:
        return None
    probabilities = np.vstack(out_of_fold_probabilities)
    truth = np.concatenate(out_of_fold_truth)
    grid = (0.50, 0.65, 0.80, 1.00, 1.20, 1.40, 1.60)
    candidates = []
    for delay_weight, attack_weight in itertools.product(grid, repeat=2):
        weights = {'SAVE': 1.0, 'DELAY': delay_weight, 'ATTACK': attack_weight}
        predicted = apply_action_policy(probabilities, classes, weights)
        candidates.append({
            'weights': weights,
            'macroF1': float(f1_score(truth, predicted, labels=LABELS,
                                       average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(truth, predicted)),
            'distanceFromArgmax': abs(delay_weight - 1.0) + abs(attack_weight - 1.0),
        })
    best = max(candidates, key=lambda item: (
        item['macroF1'], item['accuracy'], -item['distanceFromArgmax']))
    raw_predicted = apply_action_policy(probabilities, classes)
    return {
        'status': 'HISTORICAL_OUT_OF_FOLD_SELECTION',
        'selectionRows': int(len(truth)),
        'selectionMetric': 'macro F1, then accuracy, then closest to raw argmax',
        'weights': {key: round(value, 2) for key, value in best['weights'].items()},
        'weightedMacroF1': round(best['macroF1'], 4),
        'weightedAccuracy': round(best['accuracy'], 4),
        'rawArgmaxMacroF1': round(float(f1_score(
            truth, raw_predicted, labels=LABELS, average='macro', zero_division=0)), 4),
        'rawArgmaxAccuracy': round(float(accuracy_score(truth, raw_predicted)), 4),
        'note': 'Weights are selected without any 2026 labels and apply only to the final action choice, not probability calibration.',
    }


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


def grouped_holdout_metrics(frame, truth, prediction, group_column, result_key):
    """Report held-out quality by an observed race dimension only.

    These diagnostic slices never feed back into threshold, feature, or model
    selection. They exist to show whether one global score hides a driver- or
    circuit-specific weakness.
    """
    evaluation = frame.copy()
    evaluation['_truth'] = np.asarray(truth)
    evaluation['_prediction'] = np.asarray(prediction)
    result = []
    for group_value, group in evaluation.groupby(group_column, dropna=True, sort=True):
        labels = group['_truth'].to_numpy()
        predicted = group['_prediction'].to_numpy()
        always_save = np.full(len(group), 'SAVE')
        gap_only = gap_only_actions(group)
        result.append({
            result_key: str(group_value),
            'rows': int(len(group)),
            'accuracy': round(float(accuracy_score(labels, predicted)), 4),
            'macroF1': round(float(f1_score(labels, predicted, labels=LABELS,
                                              average='macro', zero_division=0)), 4),
            'alwaysSaveAccuracy': round(float(accuracy_score(labels, always_save)), 4),
            'gapOnlyAccuracy': round(float(accuracy_score(labels, gap_only)), 4),
            'classCounts': group['label'].value_counts().to_dict(),
        })
    return result


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


def train_immediate_pass_component(train_df, test_df, features, n_jobs):
    """Benchmark P(observed immediate pass | causal battle-state features).

    This is intentionally descriptive rather than an ATTACK-policy effect.
    Race history does not tell us what would have happened had the driver
    chosen a different energy deployment at the same state.
    """
    y_train = train_df['passedNow'].fillna(False).astype(int)
    y_test = test_df['passedNow'].fillna(False).astype(int)
    summary = {
        'schemaVersion': 'immediate-pass.v1',
        'status': 'BENCHMARK_NOT_REPLAY_INPUT',
        'definition': 'Probability of an observed immediate on-track pass from the causal battle state.',
        'features': features,
        'trainRows': int(len(train_df)),
        'testRows': int(len(test_df)),
        'trainObservedPassRate': round(float(y_train.mean()), 4) if len(y_train) else None,
        'testObservedPassRate': round(float(y_test.mean()), 4) if len(y_test) else None,
        'note': 'Observed passes reflect the real driver action and environment; this is not a causal ATTACK-versus-SAVE effect model.',
    }
    if len(train_df) < 30 or len(test_df) == 0 or y_train.nunique() < 2:
        summary['status'] = 'INSUFFICIENT_CLASS_VARIATION'
        return summary, {'schemaVersion': 'immediate-pass.v1', 'features': features, 'model': None}
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight='balanced', random_state=71, n_jobs=n_jobs)
    model.fit(train_df[features].to_numpy(), y_train.to_numpy())
    positive_index = list(model.classes_).index(1)
    probability = model.predict_proba(test_df[features].to_numpy())[:, positive_index]
    predicted = (probability >= 0.5).astype(int)
    summary.update({
        'accuracy': round(float(accuracy_score(y_test, predicted)), 4),
        'f1': round(float(f1_score(y_test, predicted, zero_division=0)), 4),
        'brier': round(float(brier_score_loss(y_test, probability)), 4),
        'rocAuc': round(float(roc_auc_score(y_test, probability)), 4) if y_test.nunique() > 1 else None,
        'constantRateBrier': round(float(brier_score_loss(
                y_test, np.full(len(y_test), float(y_train.mean())))), 4),
    })
    summary['calibrationBenchmark'] = calibrated_binary_benchmark(
        train_df, test_df, features, 'passedNow', n_jobs)
    return summary, {
        'schemaVersion': 'immediate-pass.v1',
        'features': features,
        'model': model,
        'classes': list(model.classes_),
    }


def calibrated_binary_benchmark(train_df, test_df, features, target_column, n_jobs):
    """Evaluate sigmoid calibration without fitting any map on the 2026 holdout.

    This improves the *observational* pass/durability probability benchmark;
    it does not turn the target into a causal ATTACK-versus-SAVE estimate.
    """
    calibration_year = int(train_df['year'].max())
    fit_df = train_df[train_df['year'] < calibration_year]
    calibration_df = train_df[train_df['year'] == calibration_year]
    target = lambda frame: frame[target_column].fillna(False).astype(int)
    y_fit, y_calibration, y_test = target(fit_df), target(calibration_df), target(test_df)
    if (len(fit_df) < 30 or len(calibration_df) < 20 or y_fit.nunique() < 2
            or y_calibration.nunique() < 2 or y_test.nunique() < 2):
        return {'status': 'INSUFFICIENT_CLASS_VARIATION'}
    base = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight='balanced', random_state=173, n_jobs=n_jobs)
    base.fit(fit_df[features].to_numpy(), y_fit.to_numpy())
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid')
    calibrated.fit(calibration_df[features].to_numpy(), y_calibration.to_numpy())
    positive_index = list(calibrated.classes_).index(1)
    probability = calibrated.predict_proba(test_df[features].to_numpy())[:, positive_index]
    return {
        'status': 'BENCHMARK_ONLY',
        'method': 'sigmoid',
        'fitYears': sorted(fit_df['year'].unique().tolist()),
        'calibrationYear': calibration_year,
        'heldOutYears': sorted(test_df['year'].unique().tolist()),
        'brier': round(float(brier_score_loss(y_test, probability)), 4),
        'rocAuc': round(float(roc_auc_score(y_test, probability)), 4),
        'note': 'Calibration is selected and fitted before the final holdout; it remains an observational benchmark.',
    }


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
        calibration_frame = train_passes.copy()
        test_calibration_frame = test_passes.copy()
        calibration_frame['_holdTarget'] = y_train
        test_calibration_frame['_holdTarget'] = y_test
        summary['calibrationBenchmark'] = calibrated_binary_benchmark(
            calibration_frame, test_calibration_frame, features, '_holdTarget', n_jobs)
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


def selection_metrics(estimator, X_fit, y_fit, X_validation, y_validation, fit_kwargs=None):
    """Compact candidate score for internal historical model selection."""
    estimator.fit(X_fit, y_fit, **(fit_kwargs or {}))
    prediction = estimator.predict(X_validation)
    return {
        'accuracy': round(float(accuracy_score(y_validation, prediction)), 4),
        'macroF1': round(float(f1_score(
            y_validation, prediction, labels=LABELS, average='macro', zero_division=0)), 4),
    }


def rolling_temporal_selection(df, features, final_holdout_from, first_validation_year, n_jobs):
    """Choose model families on expanding historical windows, never 2026.

    Every fold fits only seasons before its validation year. The final 2026
    holdout is excluded from all folds, so it stays a one-time final check.
    """
    historical = df[df['year'] < final_holdout_from]
    candidate_names = (
        'RandomForestClassifier', 'LogisticRegression', 'GradientBoostingClassifier')
    folds = []
    collected = {name: [] for name in candidate_names}
    rf_oof_probabilities = []
    rf_oof_truth = []
    rf_classes = None
    for validation_year in range(first_validation_year, final_holdout_from):
        fit_df = historical[historical['year'] < validation_year]
        validation_df = historical[historical['year'] == validation_year]
        if len(fit_df) < 50 or validation_df.empty or fit_df['label'].nunique() < 2:
            continue
        X_fit, y_fit = fit_df[features].to_numpy(), fit_df['label'].to_numpy()
        X_validation = validation_df[features].to_numpy()
        y_validation = validation_df['label'].to_numpy()
        random_forest = RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=20,
            class_weight='balanced', random_state=7, n_jobs=n_jobs)
        logistic = Pipeline([
            ('scale', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=2000, class_weight='balanced', random_state=7)),
        ])
        class_counts = pd.Series(y_fit).value_counts()
        class_weights = np.asarray([
            len(y_fit) / (len(LABELS) * class_counts[label]) for label in y_fit
        ])
        boosted = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.06, max_depth=3,
            min_samples_leaf=20, random_state=7)
        model_metrics = {
            'RandomForestClassifier': selection_metrics(
                random_forest, X_fit, y_fit, X_validation, y_validation),
            'LogisticRegression': selection_metrics(
                logistic, X_fit, y_fit, X_validation, y_validation),
            'GradientBoostingClassifier': selection_metrics(
                boosted, X_fit, y_fit, X_validation, y_validation,
                {'sample_weight': class_weights}),
        }
        # Reuse the already fitted RF from selection_metrics to derive a
        # separate action policy only from historical validation folds. This
        # does not use the final holdout labels and does not change the model's
        # probability estimates.
        rf_probability = random_forest.predict_proba(X_validation)
        rf_oof_probabilities.append(rf_probability)
        rf_oof_truth.append(y_validation)
        rf_classes = random_forest.classes_
        for name, metrics in model_metrics.items():
            collected[name].append((len(validation_df), metrics))
        always_save = np.full(len(y_validation), 'SAVE')
        folds.append({
            'fitYears': sorted(fit_df['year'].unique().tolist()),
            'validationYear': validation_year,
            'rows': int(len(validation_df)),
            'models': model_metrics,
            'baselines': {
                'alwaysSaveAccuracy': round(float(accuracy_score(y_validation, always_save)), 4),
                'alwaysSaveMacroF1': round(float(f1_score(
                    y_validation, always_save, labels=LABELS, average='macro', zero_division=0)), 4),
            },
        })
    aggregates = {}
    for name, values in collected.items():
        total_rows = sum(rows for rows, _ in values)
        aggregates[name] = {
            'weightedAccuracy': round(sum(rows * metric['accuracy'] for rows, metric in values) / total_rows, 4)
            if total_rows else None,
            'weightedMacroF1': round(sum(rows * metric['macroF1'] for rows, metric in values) / total_rows, 4)
            if total_rows else None,
            'folds': len(values),
            'macroF1Wins': sum(
                1 for fold in folds
                if fold['models'][name]['macroF1'] == max(
                    item['macroF1'] for item in fold['models'].values())),
        }
    selected_candidate = max(
        aggregates,
        key=lambda name: (aggregates[name]['weightedMacroF1'] or -1,
                          aggregates[name]['weightedAccuracy'] or -1),
        default=None,
    )
    random_forest_action_policy = select_action_policy(
        rf_oof_probabilities, rf_oof_truth, rf_classes) if rf_classes is not None else None
    return {
        'status': 'ROLLING_SELECTION_ONLY' if folds else 'UNAVAILABLE',
        'strategy': 'EXPANDING_WINDOW_TEMPORAL_VALIDATION',
        'finalHoldoutExcluded': True,
        'folds': folds,
        'aggregate': aggregates,
        'selectedCandidate': selected_candidate,
        'randomForestActionPolicy': random_forest_action_policy,
        'selectionCriterion': 'Highest weighted macro F1 across expanding temporal folds; raw accuracy is secondary because labels are imbalanced.',
        'note': 'Use these historical folds for candidate selection; train the selected approach on all 2018–2025 before the one-time 2026 final evaluation.',
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
    parser.add_argument('--rolling-first-validation-year', type=int, default=2022,
                        help='first expanding-window historical validation year')
    args = parser.parse_args()

    stale, invalid = decision_cache_schema_status()
    if stale or invalid:
        raise SystemExit(json.dumps({
            'error': 'decision-point cache audit is not training-ready',
            'expectedSchemaVersion': DECISION_SCHEMA_VERSION,
            'staleSchemaFiles': len(stale),
            'invalidFiles': len(invalid),
            'nextStep': (
                'Rebuild every cached race with batch_extract_decision_points.py '
                'and confirm audit_decision_labels.py reports readyForTraining=true.'
            ),
        }))

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
    if args.rolling_first_validation_year >= args.test_from:
        raise SystemExit('first rolling validation year must precede the final holdout')
    X_train, y_train = train_df[active_features].to_numpy(), train_df['label'].to_numpy()
    X_test, y_test = test_df[active_features].to_numpy(), test_df['label'].to_numpy()

    model = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=20,
        class_weight='balanced', random_state=7, n_jobs=args.n_jobs)
    model.fit(X_train, y_train)

    raw_argmax_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    rolling_selection = rolling_temporal_selection(
        df, active_features, args.test_from, args.rolling_first_validation_year, args.n_jobs)
    action_policy = rolling_selection.get('randomForestActionPolicy')
    action_weights = action_policy.get('weights') if action_policy else None
    pred = apply_action_policy(proba, model.classes_, action_weights)
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
    immediate_pass_component, immediate_pass_artifact = train_immediate_pass_component(
        train_df, test_df, active_features, args.n_jobs)
    always_save = np.full(len(y_test), 'SAVE')
    gap_only = gap_only_actions(test_df)
    test_by_race = []
    for (year, round_number, session), race_df in test_df.groupby(
            ['year', 'round', 'session'], sort=True):
        race_proba = model.predict_proba(race_df[active_features].to_numpy())
        race_pred = apply_action_policy(race_proba, model.classes_, action_weights)
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
        'decisionPointSchemaVersion': DECISION_SCHEMA_VERSION,
        'featureSchemaVersion': FEATURE_SCHEMA_VERSION,
        'temporalSplit': {'trainYears': sorted(train_df['year'].unique().tolist()),
                          'testYears': sorted(test_df['year'].unique().tolist()),
                          'strategy': 'STRICT_TEMPORAL_BY_SEASON',
                          'groupBoundary': 'RACE'},
        'rows': {'train': len(train_df), 'test': len(test_df)},
        'classCounts': {'train': train_df['label'].value_counts().to_dict(),
                        'test': test_df['label'].value_counts().to_dict()},
        'testAccuracy': round(float(accuracy_score(y_test, pred)), 4),
        'testMacroF1': round(float(f1_score(y_test, pred, average='macro', zero_division=0)), 4),
        'rawArgmaxMetrics': {
            'accuracy': round(float(accuracy_score(y_test, raw_argmax_pred)), 4),
            'macroF1': round(float(f1_score(
                y_test, raw_argmax_pred, labels=LABELS, average='macro', zero_division=0)), 4),
        },
        'actionPolicy': action_policy,
        'testUncertainty': race_bootstrap_accuracy(test_df, pred),
        'testCalibration': calibration_metrics(y_test, pred, proba, model.classes_),
        'testRaceCount': len(test_by_race),
        'testByRace': test_by_race,
        'testByDriver': grouped_holdout_metrics(test_df, y_test, pred, 'driver', 'driver'),
        'testByTrack': grouped_holdout_metrics(test_df, y_test, pred, '_eventName', 'eventName'),
        'modelComparison': {
            'RandomForestClassifier': {
                'modelType': 'RandomForestClassifier',
                'accuracy': round(float(accuracy_score(y_test, pred)), 4),
                'macroF1': round(float(f1_score(y_test, pred, average='macro', zero_division=0)), 4),
                'uncertainty': race_bootstrap_accuracy(test_df, pred),
                'calibration': calibration_metrics(y_test, pred, proba, model.classes_),
                'production': False,
                'status': 'FINAL_HOLDOUT_RESULT',
            },
            'LogisticRegression': logistic_summary,
            'GradientBoostingClassifier': boosted_summary,
            **({'RandomForestSigmoidCalibrated': calibrated_summary}
               if calibrated_summary is not None else {}),
        },
        'passDurabilityComponent': durability_component,
        'immediatePassComponent': immediate_pass_component,
        'rollingSelection': rolling_selection,
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
        'note': 'Outcome labels: ATTACK = pass made and held 6 laps; DELAY = durable pass beginning within 5 laps and held 6 laps; SAVE = none. SoC features are lap-time battle surrogates, modelled rather than measured.',
    }
    report['modelVsAlwaysSave'] = {
        'accuracyDelta': round(report['testAccuracy'] - report['baselines']['alwaysSaveAccuracy'], 4),
        'beatsBaseline': report['testAccuracy'] > report['baselines']['alwaysSaveAccuracy'],
    }

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'features': active_features, 'classes': list(model.classes_),
                 'featureSchemaVersion': FEATURE_SCHEMA_VERSION,
                 'actionPolicy': action_policy},
                MODELS / 'overtake_rf.joblib')
    joblib.dump({'model': logistic, 'features': active_features,
                 'featureSchemaVersion': FEATURE_SCHEMA_VERSION,
                 'classes': list(logistic.named_steps['classifier'].classes_)},
                MODELS / 'overtake_logistic.joblib')
    joblib.dump({'model': boosted, 'features': active_features,
                 'featureSchemaVersion': FEATURE_SCHEMA_VERSION,
                 'classes': list(boosted.classes_)},
                MODELS / 'overtake_gradientboost.joblib')
    if calibrated_rf is not None:
        joblib.dump({'model': calibrated_rf, 'features': active_features,
                     'featureSchemaVersion': FEATURE_SCHEMA_VERSION,
                     'classes': list(calibrated_rf.classes_)},
                    MODELS / 'overtake_rf_sigmoid.joblib')
    joblib.dump(durability_artifact, MODELS / 'pass_durability_rf.joblib')
    joblib.dump(immediate_pass_artifact, MODELS / 'immediate_pass_rf.joblib')
    (MODELS / 'overtake_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('rows', 'temporalSplit', 'testAccuracy', 'featureImportances')}, indent=2))


if __name__ == '__main__':
    main()
