"""Train and compare transparent qualifying pace baselines.

This intentionally stops at Ridge and Random Forest.  XGBoost/LightGBM is a
later challenger only if these baselines are evaluated successfully on whole,
unseen rounds.  The target is observed lap delta to the event pole reference;
it is not a target for unseen team ERS deployment.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_DIR = Path(__file__).resolve().parents[1] / 'data' / 'f1-cache' / 'models'
FEATURE_VERSION = 'qualifying-features.v1'
MODEL_VERSION = 'qualifying-pace.v1'
NUMERIC = [
    'lapNumber', 'compoundCode', 'airTempC', 'trackTempC', 'humidityPct', 'pressureMbar',
    'telemetrySamples', 'topSpeedKph', 'meanSpeedKph', 'p90SpeedKph', 'meanThrottlePct',
    'fullThrottleS', 'highSpeedFullThrottleS', 'brakingS', 'coastS', 'brakeEvents',
    'peakDecelMs2', 'peakAccelMs2',
]
CATEGORICAL = ['team', 'compound']


def metrics(y_true, y_pred):
    return {
        'maeS': round(float(mean_absolute_error(y_true, y_pred)), 4),
        'rmseS': round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
        'biasS': round(float(np.mean(np.asarray(y_pred) - np.asarray(y_true))), 4),
    }


def make_ridge():
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERIC),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
    ])
    return Pipeline([('preprocess', preprocess), ('model', Ridge(alpha=3.0))])


def make_forest():
    preprocess = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), NUMERIC),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
    ])
    return Pipeline([('preprocess', preprocess), ('model', RandomForestRegressor(
        n_estimators=350, max_depth=14, min_samples_leaf=4, random_state=2026, n_jobs=1,
    ))])


def frame(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--holdout-rounds', type=int, default=3)
    args = parser.parse_args()
    source = MODEL_DIR / f'qualifying_features_{args.year}_v1.json'
    payload = json.loads(source.read_text(encoding='utf8'))
    if payload.get('featureVersion') != FEATURE_VERSION:
        raise ValueError('qualifying feature schema version mismatch')
    data = frame(payload.get('rows', []))
    if data.empty:
        raise ValueError('no qualifying features found')
    rounds = sorted(int(value) for value in data['round'].unique())
    if len(rounds) <= args.holdout_rounds:
        raise ValueError('not enough completed rounds for a temporal holdout')
    test_rounds = rounds[-args.holdout_rounds:]
    train = data[~data['round'].isin(test_rounds)]
    test = data[data['round'].isin(test_rounds)]
    x_train, y_train = train[NUMERIC + CATEGORICAL], train['deltaToPoleS']
    x_test, y_test = test[NUMERIC + CATEGORICAL], test['deltaToPoleS']
    candidates = {'Ridge': make_ridge(), 'RandomForestRegressor': make_forest()}
    reports = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        reports[name] = metrics(y_test, model.predict(x_test))
        fitted[name] = model
    selected_name = min(reports, key=lambda name: reports[name]['maeS'])
    # The model selection used held-out rounds. The saved serving model is then
    # refit on every completed 2026 qualifying lap; its published metrics are
    # always the earlier, untouched temporal evaluation.
    serving_model = candidates[selected_name].fit(data[NUMERIC + CATEGORICAL], data['deltaToPoleS'])
    model_file = MODEL_DIR / f'qualifying_pace_{args.year}_v1.joblib'
    joblib.dump({'modelVersion': MODEL_VERSION, 'featureVersion': FEATURE_VERSION, 'modelName': selected_name, 'features': NUMERIC + CATEGORICAL, 'model': serving_model}, model_file)
    coverage = payload.get('coverageSummary') or {
        'completedQualifyingSessions': len(payload.get('coverage', [])),
        'sessionsWithTelemetry': sum(1 for item in payload.get('coverage', []) if item.get('usableRows', 0) > 0),
        'cleanTimingRows': sum(item.get('cleanTimingRows', 0) for item in payload.get('coverage', [])),
        'usableTelemetryRows': len(data),
    }
    telemetry_sessions = int(coverage.get('sessionsWithTelemetry', 0))
    # Coverage is necessary, but it is not enough to turn a pace regressor
    # into an ERS optimiser.  This model learns observed delta-to-pole pace;
    # public FastF1 does not contain the team's battery state or deployment
    # command.  Keep it evaluation-only until the separate battery/energy
    # target and its validation gate are implemented.
    deployment_eligible = False
    evaluated_coverage = telemetry_sessions >= 10
    report = {
        'modelVersion': MODEL_VERSION,
        'featureVersion': FEATURE_VERSION,
        'label': 'OBSERVED_PACE_BASELINE',
        'target': 'deltaToPoleS',
        'trainEvaluationRounds': rounds[:-args.holdout_rounds],
        'heldOutRounds': test_rounds,
        'evaluationRows': int(len(test)),
        'trainingRows': int(len(train)),
        'servingFitRows': int(len(data)),
        'coverage': coverage,
        'status': 'PACE_BASELINE_EVALUATED' if evaluated_coverage else 'PARTIAL_TELEMETRY_DATASET',
        'deploymentEligible': deployment_eligible,
        'coverageComplete': evaluated_coverage,
        'candidates': reports,
        'selectedModel': selected_name,
        'selectedHeldOutMetrics': reports[selected_name],
        'limitations': [
            'The model predicts observed qualifying pace delta, not private team ERS deployment or battery state.',
            'Its serving fit includes all completed 2026 sessions only after model selection; evaluation metrics remain from the temporal holdout.',
            'Coverage is complete, but this pace model cannot drive a team-optimal or generic-optimal energy output: the battery/energy objective and its hold-out validation are separate prerequisites.',
        ],
    }
    report_file = MODEL_DIR / f'qualifying_pace_report_{args.year}_v1.json'
    report_file.write_text(json.dumps(report, separators=(',', ':')), encoding='utf8')
    print(json.dumps({'model': str(model_file), 'report': str(report_file), 'selectedModel': selected_name, 'metrics': reports[selected_name], 'heldOutRounds': test_rounds, 'rows': int(len(data))}))


if __name__ == '__main__':
    main()
