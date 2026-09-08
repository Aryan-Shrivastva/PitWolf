"""Train a transparent, separate historical pit-window (BOX) baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from extract_box_candidates import OUT_ROOT, SCHEMA_VERSION


ROOT = OUT_ROOT.parent
MODELS = ROOT / 'models'
MODEL_PATH = MODELS / 'box_logistic.joblib'
REPORT_PATH = MODELS / 'box_report.json'
NUMERIC = ['lapFraction', 'tyreLifeLaps', 'stint', 'position', 'previousLapTimeS']
CATEGORICAL = ['compound']


def load_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(OUT_ROOT.glob('*/*_r.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get('schemaVersion') != SCHEMA_VERSION:
            continue
        rows.extend(row for row in payload.get('rows', []) if row.get('outcomeEligible'))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame['lapFraction'] = frame['lap'] / frame.groupby(['year', 'round', 'session'])['lap'].transform('max').clip(lower=1)
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame.get(column), errors='coerce').fillna(0.0)
    frame['compound'] = frame['compound'].fillna('UNKNOWN').astype(str)
    frame['pitNextLapObserved'] = frame['pitNextLapObserved'].astype(int)
    return frame


def metric_block(y_true, probability) -> dict:
    predicted = (probability >= 0.5).astype(int)
    return {
        'accuracy': round(float(accuracy_score(y_true, predicted)), 4),
        'precision': round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        'recall': round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        'f1': round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        'brier': round(float(brier_score_loss(y_true, probability)), 4),
        'auc': round(float(roc_auc_score(y_true, probability)), 4) if len(set(y_true)) == 2 else None,
    }


def main() -> None:
    frame = load_rows()
    if frame.empty:
        raise SystemExit('No eligible BOX candidates available; run batch_extract_box_candidates.py first.')
    train = frame[frame['year'] <= 2025].copy()
    test = frame[frame['year'] == 2026].copy()
    if train.empty or test.empty or train['pitNextLapObserved'].nunique() < 2:
        raise SystemExit('BOX temporal split needs both 2018–2025 training candidates and 2026 held-out candidates.')
    features = NUMERIC + CATEGORICAL
    pipeline = Pipeline([
        ('prepare', ColumnTransformer([
            ('numeric', StandardScaler(), NUMERIC),
            ('compound', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ])),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    ])
    pipeline.fit(train[features], train['pitNextLapObserved'])
    probability = pipeline.predict_proba(test[features])[:, 1]
    prevalence = float(train['pitNextLapObserved'].mean())
    report = {
        'model': 'class-weighted logistic regression',
        'purpose': 'Historical next-lap pit-window feasibility baseline; separate from ATTACK/SAVE/DELAY.',
        'boundary': 'OBSERVED_PIT_IN_LABEL_NOT_OPTIMAL_STRATEGY_OR_LIVE_BOX_COMMAND',
        'temporalSplit': {'trainYears': list(range(2018, 2026)), 'testYears': [2026], 'strategy': 'STRICT_TEMPORAL_BY_SEASON'},
        'features': features,
        'trainingRows': int(len(train)), 'heldOutRows': int(len(test)),
        'trainingPitRate': round(prevalence, 4), 'heldOutPitRate': round(float(test['pitNextLapObserved'].mean()), 4),
        'heldOutMetrics': metric_block(test['pitNextLapObserved'].to_numpy(), probability),
        'alwaysHoldBaseline': metric_block(test['pitNextLapObserved'].to_numpy(), np.zeros(len(test))),
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': pipeline, 'features': features, 'report': report}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
