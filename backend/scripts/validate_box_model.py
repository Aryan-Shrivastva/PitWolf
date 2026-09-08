"""Validate BOX model provenance and its no-command boundary."""

from __future__ import annotations

import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1] / 'data' / 'f1-cache' / 'models'


def main() -> None:
    report = json.loads((ROOT / 'box_report.json').read_text(encoding='utf-8'))
    artifact = joblib.load(ROOT / 'box_logistic.joblib')
    assert report['boundary'] == 'OBSERVED_PIT_IN_LABEL_NOT_OPTIMAL_STRATEGY_OR_LIVE_BOX_COMMAND'
    assert report['temporalSplit']['trainYears'] == list(range(2018, 2026))
    assert report['temporalSplit']['testYears'] == [2026]
    assert report['trainingRows'] > 0 and report['heldOutRows'] > 0
    assert artifact['features'] == report['features']
    assert 'pitNextLapObserved' not in report['features']
    assert report['heldOutMetrics']['auc'] is not None
    print('BOX_MODEL_OK temporal=PASS noTargetLeakage=PASS separateCommandBoundary=PASS')


if __name__ == '__main__':
    main()
