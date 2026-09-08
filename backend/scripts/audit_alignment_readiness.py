"""Audit PitWolf's reproducible research baseline without masking live-data gates.

This is intentionally not a "live command ready" audit. It proves the
internally reproducible contract (cache, feature schema, trained artifact and
temporal split) while listing the externally sourced evidence still required
for a FIA event command.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from extract_decision_points import OUT_ROOT, SCHEMA_VERSION
from overtake_feature_schema import FEATURE_SCHEMA_VERSION


ROOT = OUT_ROOT.parent
MODEL_PATH = ROOT / 'models' / 'overtake_rf.joblib'
REPORT_PATH = ROOT / 'models' / 'overtake_report.json'
RULE_CONTEXT_PATH = ROOT.parent / 'overtake-rule-context.json'
STRAIGHT_MODE_VISUAL_EVIDENCE_PATH = ROOT.parent / 'straight-mode-visual-evidence.json'


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    # The frozen classifier and its UI selectors analyse race sessions only.
    # Other cached session types are retained as separate raw evidence and
    # must not change this race-baseline readiness result.
    files = sorted(OUT_ROOT.glob('*/*_r.json'))
    non_race_files = sorted(path.name for path in OUT_ROOT.glob('*/*.json')
                            if not path.name.endswith('_r.json'))
    cache_failures = []
    clean_rows = 0
    analysis_rows = 0
    for path in files:
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError) as error:
            cache_failures.append(f'{path.name}: unreadable ({error})')
            continue
        if payload.get('schemaVersion') != SCHEMA_VERSION:
            cache_failures.append(f'{path.name}: decision schema is stale')
        if payload.get('featureSchemaVersion') != FEATURE_SCHEMA_VERSION:
            cache_failures.append(f'{path.name}: feature schema is stale')
        if not isinstance(payload.get('analysisRows'), list):
            cache_failures.append(f'{path.name}: analysisRows missing')
        else:
            analysis_rows += len(payload['analysisRows'])
        if not isinstance(payload.get('rows'), list):
            cache_failures.append(f'{path.name}: clean rows missing')
        else:
            clean_rows += len(payload['rows'])
        if not isinstance(payload.get('participants'), list) or not isinstance(payload.get('driverSummaries'), list):
            cache_failures.append(f'{path.name}: participant evidence missing')

    model_failures = []
    artifact = None
    report = None
    try:
        artifact = joblib.load(MODEL_PATH)
        if artifact.get('featureSchemaVersion') != FEATURE_SCHEMA_VERSION:
            model_failures.append('trained model feature schema is stale')
    except (OSError, ValueError, EOFError) as error:
        model_failures.append(f'trained model unreadable ({error})')
    try:
        report = read_json(REPORT_PATH)
        split = report.get('temporalSplit') or {}
        if split.get('trainYears') != list(range(2018, 2026)) or split.get('testYears') != [2026]:
            model_failures.append('model report does not have the frozen 2018–2025 / 2026 split')
        if report.get('featureSchemaVersion') != FEATURE_SCHEMA_VERSION:
            model_failures.append('model report feature schema is stale')
    except (OSError, json.JSONDecodeError) as error:
        model_failures.append(f'model report unreadable ({error})')

    rule_context = read_json(RULE_CONTEXT_PATH)
    events = rule_context.get('events') if isinstance(rule_context.get('events'), dict) else {}
    loaded_event_keys = sorted(
        key for key, entry in events.items()
        if isinstance(entry, dict) and entry.get('eventSpecificDataLoaded') is True
    )
    zone_evidence_keys = sorted(
        key for key, entry in events.items()
        if isinstance(entry, dict) and isinstance(entry.get('zoneEvidence'), dict)
        and entry['zoneEvidence'].get('zoneEvidenceLoaded') is True
    )
    visual_evidence = read_json(STRAIGHT_MODE_VISUAL_EVIDENCE_PATH)
    visual_events = visual_evidence.get('events') if isinstance(visual_evidence.get('events'), dict) else {}
    visual_evidence_keys = sorted(visual_events)
    visual_zone_count = sum(
        len(entry.get('zones') or []) for entry in visual_events.values()
        if isinstance(entry, dict)
    )
    external_gates = []
    if not loaded_event_keys:
        external_gates.append('No verified FIA 2026 event appendix is imported; live Overtake Mode commands remain blocked.')
    external_gates.extend([
        'Public data does not include measured team battery SoC or deployment maps.',
        'RaceSimView remains deferred until it consumes the canonical replay state and verified event-zone data.',
    ])

    failures = cache_failures + model_failures
    result = {
        'status': 'BASELINE_READY_WITH_EXTERNAL_GATES' if not failures else 'NOT_READY',
        'decisionPointSchemaVersion': SCHEMA_VERSION,
        'featureSchemaVersion': FEATURE_SCHEMA_VERSION,
        'cachedRaces': len(files),
        'nonRaceCacheFilesNotInBaseline': non_race_files,
        'analysisRows': analysis_rows,
        'cleanTrainingRows': clean_rows,
        'modelArtifactPresent': artifact is not None,
        'temporalSplit': (report or {}).get('temporalSplit'),
        'verifiedFiaEventAppendices': loaded_event_keys,
        'verifiedFiaZoneEvidenceForRetrospectiveAlignment': zone_evidence_keys,
        'userSuppliedStraightModeMapReferences': visual_evidence_keys,
        'userSuppliedStraightModeTurnRangeCount': visual_zone_count,
        'failures': failures,
        'externalEvidenceGates': external_gates,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
