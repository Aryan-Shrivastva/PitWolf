"""Attach observed outcomes to FIA-line-aligned replay opportunities.

The source action/outcome labels were originally derived at lap start.  This
builder keeps them useful for research comparison while making the temporal
boundary explicit: they are associated same-pair/same-lap observations, not
proof that an Overtake Mode deployment caused a later position change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / 'data' / 'f1-cache'
ZONE_ROOT = CACHE_ROOT / 'zone-opportunities'
DECISION_ROOT = CACHE_ROOT / 'decision-points'
OUT_ROOT = CACHE_ROOT / 'zone-replays'
SCHEMA_VERSION = 'zone-replay.v1'


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def outcome_index(decision_payload: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    rows = decision_payload.get('analysisRows') if isinstance(decision_payload.get('analysisRows'), list) else []
    index = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            key = (int(row.get('lap')), str(row.get('driver')), str(row.get('defender')))
        except (TypeError, ValueError):
            continue
        # A cache should have one ordered pair per lap. Keep the first stable
        # record if malformed duplicates occur instead of silently selecting a
        # target-bearing one.
        index.setdefault(key, row)
    return index


def associated_state(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            'status': 'NO_LAP_START_ASSOCIATION',
            'causalStatus': 'NOT_AVAILABLE',
            'energyState': None,
            'observedOutcome': None,
        }
    return {
        'status': 'LAP_START_ROW_ASSOCIATED_NOT_ZONE_CAUSAL',
        'causalStatus': 'OBSERVATIONAL_ASSOCIATION_ONLY',
        'featureCutoff': row.get('featureCutoff'),
        'energyState': {
            'attackerSoCMj': row.get('attackerSoCMj'),
            'defenderSoCMj': row.get('defenderSoCMj'),
            'energyDeltaMj': row.get('energyDeltaMj'),
            'source': 'MODELLED_LAP_START_SURROGATE_NOT_PRIVATE_TELEMETRY',
        },
        'decisionState': {
            'position': row.get('position'),
            'defenderPosition': row.get('defenderPosition'),
            'closingRateS': row.get('closingRateS'),
            'speedDeltaKph': row.get('speedDeltaKph'),
            'attackerCompound': row.get('attackerCompound'),
            'defenderCompound': row.get('defenderCompound'),
            'pitDistorted': row.get('pitDistorted'),
            'attackerTrackStatus': row.get('attackerTrackStatus'),
            'defenderTrackStatus': row.get('defenderTrackStatus'),
        },
        'observedOutcome': {
            'outcomeLabelEligible': row.get('outcomeLabelEligible'),
            'label': row.get('label'),
            'passedNow': row.get('passedNow'),
            'held': row.get('held'),
            'observedPassLap': row.get('observedPassLap'),
            'observedLeadLaps': row.get('observedLeadLaps'),
            'outcomeExclusionReasons': row.get('outcomeExclusionReasons') or [],
            'source': 'LAP_CLASSIFICATION_OUTCOME',
        },
    }


def build_replay(zone_payload: dict[str, Any], decision_payload: dict[str, Any]) -> dict[str, Any]:
    index = outcome_index(decision_payload)
    replays = []
    associated = 0
    for opportunity in zone_payload.get('opportunities') or []:
        try:
            key = (int(opportunity.get('lap')), str(opportunity.get('driver')), str(opportunity.get('defender')))
        except (TypeError, ValueError):
            continue
        state = associated_state(index.get(key))
        associated += state['status'] != 'NO_LAP_START_ASSOCIATION'
        replays.append({
            **opportunity,
            'replayState': state,
            'liveCommandEligible': False,
            'liveCommandBoundary': 'RETROSPECTIVE_ZONE_REPLAY_ONLY',
        })
    return {
        'schemaVersion': SCHEMA_VERSION,
        'year': zone_payload.get('year'),
        'round': zone_payload.get('round'),
        'session': zone_payload.get('session'),
        'eventName': zone_payload.get('eventName'),
        'trackLengthM': zone_payload.get('trackLengthM'),
        'trackLengthSource': zone_payload.get('trackLengthSource'),
        'availability': zone_payload.get('availability'),
        'sourceOpportunityCount': len(zone_payload.get('opportunities') or []),
        'lapStartAssociatedCount': associated,
        'unassociatedCount': len(replays) - associated,
        'replays': replays,
        'methodology': {
            'zoneState': 'SAME_TRACK_DISTANCE_INTERPOLATED_CAR_DATA',
            'outcomes': 'LAP_START_OUTCOME_ASSOCIATION',
            'causalStatus': 'OBSERVATIONAL_ONLY',
            'energy': 'MODELLED_SURROGATE_NOT_MEASURED_TEAM_SOC',
        },
    }


def build_for_event(year: int, round_number: int, session: str = 'R') -> dict[str, Any]:
    zone_path = ZONE_ROOT / str(year) / f'{round_number}_{session.lower()}.json'
    decision_path = DECISION_ROOT / str(year) / f'{round_number}_{session.lower()}.json'
    return build_replay(read_json(zone_path), read_json(decision_path))


def write_event(year: int, round_number: int, session: str = 'R') -> Path:
    payload = build_for_event(year, round_number, session)
    target = OUT_ROOT / str(year) / f'{round_number}_{session.lower()}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--session', default='R')
    parser.add_argument('--output', action='store_true')
    args = parser.parse_args()
    payload = build_for_event(args.year, args.round, args.session)
    if args.output:
        write_event(args.year, args.round, args.session)
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
