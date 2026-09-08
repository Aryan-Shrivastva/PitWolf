"""Deterministic tests for non-authoritative Straight Mode distance mapping."""

from map_straight_mode_ranges import build_payload, map_turn_ranges


def main() -> None:
    pending = build_payload(2026, 1, 'R')
    assert pending['alignment']['status'] == 'CORNER_DISTANCE_EVIDENCE_REQUIRED'
    assert pending['alignment']['liveCommandEligible'] is False
    reference = {
        'trackLengthM': 5000,
        'turnDistancesM': {'T1': 400, 'T2': 800, 'T3': 1200, 'T5': 1900, 'T6': 2200, 'T8': 3000, 'T9': 3300, 'T10': 3800, 'T11': 4100},
        'source': {'url': 'https://example.invalid/corner-reference', 'published': '2026-01-01'},
    }
    visual = {'enabled': True, 'zones': [{'zoneId': 'SM-1', 'from': 'START_FINISH', 'to': 'T1'}, {'zoneId': 'SM-2', 'from': 'T2', 'to': 'T3'}]}
    mapped = map_turn_ranges(visual, reference)
    assert mapped['status'] == 'COMPLETE'
    assert mapped['mappedZones'][0]['startDistanceM'] == 0.0
    assert mapped['mappedZones'][1]['endDistanceM'] == 1200.0
    assert mapped['liveCommandEligible'] is False
    monaco = build_payload(2026, 6, 'R')
    assert monaco['alignment']['status'] == 'STRAIGHT_MODE_DISABLED_REFERENCE'
    print('STRAIGHT_MODE_MAPPING_OK pending=PASS mapped=PASS live-gate=BLOCKED')


if __name__ == '__main__':
    main()
