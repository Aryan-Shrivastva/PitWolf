"""Validate user-supplied Straight Mode map references stay non-authoritative."""

from __future__ import annotations

import json
from pathlib import Path

from extract_zone_opportunities import rule_context_for
from zone_alignment import zone_alignment_status


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / 'data' / 'straight-mode-visual-evidence.json'
EXPECTED_ZONE_COUNTS = {
    '2026:1:r': 5, '2026:2:r': 4, '2026:3:r': 2, '2026:5:r': 4,
    '2026:6:r': 0, '2026:7:r': 4, '2026:8:r': 4, '2026:9:r': 4,
    '2026:10:r': 5, '2026:11:r': 4, '2026:12:r': 2,
}


def main() -> None:
    payload = json.loads(EVIDENCE_PATH.read_text(encoding='utf-8'))
    assert payload.get('schemaVersion') == 'straight-mode-visual-evidence.v1'
    provenance = payload.get('provenance') or {}
    assert provenance.get('legalStatus') == 'NOT_FIA_EVENT_APPENDIX'
    events = payload.get('events') or {}
    assert set(events) == set(EXPECTED_ZONE_COUNTS), 'unexpected event coverage'
    for event_key, expected_count in EXPECTED_ZONE_COUNTS.items():
        entry = events[event_key]
        assert entry.get('eventSpecificDataLoaded') is None, f'{event_key} must not set live FIA state'
        assert len(entry.get('zones') or []) == expected_count, f'{event_key} zone count mismatch'
        assert entry.get('enabled') is (expected_count > 0), f'{event_key} enabled mismatch'
        coordinate_status = entry.get('coordinateStatus')
        if expected_count:
            assert coordinate_status == 'TURN_RANGE_ONLY_NOT_TRACK_DISTANCE_MAPPED'
            for zone in entry['zones']:
                assert zone.get('zoneId') and zone.get('from') and zone.get('to')
        else:
            assert coordinate_status == 'NO_STRAIGHT_MODE_ZONES'
    # The extractor's direct Python entry point must carry the same map
    # evidence as the HTTP API, while retaining its strict FIA line gate.
    australia = rule_context_for(2026, 1, 'R')
    assert len(australia['straightModeEvidence']['zones']) == 5
    australia_status = zone_alignment_status(australia, 5278.0)
    assert australia_status['telemetryAlignmentAvailable'] is False
    assert len(australia_status['straightModeMapReference']['zones']) == 5
    print(f'STRAIGHT_MODE_EVIDENCE_OK events={len(events)} zones={sum(EXPECTED_ZONE_COUNTS.values())} liveGate=BLOCKED')


if __name__ == '__main__':
    main()
