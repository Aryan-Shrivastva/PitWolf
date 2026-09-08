"""Audit generated cited-FIA-line zone replay caches."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / 'data' / 'f1-cache' / 'zone-replays' / '2026'
EXPECTED_ROUNDS = {2, 4, 5, 6, 8, 10, 11}


def main() -> None:
    found = set()
    replay_count = 0
    for path in sorted(ROOT.glob('*_r.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        round_number = int(payload['round'])
        found.add(round_number)
        assert payload.get('schemaVersion') == 'zone-replay.v1'
        assert payload.get('availability', {}).get('liveCommandEligible') is False
        assert payload.get('methodology', {}).get('causalStatus') == 'OBSERVATIONAL_ONLY'
        for replay in payload.get('replays') or []:
            assert replay.get('liveCommandEligible') is False
            assert replay.get('alignmentMethod') == 'SAME_TRACK_DISTANCE_INTERPOLATED_CAR_DATA'
            assert replay.get('alignmentConfidence') == 'MEDIUM'
        replay_count += len(payload.get('replays') or [])
    assert found == EXPECTED_ROUNDS
    print(f'ZONE_REPLAY_CACHE_OK events={len(found)} replays={replay_count} liveGate=BLOCKED')


if __name__ == '__main__':
    main()
