"""Map visual Straight Mode turn ranges to a cited corner-distance reference.

The user-supplied maps describe ranges such as T5--T6.  They are useful for a
future simulation overlay, but an image alone is not a metre coordinate. This
tool only creates distance ranges after a separate corner-distance reference
has been supplied. It deliberately does not affect FIA Overtake eligibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VISUAL_EVIDENCE_PATH = ROOT / 'data' / 'straight-mode-visual-evidence.json'
OUT_ROOT = ROOT / 'data' / 'f1-cache' / 'straight-mode-alignment'
SCHEMA_VERSION = 'straight-mode-alignment.v1'


def event_key(year: int, round_number: int, session: str) -> str:
    return f'{year}:{round_number}:{session.lower()}'


def load_visual_reference(year: int, round_number: int, session: str) -> dict[str, Any] | None:
    registry = json.loads(VISUAL_EVIDENCE_PATH.read_text(encoding='utf-8'))
    entry = (registry.get('events') or {}).get(event_key(year, round_number, session))
    return entry if isinstance(entry, dict) else None


def normalise_turn(value: str) -> str:
    return str(value).strip().upper()


def validate_corner_reference(payload: Any) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    """Validate a source-cited mapping of turn identifiers to lap metres."""
    if not isinstance(payload, dict):
        return None, None
    source = payload.get('source')
    if not isinstance(source, dict) or not source.get('url') or not source.get('published'):
        return None, None
    distances = payload.get('turnDistancesM')
    if not isinstance(distances, dict):
        return None, None
    parsed: dict[str, float] = {}
    for turn, value in distances.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None, None
        if numeric < 0:
            return None, None
        parsed[normalise_turn(turn)] = numeric
    try:
        track_length = float(payload.get('trackLengthM'))
    except (TypeError, ValueError):
        return None, None
    if track_length <= 0 or any(distance > track_length for distance in parsed.values()):
        return None, None
    return parsed, {**source, 'trackLengthM': track_length}


def map_turn_ranges(visual: dict[str, Any], corner_reference: Any) -> dict[str, Any]:
    """Return only evidenced map coordinates; never manufacture missing turns."""
    zones = visual.get('zones') if isinstance(visual.get('zones'), list) else []
    if visual.get('enabled') is False:
        return {
            'status': 'STRAIGHT_MODE_DISABLED_REFERENCE',
            'coordinateStatus': 'NO_STRAIGHT_MODE_ZONES',
            'mappedZones': [],
            'liveCommandEligible': False,
        }
    distances, source = validate_corner_reference(corner_reference)
    if distances is None or source is None:
        return {
            'status': 'CORNER_DISTANCE_EVIDENCE_REQUIRED',
            'coordinateStatus': 'TURN_RANGE_ONLY_NOT_TRACK_DISTANCE_MAPPED',
            'mappedZones': [],
            'pendingZoneIds': [zone.get('zoneId') for zone in zones],
            'liveCommandEligible': False,
            'reason': 'A cited 2026 corner-distance reference is required; the visual map is not converted into metres by inference.',
        }
    track_length = source['trackLengthM']
    mapped = []
    missing = []
    for zone in zones:
        start_turn = normalise_turn(zone.get('from', ''))
        end_turn = normalise_turn(zone.get('to', ''))
        start_distance = 0.0 if start_turn == 'START_FINISH' else distances.get(start_turn)
        end_distance = distances.get(end_turn)
        if start_distance is None or end_distance is None:
            missing.append(zone.get('zoneId'))
            continue
        mapped.append({
            'zoneId': zone.get('zoneId'),
            'startDistanceM': round(start_distance, 1),
            'endDistanceM': round(end_distance, 1),
            'wrapsStartFinish': bool(end_distance <= start_distance),
            'mappingMethod': 'TURN_APEX_REFERENCE_FOR_MAP_OVERLAY_ONLY',
            'description': zone.get('description'),
        })
    return {
        'status': 'COMPLETE' if not missing else 'PARTIAL_CORNER_DISTANCE_REFERENCE',
        'coordinateStatus': 'TURN_APEX_REFERENCE_NOT_FIA_LINE',
        'trackLengthM': track_length,
        'cornerDistanceSource': source,
        'mappedZones': mapped,
        'unmappedZoneIds': missing,
        'liveCommandEligible': False,
        'note': 'These are map-overlay coordinates only. FIA Overtake detection and activation lines require their own cited event data.',
    }


def build_payload(year: int, round_number: int, session: str, corner_reference: Any = None) -> dict[str, Any]:
    visual = load_visual_reference(year, round_number, session)
    return {
        'schemaVersion': SCHEMA_VERSION,
        'year': year,
        'round': round_number,
        'session': session,
        'eventKey': event_key(year, round_number, session),
        'visualReferenceAvailable': visual is not None,
        'alignment': map_turn_ranges(visual, corner_reference) if visual else {
            'status': 'VISUAL_REFERENCE_NOT_AVAILABLE',
            'mappedZones': [],
            'liveCommandEligible': False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--session', default='R')
    parser.add_argument('--corner-reference', type=Path)
    parser.add_argument('--output', action='store_true')
    args = parser.parse_args()
    reference = json.loads(args.corner_reference.read_text(encoding='utf-8')) if args.corner_reference else None
    payload = build_payload(args.year, args.round, args.session, reference)
    if args.output:
        target = OUT_ROOT / str(args.year) / f'{args.round}_{args.session.lower()}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
