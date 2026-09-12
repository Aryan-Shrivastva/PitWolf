"""User-supplied Straight Mode ranges for qualifying battery policy.

The supplied maps name turn-to-turn ranges, not FIA control lines.  The user
has explicitly chosen to use them as a PitWolf qualifying deployment policy:
ERS deployment is allowed only inside a mapped range.  This module projects
those turn ranges onto the recorded session's circuit distances but never
labels them as FIA Overtake zones or a measured team deployment command.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISUAL_EVIDENCE_PATH = ROOT / 'data' / 'straight-mode-visual-evidence.json'


def _turn_number(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _turn_distances(session, trace_length_m):
    """Read FastF1's circuit-corner distances and align them to this lap."""
    try:
        circuit = session.get_circuit_info()
        corners = circuit.corners if circuit is not None else None
    except Exception:
        corners = None
    if corners is None or corners.empty:
        return {}

    raw = {}
    for _, corner in corners.iterrows():
        turn = _turn_number(corner.get('Number'))
        distance = corner.get('Distance')
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            continue
        if turn is not None and distance >= 0:
            raw[turn] = distance
    if not raw:
        return {}

    furthest_corner = max(raw.values())
    # A final apex is normally *before* the timing line, so it must not be
    # treated as circuit length and scaled up to the end of the telemetry.
    # FastF1 circuit information and lap telemetry share a distance origin in
    # normal cases. Only scale down when a source-coordinate mismatch would
    # otherwise put the final corner beyond the recorded lap.
    scale = 1.0
    if trace_length_m > 0 and furthest_corner > trace_length_m * 1.02:
        scale = float(trace_length_m) / furthest_corner
    return {turn: min(float(trace_length_m), distance * scale) for turn, distance in raw.items()}


def _reference(year, round_number):
    try:
        payload = json.loads(VISUAL_EVIDENCE_PATH.read_text(encoding='utf8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    # The user supplied this map for the circuit, originally under the race
    # key. The explicit battery policy applies the same circuit ranges to the
    # qualifying lap; it is not a race Overtake activation rule.
    return (payload.get('events') or {}).get(f'{year}:{round_number}:r')


def qualifying_straight_mode_policy(session, *, year, round_number, trace_length_m):
    """Return the user policy and its telemetry-distance windows.

    An absent or unmappable reference intentionally returns an empty window
    list. Under the user's policy that means ICE-only qualifying deployment,
    rather than inventing a straight-mode zone.
    """
    reference = _reference(year, round_number)
    base = {
        'policy': 'USER_STRAIGHT_MODE_ZONES_ONLY',
        'isFiaRule': False,
        'referenceKey': f'{year}:{round_number}:r',
        'windows': [],
    }
    if reference is None:
        return {**base, 'status': 'USER_STRAIGHT_MODE_REFERENCE_MISSING'}
    if reference.get('enabled') is False:
        return {**base, 'status': 'USER_STRAIGHT_MODE_DISABLED_FOR_CIRCUIT'}

    turn_distances = _turn_distances(session, trace_length_m)
    if not turn_distances:
        return {**base, 'status': 'USER_STRAIGHT_MODE_CORNERS_UNAVAILABLE'}

    windows, missing = [], []
    for zone in reference.get('zones') or []:
        start_ref = str(zone.get('from') or '').strip().upper()
        end_turn = _turn_number(str(zone.get('to') or '').strip().upper().replace('T', ''))
        start_turn = _turn_number(start_ref.replace('T', ''))
        start = 0.0 if start_ref == 'START_FINISH' else turn_distances.get(start_turn)
        end = turn_distances.get(end_turn)
        if start is None or end is None:
            missing.append(zone.get('zoneId'))
            continue
        windows.append({
            'zoneId': zone.get('zoneId'),
            'startDistanceM': round(float(start), 1),
            'endDistanceM': round(float(end), 1),
            'wrapsStartFinish': bool(end <= start),
            'description': zone.get('description') or f"{zone.get('from')} → {zone.get('to')}",
        })
    return {
        **base,
        'status': 'USER_STRAIGHT_MODE_POLICY_LOADED' if windows and not missing else 'USER_STRAIGHT_MODE_POLICY_PARTIAL',
        'windows': windows,
        'unmappedZoneIds': missing,
        'referenceStatus': reference.get('status'),
        'referenceCoordinateStatus': reference.get('coordinateStatus'),
        'mappingBasis': reference.get('mappingBasis', 'USER_SUPPLIED_TURN_RANGE_REFERENCE'),
    }
