import argparse
import json
import math
import re
import urllib.request
from pathlib import Path

import fastf1
import numpy as np

from fetch_f1_session import CACHE_DIR, clean

GP_TEMPO_CIRCUIT_URL = 'https://www.gp-tempo.com/api/circuit?year={year}&event={round}'
VISUAL_EVIDENCE_PATH = Path(__file__).resolve().parents[1] / 'data' / 'straight-mode-visual-evidence.json'


def visual_reference(year, round_number, session_name):
    """Read the user's turn-range maps without promoting them to FIA data."""
    try:
        evidence = json.loads(VISUAL_EVIDENCE_PATH.read_text(encoding='utf8'))
        return evidence.get('events', {}).get(f'{year}:{round_number}:{session_name.lower()}')
    except Exception:
        return None


def geometry_corners(distances, xs, ys, count):
    """Find turn apices directly on the recorded circuit geometry.

    The supplied 2026 maps establish the numbered turn count and order. This
    derives the visible marker locations from the actual recorded centre line,
    rather than borrowing an unrelated third-party circuit layout.
    """
    if not count or len(distances) < 30:
        return []
    radius = max(3, min(12, len(distances) // 90))
    scores = []
    for index in range(radius, len(distances) - radius):
        before = np.array([xs[index] - xs[index - radius], ys[index] - ys[index - radius]])
        after = np.array([xs[index + radius] - xs[index], ys[index + radius] - ys[index]])
        before_norm = np.linalg.norm(before)
        after_norm = np.linalg.norm(after)
        if before_norm < 1 or after_norm < 1:
            continue
        cross = before[0] * after[1] - before[1] * after[0]
        dot = before[0] * after[0] + before[1] * after[1]
        scores.append((abs(math.atan2(cross, dot)), index))
    if not scores:
        return []
    track_length = float(distances[-1])
    min_separation = max(28.0, track_length / max(1, count * 7))
    selected = []
    for _, index in sorted(scores, reverse=True):
        distance = float(distances[index])
        if any(min(abs(distance - float(distances[other])), track_length - abs(distance - float(distances[other]))) < min_separation for other in selected):
            continue
        selected.append(index)
        if len(selected) == count:
            break
    if len(selected) != count:
        return []
    selected.sort(key=lambda index: distances[index])
    return [
        {
            'n': str(number),
            'd': round(float(distances[index]), 1),
            'x': round(float(xs[index]), 2),
            'y': round(float(ys[index]), 2),
        }
        for number, index in enumerate(selected, start=1)
    ]


def corner_distance(reference, corners, track_length):
    text = str(reference or '').upper()
    if 'START' in text or 'FINISH' in text:
        return 0.0
    numbers = [int(number) for number in re.findall(r'T(\d+)', text)]
    by_number = {int(corner['n']): float(corner['d']) for corner in corners if str(corner.get('n', '')).isdigit()}
    values = [by_number[number] for number in numbers if number in by_number]
    if not values:
        return None
    if 'BETWEEN' in text and len(values) > 1:
        return round(sum(values[:2]) / 2, 1)
    return values[0]


def visual_overlay(reference, corners, track_length):
    if not reference:
        return {'status': 'NO_VISUAL_REFERENCE', 'zones': [], 'markers': []}
    zones = []
    for zone in reference.get('zones') or []:
        start = corner_distance(zone.get('from'), corners, track_length)
        end = corner_distance(zone.get('to'), corners, track_length)
        if start is None or end is None:
            continue
        zones.append({
            'id': zone.get('zoneId'), 'startD': start, 'endD': end,
            'description': zone.get('description') or f"{zone.get('from')} → {zone.get('to')}",
        })
    marker_reference = reference.get('overtakeMarkerReference') or {}
    markers = []
    for marker_type in ('detection', 'activation'):
        distance = corner_distance(marker_reference.get(marker_type), corners, track_length)
        if distance is not None:
            markers.append({'type': marker_type.upper(), 'd': distance, 'reference': marker_reference.get(marker_type)})
    return {
        'status': reference.get('coordinateStatus', 'TURN_RANGE_ONLY_NOT_TRACK_DISTANCE_MAPPED'),
        'mode': reference.get('status'),
        'source': 'USER_VISUAL_TURN_REFERENCE',
        'zones': zones,
        'markers': markers,
        'note': 'Turn-range overlay from user-supplied track reference. It is visual analysis only, not an FIA distance-aligned command line.',
    }


def fetch_fallback_corners(year, round_number, distances, xs, ys):
    request = urllib.request.Request(
        GP_TEMPO_CIRCUIT_URL.format(year=year, round=round_number),
        headers={'User-Agent': 'Mozilla/5.0 (PitWolf track map)'},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.load(response)
    except Exception:
        return [], 'none'
    corners = []
    for corner in payload.get('Corners') or []:
        number = corner.get('Number')
        distance = corner.get('Distance')
        if number is None or distance is None:
            continue
        index = int(np.searchsorted(distances, float(distance)))
        index = max(0, min(index, len(distances) - 1))
        corners.append({
            'n': f"{int(number)}{corner.get('Letter') or ''}",
            'd': round(float(distances[index]), 1),
            'x': round(float(xs[index]), 2),
            'y': round(float(ys[index]), 2),
        })
    return corners, 'gp-tempo' if corners else 'none'


def fetch_circuit_info_corners(session, distances, xs, ys):
    """Use FastF1/MultiViewer's numbered circuit markers.

    Corner *numbers* cannot be safely reconstructed from curvature alone: a
    complex, low-speed section can contain several sharp geometry points that
    are not individual FIA turn markers. CircuitInfo supplies the published
    turn sequence and maps every marker to a telemetry distance on this exact
    circuit. We then place it on the recorded centre line used by the player.
    """
    try:
        circuit_info = session.get_circuit_info()
        source_corners = circuit_info.corners if circuit_info is not None else None
    except Exception:
        source_corners = None
    if source_corners is None or source_corners.empty:
        return [], 'none'

    corners = []
    for _, corner in source_corners.iterrows():
        number = corner.get('Number')
        distance = corner.get('Distance')
        if number is None or distance is None or not np.isfinite(float(distance)):
            continue
        index = int(np.searchsorted(distances, float(distance)))
        index = max(0, min(index, len(distances) - 1))
        letter = clean(corner.get('Letter')) or ''
        corners.append({
            'n': f"{int(number)}{letter}",
            'd': round(float(distances[index]), 1),
            'x': round(float(xs[index]), 2),
            'y': round(float(ys[index]), 2),
        })
    corners.sort(key=lambda corner: (int(re.match(r'\d+', corner['n']).group()), corner['n']))
    return corners, 'fastf1-circuit-info / multiviewer-numbered-corners' if corners else 'none'


def pick_lap_with_position(session):
    laps = session.laps
    if laps is None or laps.empty:
        return None, None
    candidates = laps[laps['PitInTime'].isna() & laps['PitOutTime'].isna()]
    if 'IsAccurate' in candidates.columns:
        candidates = candidates[candidates['IsAccurate'].isna() | candidates['IsAccurate'].astype(bool)]
    candidates = candidates[candidates['LapTime'].notna()].sort_values('LapTime')
    for _, row in candidates.head(10).iterrows():
        match = session.laps[(session.laps['Driver'] == row['Driver']) & (session.laps['LapNumber'] == row['LapNumber'])]
        if match.empty:
            continue
        try:
            tel = match.iloc[0].get_telemetry()
        except Exception:
            continue
        if tel is None or 'X' not in tel.columns or 'Y' not in tel.columns:
            continue
        tel = tel[tel['X'].notna() & tel['Y'].notna() & tel['Speed'].notna() & tel['Distance'].notna()]
        if len(tel) < 50:
            continue
        return row, tel
    return None, None


def build_trackmap_payload(year, round_number, session_name):
    event = fastf1.get_event(year, round_number)
    session = event.get_session(session_name)
    try:
        session.load(laps=True, telemetry=True, weather=False, messages=False)
    except Exception:
        return {'error': 'no_position_data', 'points': [], 'corners': []}

    row, tel = pick_lap_with_position(session)
    if row is None:
        return {'error': 'no_position_data', 'points': [], 'corners': []}

    if len(tel) > 500:
        keep = np.unique(np.linspace(0, len(tel) - 1, 500).round().astype(int))
        tel = tel.iloc[keep]

    distances = tel['Distance'].to_numpy(dtype=float)
    xs = tel['X'].to_numpy(dtype=float)
    ys = tel['Y'].to_numpy(dtype=float)
    speeds = tel['Speed'].to_numpy(dtype=float)
    track_length = float(distances[-1])

    reference = visual_reference(year, round_number, session_name)
    corners, corner_source = fetch_circuit_info_corners(session, distances, xs, ys)
    if not corners:
        corners, corner_source = fetch_fallback_corners(year, round_number, distances, xs, ys)
    overlay = visual_overlay(reference, corners, track_length)

    return {
        'trackLength': round(track_length, 1),
        'points': [
            {
                'd': round(float(d), 1),
                'x': round(float(x), 2),
                'y': round(float(y), 2),
                's': round(float(s), 1),
            }
            for d, x, y, s in zip(distances, xs, ys, speeds)
        ],
        'corners': corners,
        'cornerSource': corner_source,
        'visualOverlay': overlay,
        'source': f"{clean(row['Driver'])} L{int(row['LapNumber'])}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--session', required=True)
    args = parser.parse_args()

    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    print(json.dumps(build_trackmap_payload(args.year, args.round, args.session)))


if __name__ == '__main__':
    import sys
    try:
        main()
    except Exception as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        raise SystemExit(1)
