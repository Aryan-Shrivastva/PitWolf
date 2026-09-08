"""Extract FIA-zone-aligned overtake opportunities from public telemetry.

For an event with a verified FIA Power Unit Information entry, this calculates
the time at which each car crosses the official Detection Line and pairs each
car with the car immediately ahead at that same line.  The resulting gap is a
real telemetry-derived same-track-distance measurement, not the older
lap-start timing proxy.  It remains retrospective public-data evidence; it
does not reveal private battery state or issue a live command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd

from fetch_f1_session import CACHE_DIR
from zone_alignment import zone_alignment_status


ROOT = Path(CACHE_DIR).parent
OUT_ROOT = ROOT / 'zone-opportunities'
RULE_CONTEXT_PATH = ROOT.parent / 'overtake-rule-context.json'
STRAIGHT_MODE_VISUAL_EVIDENCE_PATH = ROOT.parent / 'straight-mode-visual-evidence.json'
SCHEMA_VERSION = 'zone-opportunity.v1'


def seconds(value: Any) -> float | None:
    try:
        if hasattr(value, 'total_seconds'):
            result = float(value.total_seconds())
        else:
            result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def rule_context_for(year: int, round_number: int, session_name: str) -> dict[str, Any]:
    registry = json.loads(RULE_CONTEXT_PATH.read_text(encoding='utf-8'))
    visual_registry = json.loads(STRAIGHT_MODE_VISUAL_EVIDENCE_PATH.read_text(encoding='utf-8'))
    defaults = registry.get('defaults') if isinstance(registry.get('defaults'), dict) else {}
    base = defaults.get('2026' if year >= 2026 else 'historical') or {}
    key = f'{year}:{round_number}:{session_name.lower()}'
    event = (registry.get('events') or {}).get(key) or {}
    straight_mode = (visual_registry.get('events') or {}).get(key)
    return {
        **base,
        **event,
        **({'straightModeEvidence': straight_mode} if isinstance(straight_mode, dict) else {}),
        'eventKey': key,
    }


def crossing_at_distance(lap: Any, target_distance_m: float) -> dict[str, float] | None:
    """Interpolate one lap's session time and speed at a track distance."""
    try:
        telemetry = lap.get_telemetry()
    except Exception:
        return None
    if telemetry is None or telemetry.empty or 'Distance' not in telemetry:
        return None
    telemetry = telemetry.dropna(subset=['Distance']).sort_values('Distance')
    distances = telemetry['Distance'].to_numpy(dtype=float)
    if len(distances) < 2 or target_distance_m < distances[0] or target_distance_m > distances[-1]:
        return None
    time_column = 'SessionTime' if 'SessionTime' in telemetry else 'Time'
    values = np.asarray([seconds(value) for value in telemetry[time_column]], dtype=object)
    if any(value is None for value in values):
        return None
    times = values.astype(float)
    session_start = seconds(lap.get('LapStartTime'))
    # ``Time`` in FastF1 lap telemetry can be lap-relative; SessionTime is
    # preferred. Fall back only when it is absent and a lap start exists.
    if time_column == 'Time' and session_start is not None:
        times = times + session_start
    speed = telemetry['Speed'].to_numpy(dtype=float) if 'Speed' in telemetry else np.zeros(len(distances))
    return {
        'sessionTimeS': float(np.interp(target_distance_m, distances, times)),
        'speedKph': float(np.interp(target_distance_m, distances, speed)),
    }


def crossing_at_cumulative_distance(car_data: Any, target_distance_m: float) -> dict[str, float] | None:
    """Interpolate public FastF1 car data at one cumulative distance."""
    if car_data is None or car_data.empty:
        return None
    data = car_data.dropna(subset=['Distance', 'SessionTime']).sort_values('Distance')
    distances = data['Distance'].to_numpy(dtype=float)
    if len(distances) < 2 or target_distance_m < distances[0] or target_distance_m > distances[-1]:
        return None
    times = np.asarray([seconds(value) for value in data['SessionTime']], dtype=object)
    if any(value is None for value in times):
        return None
    speed = data['Speed'].to_numpy(dtype=float) if 'Speed' in data else np.zeros(len(data))
    return {
        'sessionTimeS': float(np.interp(target_distance_m, distances, times.astype(float))),
        'speedKph': float(np.interp(target_distance_m, distances, speed)),
    }


def crossings_from_car_data(session: Any, zone: dict[str, Any], track_length_m: float) -> dict[int, list[dict[str, Any]]]:
    """Create zone crossings from each driver's cached session car data.

    Building full merged telemetry for every individual lap is prohibitively
    expensive. FastF1's session-level car data supplies the same timing and
    speed stream once per driver. Its integrated Distance is anchored at the
    lap start and scaled to the reference-lap distance before interpolation.
    The result is therefore useful zone replay evidence, but deliberately
    marked medium confidence rather than pretending it is surveyed FIA
    position telemetry.
    """
    crossings: dict[int, list[dict[str, Any]]] = {}
    detection_fraction = float(zone['detectionLineDistanceM']) / track_length_m
    for driver, laps in session.laps.groupby('Driver', sort=False):
        if not isinstance(driver, str) or not driver:
            continue
        try:
            driver_result = session.get_driver(driver)
            driver_number = str(driver_result['DriverNumber'])
            car_data = session.car_data[driver_number].add_distance()
        except Exception:
            continue
        car_data = car_data.dropna(subset=['Distance', 'SessionTime']).sort_values('Distance')
        time_values = np.asarray([seconds(value) for value in car_data['SessionTime']], dtype=object)
        distance_values = car_data['Distance'].to_numpy(dtype=float)
        speed_values = car_data['Speed'].to_numpy(dtype=float) if 'Speed' in car_data else np.zeros(len(car_data))
        if len(distance_values) < 2 or any(value is None for value in time_values):
            continue
        session_times = time_values.astype(float)
        for _, lap in laps.iterrows():
            lap_number = lap.get('LapNumber')
            lap_start = seconds(lap.get('LapStartTime'))
            lap_time = seconds(lap.get('LapTime'))
            if lap_number is None or pd.isna(lap_number) or lap_start is None or lap_time is None:
                continue
            lap_end = lap_start + lap_time
            if lap_start < session_times[0] or lap_end > session_times[-1]:
                continue
            start_distance = float(np.interp(lap_start, session_times, distance_values))
            end_distance = float(np.interp(lap_end, session_times, distance_values))
            lap_distance = end_distance - start_distance
            if lap_distance <= 0.6 * track_length_m or lap_distance >= 1.4 * track_length_m:
                continue
            target = start_distance + (detection_fraction * lap_distance)
            if target < distance_values[0] or target > distance_values[-1]:
                continue
            crossing = {
                'sessionTimeS': float(np.interp(target, distance_values, session_times)),
                'speedKph': float(np.interp(target, distance_values, speed_values)),
            }
            crossings.setdefault(int(lap_number), []).append({
                'driver': driver,
                'lap': int(lap_number),
                'pitContext': bool(pd.notna(lap.get('PitInTime')) or pd.notna(lap.get('PitOutTime'))),
                **crossing,
            })
    return crossings


def reference_track_length(reference_lap: Any) -> tuple[float, str]:
    """Prefer merged telemetry; fall back to the lap's car-data integration.

    Some cached sessions lack position samples required by FastF1's merged
    telemetry. Car speed data can still support a reproducible replay, so the
    fallback keeps that event analyzable while declaring the weaker source.
    """
    try:
        telemetry = reference_lap.get_telemetry()
        distances = telemetry['Distance'].dropna()
        if not distances.empty:
            return float(distances.max()), 'MERGED_TELEMETRY_DISTANCE'
    except Exception:
        pass
    car_data = reference_lap.get_car_data().add_distance()
    distances = car_data['Distance'].dropna()
    if distances.empty:
        raise ValueError('no usable telemetry or car-data distance for zone alignment')
    return float(distances.max() - distances.min()), 'CAR_DATA_DISTANCE_FALLBACK'


def build_zone_opportunities(year: int, round_number: int, session_name: str = 'R') -> dict[str, Any]:
    context = rule_context_for(year, round_number, session_name)
    # Avoid fetching a session simply to confirm that an official appendix has
    # not been imported. This also makes the missing-evidence state fast and
    # usable when FastF1's remote schedule providers are unavailable.
    zone_evidence = context.get('zoneEvidence') if isinstance(context.get('zoneEvidence'), dict) else {}
    if context.get('eventSpecificDataLoaded') is not True and zone_evidence.get('zoneEvidenceLoaded') is not True:
        return {
            'schemaVersion': SCHEMA_VERSION,
            'year': year,
            'round': round_number,
            'session': session_name,
            'ruleContext': context,
            'availability': {
                'status': 'EVENT_APPENDIX_REQUIRED',
                'telemetryAlignmentAvailable': False,
                'reason': 'A complete cited FIA Power Unit Information record is required before zone alignment is computed.',
            },
            'opportunities': [],
        }
    # Replays must remain reproducible and fast when the machine has a local
    # FastF1 cache but no network route. This extractor intentionally consumes
    # the existing cache rather than making an incidental network refresh.
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.Cache.offline_mode(True)
    event = fastf1.get_event(year, round_number)
    session = event.get_session(session_name)
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    # A clean completed lap gives the event's telemetry coordinate length.
    candidate = session.laps[session.laps['LapTime'].notna()].sort_values('LapTime')
    if candidate.empty:
        raise ValueError('no completed laps available for telemetry alignment')
    reference = candidate.iloc[0]
    track_length, track_length_source = reference_track_length(reference)
    availability = zone_alignment_status(context, track_length)
    if not availability['telemetryAlignmentAvailable']:
        return {
            'schemaVersion': SCHEMA_VERSION,
            'year': year,
            'round': round_number,
            'session': session_name,
            'eventName': str(event.get('EventName', '')),
            'trackLengthM': round(track_length, 1),
            'trackLengthSource': track_length_source,
            'ruleContext': context,
            'availability': availability,
            'opportunities': [],
        }

    zone = availability['zone']
    crossings = crossings_from_car_data(session, zone, track_length)

    opportunities = []
    candidate_pair_count = 0
    for lap_number, cars in sorted(crossings.items()):
        # Earlier crossing time means further ahead at this exact track point.
        ordered = sorted(cars, key=lambda item: item['sessionTimeS'])
        for index in range(1, len(ordered)):
            defender, attacker = ordered[index - 1], ordered[index]
            gap = attacker['sessionTimeS'] - defender['sessionTimeS']
            candidate_pair_count += 1
            if not (0 < gap <= zone['detectionGapS']):
                continue
            opportunities.append({
                'year': year,
                'round': round_number,
                'session': session_name,
                'lap': lap_number,
                'driver': attacker['driver'],
                'defender': defender['driver'],
                'zoneId': zone['zoneId'],
                'zoneType': zone['type'],
                'detectionLineDistanceM': zone['detectionLineDistanceM'],
                'activationLineDistanceM': zone['activationLineDistanceM'],
                'officialDetectionGapS': zone['detectionGapS'],
                'gapAtDetectionS': round(gap, 3),
                'attackerDetectionTimeS': round(attacker['sessionTimeS'], 3),
                'defenderDetectionTimeS': round(defender['sessionTimeS'], 3),
                'attackerDetectionSpeedKph': round(attacker['speedKph'], 1),
                'defenderDetectionSpeedKph': round(defender['speedKph'], 1),
                'attackerPitContext': attacker['pitContext'],
                'defenderPitContext': defender['pitContext'],
                'alignmentMethod': 'SAME_TRACK_DISTANCE_INTERPOLATED_CAR_DATA',
                'alignmentConfidence': 'MEDIUM',
                'source': 'FASTF1_CAR_DATA + CITED_FIA_EVENT_APPENDIX',
                'eventAppendixSource': zone['eventAppendixSource'],
            })
    return {
        'schemaVersion': SCHEMA_VERSION,
        'year': year,
        'round': round_number,
        'session': session_name,
        'eventName': str(event.get('EventName', '')),
        'trackLengthM': round(track_length, 1),
        'trackLengthSource': track_length_source,
        'ruleContext': context,
        'availability': availability,
        'crossingLapCount': len(crossings),
        'crossingCount': sum(len(cars) for cars in crossings.values()),
        'candidatePairCount': candidate_pair_count,
        'opportunityCount': len(opportunities),
        'opportunities': opportunities,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--session', default='R')
    parser.add_argument('--output', action='store_true', help='write the generated cache entry')
    args = parser.parse_args()
    fastf1.set_log_level('ERROR')
    payload = build_zone_opportunities(args.year, args.round, args.session)
    if args.output:
        destination = OUT_ROOT / str(args.year) / f'{args.round}_{args.session.lower()}.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
