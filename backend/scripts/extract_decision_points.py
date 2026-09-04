"""Decision-point extraction for the PitWolf overtake model.

For one race session, finds every moment a driver ran inside the detection gap
(~1.2 s, approximating the 2026 Override Detection Gap / DRS window) of the
car ahead at a lap crossing, then labels the outcome-optimal decision:

- ATTACK: the pass happened on the next lap and held for `--hold-laps`
  (user-confirmed durability cutoff ~5-6 laps; a pass that was given back
  inside the window is NOT outcome-optimal and lands in SAVE);
- DELAY:  no immediate pass but a durable pass within the next 5 laps;
- SAVE:   no durable pass at all.

Features per row: gap, closing rate, straight-line speed delta (max speed of
the lap), tyre-age differential, compounds, lap fraction, pit distortion flag.

Output is cached as JSON under data/f1-cache/decision-points/.

The feature cutoff is LAP_START. Pace, speed, tyre-life, and lap-time features
must therefore come from the previous completed lap; future laps are used only
for retrospective labels and evaluation fields.
"""

import argparse
import json
import os
import pathlib

import fastf1
import numpy as np
import pandas as pd

from energy_surrogate import add_surrogate_energy
from fetch_f1_session import CACHE_DIR

OUT_ROOT = pathlib.Path(CACHE_DIR).parent / 'decision-points'
SCHEMA_VERSION = 'decision-point.v5'

COMPOUND_ORDINAL = {
    'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4,
    'TEST_UNKNOWN': 1,
}


def num(value):
    try:
        f = float(value)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def build_timelines(session):
    """Per driver/lap timeline with the causal race-state fields we have."""
    laps = session.laps
    timelines = {}
    for driver, rows in laps.groupby('Driver'):
        tl = {}
        for _, lap in rows.iterrows():
            lap_number = num(lap.get('LapNumber'))
            pos = num(lap.get('Position'))
            start = lap.get('LapStartTime')
            if lap_number is None:
                continue
            tl[int(lap_number)] = {
                'lapNumber': int(lap_number),
                'pos': int(pos) if pos is not None else None,
                'startS': start.total_seconds() if start is not None and not (isinstance(start, float) and np.isnan(start)) else None,
                'lapTimeS': lap.get('LapTime').total_seconds() if lap.get('LapTime') is not None and hasattr(lap.get('LapTime'), 'total_seconds') else None,
                'compound': str(c) if (c := lap.get('Compound')) is not None and not (isinstance(c, float) and np.isnan(c)) else None,
                'tyreLife': num(lap.get('TyreLife')),
                'stint': int(num(lap.get('Stint'))) if num(lap.get('Stint')) is not None else None,
                'pit': bool(lap.get('PitInLane', False)) or bool(lap.get('PitOutLane', False)),
                'pitIn': bool(lap.get('PitInLane', False)),
                'pitOut': bool(lap.get('PitOutLane', False)),
                'trackStatus': str(lap.get('TrackStatus')) if lap.get('TrackStatus') is not None else None,
                'isAccurate': bool(lap.get('IsAccurate')) if lap.get('IsAccurate') is not None else None,
                'deleted': bool(lap.get('Deleted')) if lap.get('Deleted') is not None else False,
            }
        timelines[driver] = tl
    return timelines


def build_max_speeds(session):
    """(driver, lap) -> max speed kph, from the session-wide car_data block.

    One sorted pass per car; per-lap get_telemetry() merges are ~1000x slower.
    """
    car_data = session.car_data
    speeds = {}
    if car_data is None or len(car_data) == 0:
        return speeds
    for _, res in session.results.iterrows():
        abbr = res.get('Abbreviation')
        car_number = res.get('DriverNumber')
        cd = car_data.get(car_number)
        if abbr is None or cd is None or cd.empty:
            continue
        t = cd['Time'].dt.total_seconds().to_numpy()
        v = cd['Speed'].to_numpy(dtype=float)
        order = np.argsort(t)
        t, v = t[order], v[order]
        for _, lap in session.laps[session.laps['Driver'] == abbr].iterrows():
            lap_number = num(lap.get('LapNumber'))
            start = lap.get('LapStartTime')
            lap_time = lap.get('LapTime')
            if lap_number is None or start is None or lap_time is None:
                continue
            t0 = start.total_seconds()
            t1 = t0 + lap_time.total_seconds()
            if not (np.isfinite(t0) and np.isfinite(t1)):
                continue
            mask = (t >= t0) & (t < t1)
            if mask.any():
                speeds[(abbr, int(lap_number))] = float(v[mask].max())
    return speeds


def gap_at(timelines, driver, defender, lap):
    d = timelines.get(driver, {}).get(lap)
    a = timelines.get(defender, {}).get(lap)
    if not d or not a or d['startS'] is None or a['startS'] is None:
        return None
    return d['startS'] - a['startS']


def driver_row_at_position_time(timelines, position, timestamp_s, tolerance_s=5.0):
    """Find the adjacent car on the shared session clock.

    LapNumber belongs to each car. A backmarker being lapped can therefore
    have a different lap number from the nearby attacker; aligning by
    LapStartTime lets the extractor reject that case explicitly.
    """
    if timestamp_s is None:
        return None, None
    best = None
    for driver, timeline in timelines.items():
        for row in timeline.values():
            start_s = row.get('startS')
            if row.get('pos') != position or start_s is None:
                continue
            distance = abs(float(start_s) - float(timestamp_s))
            if distance <= tolerance_s and (best is None or distance < best[0]):
                best = (distance, driver, row)
    return (best[1], best[2]) if best else (None, None)


def label_outcome(timelines, driver, defender, lap, hold_laps, max_lap):
    """Returns (label, passed_now, held, exclusion_reason)."""
    def pos_of(d, k):
        row = timelines.get(d, {}).get(k)
        return row['pos'] if row else None

    d_next = pos_of(driver, lap + 1)
    a_next = pos_of(defender, lap + 1)
    if d_next is None:
        return None, False, False, 'MISSING_FUTURE_POSITION'
    passed_now = a_next is not None and d_next < a_next

    if passed_now:
        held = True
        for k in range(lap + 1, min(lap + hold_laps, max_lap) + 1):
            dk, ak = pos_of(driver, k), pos_of(defender, k)
            if dk is None or ak is None:
                return None, True, False, 'MISSING_FUTURE_POSITION'
            if not dk < ak:
                held = False
                break
        return ('ATTACK' if held else 'SAVE'), True, held, None

    for k in range(lap + 2, min(lap + hold_laps, max_lap) + 1):
        dk, ak = pos_of(driver, k), pos_of(defender, k)
        if dk is None or ak is None:
            return None, False, False, 'MISSING_FUTURE_POSITION'
        if dk < ak:
            return 'DELAY', False, False, None
    return 'SAVE', False, False, None


def observed_persistence(timelines, driver, defender, lap, max_lap):
    """Return the first observed pass lap and consecutive laps held after it."""
    pass_lap = None
    for current_lap in range(lap + 1, max_lap + 1):
        driver_position = (timelines.get(driver, {}).get(current_lap) or {}).get('pos')
        defender_position = (timelines.get(defender, {}).get(current_lap) or {}).get('pos')
        if driver_position is None or defender_position is None:
            continue
        if driver_position < defender_position:
            pass_lap = current_lap
            break
    if pass_lap is None:
        return None, 0

    held_laps = 0
    for current_lap in range(pass_lap, max_lap + 1):
        driver_position = (timelines.get(driver, {}).get(current_lap) or {}).get('pos')
        defender_position = (timelines.get(defender, {}).get(current_lap) or {}).get('pos')
        if driver_position is None or defender_position is None or driver_position >= defender_position:
            break
        held_laps += 1
    return pass_lap, held_laps


def nearest_weather(weather_data, session_seconds):
    """Return a small, JSON-safe weather snapshot at a decision timestamp."""
    if weather_data is None or weather_data.empty or session_seconds is None:
        return None
    try:
        times = weather_data['Time'].dt.total_seconds().to_numpy()
        row = weather_data.iloc[int(np.argmin(np.abs(times - session_seconds)))]
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    result = {}
    for source, target in (
        ('AirTemp', 'airTempC'), ('TrackTemp', 'trackTempC'),
        ('Humidity', 'humidityPct'), ('Pressure', 'pressureMbar'),
        ('Rainfall', 'rainfall'), ('WindSpeed', 'windSpeedMps'),
    ):
        value = num(row.get(source))
        if value is not None:
            result[target] = round(value, 3)
    return result or None


def missing_fields(record, fields):
    return [field for field in fields if record.get(field) is None]


def bounded(value, low=0.0, high=1.0):
    return round(float(np.clip(value, low, high)), 3)


def traffic_context(timelines, driver, lap, position):
    """Position-based traffic context; exact distance gaps need GPS alignment."""
    positions = []
    for other, timeline in timelines.items():
        if other == driver:
            continue
        other_position = (timeline.get(lap) or {}).get('pos')
        if other_position is not None:
            positions.append((other, other_position))
    ahead = sorted((item for item in positions if item[1] < position), key=lambda item: item[1], reverse=True)
    behind = sorted((item for item in positions if item[1] > position), key=lambda item: item[1])
    ahead_near = ahead[:3]
    behind_near = behind[:3]
    return {
        'trafficAheadCount': len(ahead_near),
        'trafficBehindCount': len(behind_near),
        'packDensity': bounded((len(ahead_near) + len(behind_near)) / 6.0),
    }


def tyre_degradation_proxy(timeline, lap):
    """Approximate degradation using only completed laps before ``lap``."""
    completed = timeline.get(lap - 1) or {}
    prior = timeline.get(lap - 2) or {}
    age = max(0.0, num(completed.get('tyreLife')) or 0.0)
    completed_time = num(completed.get('lapTimeS'))
    prior_time = num(prior.get('lapTimeS'))
    drift = max(0.0, (completed_time - prior_time)
                if completed_time is not None and prior_time is not None else 0.0)
    return bounded((age / 45.0) + (drift / 5.0))


def extract_race(year, round_number, session_name='R', max_gap=1.2, hold_laps=6):
    event = fastf1.get_event(year, round_number)
    session = event.get_session(session_name)
    try:
        session.load(laps=True, telemetry=True, weather=True, messages=True)
    except Exception as error:
        # Weather and race-control messages are useful optional context, but
        # they should not prevent a cached timing/telemetry dataset from being
        # extracted when FastF1's shared request budget is exhausted.
        if '500 calls/h' not in str(error) and 'rate limit' not in str(error).lower():
            raise
        session.load(laps=True, telemetry=True, weather=False, messages=False)

    timelines = build_timelines(session)
    finish_positions = {}
    grid_positions = {}
    result_status = {}
    for _, result in session.results.iterrows():
        abbreviation = result.get('Abbreviation')
        position = num(result.get('Position'))
        if abbreviation is not None and position is not None:
            finish_positions[str(abbreviation)] = int(position)
        if abbreviation is not None:
            abbreviation = str(abbreviation)
            grid = num(result.get('GridPosition'))
            if grid is not None:
                grid_positions[abbreviation] = int(grid)
            if result.get('Status') is not None:
                result_status[abbreviation] = str(result.get('Status'))

    total_laps = session.total_laps or max(
        (max(tl) for tl in timelines.values() if tl), default=0)
    speeds = build_max_speeds(session)

    speed_values = list(speeds.values())
    race_mean_speed = float(np.mean(speed_values)) if speed_values else None
    # Lap 1 is complete by the time a lap-2 decision is evaluated, so it is a
    # valid causal baseline for the first emitted decision row.
    past_speed_values = [
        speed for (speed_driver, speed_lap), speed in speeds.items()
        if speed_lap == 1 and speed is not None
    ]

    rows = []
    excluded_rows = []
    for lap in range(2, int(total_laps) + 1):
        # Decision features are available at the start of this lap.  The
        # current lap's speed must therefore not be used to form its own
        # race-speed baseline; only completed earlier laps are admissible.
        causal_mean_speed = float(np.mean(past_speed_values)) if past_speed_values else None
        for driver, tl in timelines.items():
            row = tl.get(lap)
            if not row or row['pos'] is None or row['pos'] < 2:
                continue
            defender, d_row = driver_row_at_position_time(
                timelines, row['pos'] - 1, row.get('startS'))
            if defender is None or defender == driver or d_row is None:
                continue
            gap = row['startS'] - d_row['startS'] if row.get('startS') is not None and d_row.get('startS') is not None else None
            if gap is None or not (0.0 < gap <= max_gap):
                continue

            attacker_lap_number = int(row['lapNumber'])
            defender_lap_number = int(d_row['lapNumber'])
            lap_difference = attacker_lap_number - defender_lap_number
            if lap_difference > 0:
                excluded_rows.append({
                    'year': year,
                    'round': round_number,
                    'session': session_name,
                    'lap': lap,
                    'driver': driver,
                    'defender': defender,
                    'attackerLapNumber': attacker_lap_number,
                    'defenderLapNumber': defender_lap_number,
                    'lapDifference': lap_difference,
                    'eventType': 'LAPPING',
                    'gapS': round(gap, 3),
                    'exclusionReasons': ['LAPPING_BACKMARKER'],
                })
                continue

            label, passed_now, held, label_reason = label_outcome(
                timelines, driver, defender, lap, hold_laps, int(total_laps))
            if label is None:
                excluded_rows.append({
                    'year': year,
                    'round': round_number,
                    'session': session_name,
                    'lap': lap,
                    'driver': driver,
                    'defender': defender,
                    'gapS': round(gap, 3),
                    'exclusionReasons': [label_reason or 'AMBIGUOUS_OUTCOME'],
                })
                continue

            prev_gap = gap_at(timelines, driver, defender, lap - 1)
            pit_lap = row['pit'] or d_row.get('pit', False)
            pit_next = (timelines.get(driver, {}).get(lap + 1) or {}).get('pit', False) or \
                       (timelines.get(defender, {}).get(lap + 1) or {}).get('pit', False)

            # The decision is made at lap start. The current lap's maximum
            # speed is not available yet, so use the previous completed lap.
            speed_feature_lap = lap - 1
            d_speed = speeds.get((driver, speed_feature_lap))
            a_speed = speeds.get((defender, speed_feature_lap))
            observed_pass_lap, observed_lead_laps = observed_persistence(
                timelines, driver, defender, lap, int(total_laps))
            driver_status = timelines.get(driver, {}).get(lap) or {}
            defender_status = timelines.get(defender, {}).get(lap) or {}
            traffic = traffic_context(timelines, driver, lap, row['pos'])
            closing_rate = round(prev_gap - gap, 3) if prev_gap is not None else None
            speed_delta = round(d_speed - a_speed, 1) if d_speed is not None and a_speed is not None else None
            window = bounded((max_gap - gap) / max_gap)
            slipstream_proxy = bounded(window * (0.5 + max(0.0, (speed_delta or 0.0)) / 20.0))
            dirty_air_risk = bounded(window * (1.0 + (traffic['trafficAheadCount'] / 3.0)))
            exclusion_reasons = []
            if pit_lap or pit_next:
                exclusion_reasons.append('PIT_CYCLE')
            if driver_status.get('trackStatus') not in (None, '1') or defender_status.get('trackStatus') not in (None, '1'):
                exclusion_reasons.append('NON_GREEN_TRACK_STATUS')
            if driver_status.get('deleted') or defender_status.get('deleted'):
                exclusion_reasons.append('DELETED_LAP')
            if driver_status.get('isAccurate') is False or defender_status.get('isAccurate') is False:
                exclusion_reasons.append('INACCURATE_LAP')
            if d_speed is None or a_speed is None:
                exclusion_reasons.append('MISSING_SPEED_TELEMETRY')
            if (timelines.get(driver, {}).get(lap + 1) or {}).get('pos') is None or \
               (timelines.get(defender, {}).get(lap + 1) or {}).get('pos') is None:
                exclusion_reasons.append('MISSING_FUTURE_POSITION')

            rows.append({
                'year': year,
                'round': round_number,
                'session': session_name,
                'lap': lap,
                'driver': driver,
                'defender': defender,
                'position': row['pos'],
                'defenderPosition': d_row.get('pos'),
                'attackerLapNumber': attacker_lap_number,
                'defenderLapNumber': defender_lap_number,
                'lapDifference': lap_difference,
                'isLapping': False,
                'gridPosition': grid_positions.get(driver),
                'defenderGridPosition': grid_positions.get(defender),
                'decisionTimestampS': row['startS'],
                'decisionDistanceM': None,
                'zoneId': None,
                'alignmentMethod': 'LAP_START_TIMESTAMP',
                'alignmentToleranceS': 0.0,
                'gapS': round(gap, 3),
                'closingRateS': closing_rate,
                'speedDeltaKph': speed_delta,
                'speedFeatureLap': speed_feature_lap,
                # These are previous-completed-lap values. The current lap's
                # final time and tyre-life are not known at this cutoff.
                'attackerLapTimeS': (timelines.get(driver, {}).get(lap - 1) or {}).get('lapTimeS'),
                'defenderLapTimeS': (timelines.get(defender, {}).get(lap - 1) or {}).get('lapTimeS'),
                'tyreAgeDiff': round(
                    ((timelines.get(driver, {}).get(lap - 1) or {}).get('tyreLife') or 0)
                    - ((timelines.get(defender, {}).get(lap - 1) or {}).get('tyreLife') or 0), 1
                ) if (timelines.get(driver, {}).get(lap - 1) or {}).get('tyreLife') is not None
                and (timelines.get(defender, {}).get(lap - 1) or {}).get('tyreLife') is not None else None,
                'attackerCompound': (timelines.get(driver, {}).get(lap - 1) or {}).get('compound') or row['compound'],
                'defenderCompound': (timelines.get(defender, {}).get(lap - 1) or {}).get('compound') or d_row.get('compound'),
                'attackerStint': (timelines.get(driver, {}).get(lap - 1) or {}).get('stint') or row.get('stint'),
                'defenderStint': (timelines.get(defender, {}).get(lap - 1) or {}).get('stint') or d_row.get('stint'),
                'attackerPitIn': row.get('pitIn', False),
                'attackerPitOut': row.get('pitOut', False),
                'defenderPitIn': d_row.get('pitIn', False),
                'defenderPitOut': d_row.get('pitOut', False),
                'trafficAheadCount': traffic['trafficAheadCount'],
                'trafficBehindCount': traffic['trafficBehindCount'],
                'packDensity': traffic['packDensity'],
                'slipstreamProxy': slipstream_proxy,
                'dirtyAirRisk': dirty_air_risk,
                'attackerTyreDegProxy': tyre_degradation_proxy(timelines.get(driver, {}), lap),
                'defenderTyreDegProxy': tyre_degradation_proxy(timelines.get(defender, {}), lap),
                'attackerTrackStatus': row.get('trackStatus'),
                'defenderTrackStatus': d_row.get('trackStatus'),
                'attackerStatus': result_status.get(driver),
                'defenderStatus': result_status.get(defender),
                'lapFraction': round(lap / float(total_laps), 3),
                'raceMeanSpeedKph': round(causal_mean_speed, 1) if causal_mean_speed else None,
                'featureCutoff': 'LAP_START',
                'labelSource': 'OUTCOME_DERIVED',
                'pitDistorted': pit_lap or pit_next,
                'defenderActive': (timelines.get(defender, {}).get(lap + 1) or {}).get('pos') is not None,
                'observedPassLap': observed_pass_lap,
                'observedLeadLaps': observed_lead_laps,
                'weather': nearest_weather(session.weather_data, row['startS']),
                'exclusionReasons': exclusion_reasons,
                'eligibleForTraining': not exclusion_reasons,
                'missingFields': missing_fields({
                    'decisionTimestampS': row['startS'],
                    'gapS': gap,
                    'closingRateS': closing_rate,
                    'speedDeltaKph': speed_delta,
                    'attackerLapTimeS': (timelines.get(driver, {}).get(lap - 1) or {}).get('lapTimeS'),
                    'defenderLapTimeS': (timelines.get(defender, {}).get(lap - 1) or {}).get('lapTimeS'),
                }, ('decisionTimestampS', 'closingRateS', 'speedDeltaKph', 'attackerLapTimeS', 'defenderLapTimeS')),
                'passedNow': passed_now,
                'held': held,
                'label': label,
            })
        past_speed_values.extend(
            speed for (speed_driver, speed_lap), speed in speeds.items()
            if speed_lap == lap and speed is not None
        )

    if rows:
        row_frame = add_surrogate_energy(pd.DataFrame(rows))
        # Cast to object first: pandas otherwise keeps NaN in float columns,
        # which would leak non-standard NaN values into the JSON cache.
        rows = row_frame.astype(object).where(pd.notna(row_frame), None).to_dict(orient='records')

    counts = {}
    for r in rows:
        counts[r['label']] = counts.get(r['label'], 0) + 1

    return {
        'year': year,
        'round': round_number,
        'session': session_name,
        'eventName': str(event.get('EventName', '')),
        'totalLaps': int(total_laps),
        'maxGapThresholdS': max_gap,
        'holdLaps': hold_laps,
        'finishPositions': finish_positions,
        'raceMeanSpeedKphFinalObserved': round(race_mean_speed, 1) if race_mean_speed else None,
        'labelCounts': counts,
        'excludedRows': excluded_rows,
        'excludedCount': len(excluded_rows),
        'lappingExcludedCount': sum(
            1 for row in excluded_rows
            if 'LAPPING_BACKMARKER' in row.get('exclusionReasons', [])
        ),
        'schemaVersion': SCHEMA_VERSION,
        'featureCutoff': 'LAP_START',
        'provenance': {
            'real': ['position', 'gridPosition', 'lapNumber', 'lapTimeS', 'compound', 'tyreLife', 'pit', 'trackStatus', 'weather'],
            'derived': ['gapS', 'closingRateS', 'speedDeltaKph', 'speedFeatureLap', 'lapFraction', 'alignmentMethod',
                        'trafficAheadCount', 'trafficBehindCount', 'packDensity',
                        'slipstreamProxy', 'dirtyAirRisk', 'attackerTyreDegProxy',
                        'defenderTyreDegProxy'],
            'modelled': ['attackerSoCMj', 'defenderSoCMj', 'energyDeltaMj'],
        },
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--session', default='R')
    parser.add_argument('--max-gap', type=float, default=1.2)
    parser.add_argument('--hold-laps', type=int, default=6)
    args = parser.parse_args()

    out = OUT_ROOT / str(args.year) / f'{args.round}_{args.session.lower()}.json'
    if out.exists() and not os.environ.get('FORCE_REBUILD'):
        try:
            cached = json.loads(out.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, dict) and cached.get('schemaVersion') == SCHEMA_VERSION:
            print(json.dumps(cached, allow_nan=False))
            return

    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    payload = extract_race(args.year, args.round, args.session, args.max_gap, args.hold_laps)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, allow_nan=False)
    out.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    import sys
    try:
        main()
    except Exception as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        raise SystemExit(1)
