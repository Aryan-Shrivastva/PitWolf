"""Create a compact, recorded multi-car replay window for the visual simulator.

This is deliberately a *recorded* position stream.  It contains no modelled
pass, energy, or opponent-response outcome; those belong to later simulation
phases.  A selected driver's completed lap provides the global session-time
window, then every car with public FastF1 position data is sampled on the same
clock so the frontend never needs to invent ghost-car positions.
"""

import argparse
import json
import math
from bisect import bisect_right

import fastf1
import numpy as np
import pandas as pd
from fastf1 import _api

from fetch_f1_session import CACHE_DIR, clean, td_s


SAMPLES_PER_SECOND = 2.5
MAX_FRAMES = 360
FULL_RACE_SAMPLES_PER_SECOND = 1.0
MAX_FULL_RACE_FRAMES = 7200


def seconds(value):
    value = clean(value)
    if value is None:
        return None
    try:
        return float(value.total_seconds())
    except AttributeError:
        return None


def as_color(value):
    value = clean(value)
    return f'#{value}' if value else None


def lap_window(session, driver, lap_number):
    laps = session.laps[
        (session.laps['Driver'] == driver)
        & (session.laps['LapNumber'] == lap_number)
    ]
    if laps.empty:
        raise ValueError(f'no completed lap {lap_number} for {driver}')
    lap = laps.iloc[0]
    end_s = seconds(lap.get('Time'))
    duration_s = td_s(lap.get('LapTime'))
    start_s = seconds(lap.get('LapStartTime'))
    if start_s is None and end_s is not None and duration_s is not None:
        start_s = end_s - duration_s
    if start_s is None or end_s is None or end_s <= start_s:
        raise ValueError(f'no usable session-time window for {driver} lap {lap_number}')
    return lap, start_s, end_s


def driver_metadata(session):
    metadata = {}
    for _, result in session.results.iterrows():
        number = clean(str(result.get('DriverNumber')))
        abbreviation = clean(result.get('Abbreviation'))
        if not number or not abbreviation:
            continue
        metadata[number] = {
            'driver': abbreviation,
            'name': clean(result.get('BroadcastName')) or clean(result.get('FullName')) or abbreviation,
            'team': clean(result.get('TeamName')),
            'color': as_color(result.get('TeamColor')),
            'classifiedPosition': clean(str(result.get('ClassifiedPosition'))),
            # This is used only before a driver has a timing position for an
            # active lap. It is deliberately distinct from the final race
            # classification above.
            'gridPosition': int(grid) if (grid := clean(result.get('GridPosition'))) not in (None, 0) else None,
        }
    return metadata


def prepared_position_streams(session, metadata):
    streams = []
    for number, data in (session.pos_data or {}).items():
        info = metadata.get(str(number))
        if not info or data is None or data.empty:
            continue
        valid = data.dropna(subset=['SessionTime', 'X', 'Y']).copy()
        if valid.empty:
            continue
        times = valid['SessionTime'].dt.total_seconds().to_numpy(dtype=float)
        xs = valid['X'].to_numpy(dtype=float)
        ys = valid['Y'].to_numpy(dtype=float)
        valid_values = np.isfinite(times) & np.isfinite(xs) & np.isfinite(ys)
        if valid_values.sum() < 2:
            continue
        streams.append((info, times[valid_values], xs[valid_values], ys[valid_values]))
    return streams


def parse_timing_gap(value):
    """Return a recorded positive gap-to-leader in seconds when available."""
    value = clean(value)
    if value is None:
        return None
    if hasattr(value, 'total_seconds'):
        return round(float(value.total_seconds()), 3)
    text = str(value).strip()
    if text.startswith('+'):
        try:
            return round(float(text[1:]), 3)
        except ValueError:
            return None
    return 0.0 if text in {'0', '+0', '+0.0'} else None


def prepared_timing_positions(session, metadata):
    """Return timestamped official timing positions keyed by abbreviation.

    GPS ``pos_data`` is intended for the map, not for timing classification.
    It can reverse the order at start/finish when projected onto a reference
    lap. The cached TimingData feed, however, reports the race-control
    position and gap whenever timing updates. It lets the replay show P1/P2
    at the actual recorded instant rather than inferring it from map geometry.
    """
    number_to_driver = {number: info['driver'] for number, info in metadata.items()}
    timing = {driver: [] for driver in number_to_driver.values()}

    try:
        _, stream, _ = _api._extended_timing_data(session.api_path)
    except Exception:
        stream = None

    if stream is not None and not stream.empty:
        for _, update in stream.iterrows():
            number = clean(str(update.get('Driver')))
            driver = number_to_driver.get(number)
            position = clean(update.get('Position'))
            time_s = seconds(update.get('Time'))
            if not driver or position is None or time_s is None:
                continue
            try:
                position = int(position)
            except (TypeError, ValueError):
                continue
            if position < 1:
                continue
            timing[driver].append({
                'timeS': time_s,
                'position': position,
                'gapS': parse_timing_gap(update.get('GapToLeader')),
                'lapNumber': None,
                'source': 'OFFICIAL_TIMING_STREAM',
            })

    # Race-control timing is not available for every historical source. In
    # that case provide the recorded lap classification as a clearly bounded
    # fallback, never a GPS/map-derived ranking.
    if any(timing.values()):
        for entries in timing.values():
            entries.sort(key=lambda item: item['timeS'])
        return timing

    for _, lap in session.laps.iterrows():
        number = clean(str(lap.get('DriverNumber')))
        driver = number_to_driver.get(number)
        position = clean(lap.get('Position'))
        start_s = seconds(lap.get('LapStartTime'))
        end_s = seconds(lap.get('Time'))
        if not driver or position is None or start_s is None or end_s is None:
            continue
        try:
            position = int(position)
        except (TypeError, ValueError):
            continue
        if position < 1 or end_s < start_s:
            continue
        timing[driver].append({
            'timeS': start_s,
            'lapNumber': int(lap.get('LapNumber')),
            'position': position,
            'gapS': None,
            'source': 'OFFICIAL_LAP_TIMING',
        })
    for entries in timing.values():
        entries.sort(key=lambda item: item['timeS'])
    return timing


def timing_position_at(entries, sample_time, fallback_position=None):
    """Find the most recent official timing update at ``sample_time``."""
    index = bisect_right(entries, sample_time, key=lambda item: item['timeS']) - 1
    if index >= 0:
        return entries[index]
    if fallback_position is not None:
        return {
            'position': fallback_position,
            'lapNumber': None,
            'gapS': None,
            'source': 'RECORDED_GRID_POSITION',
        }
    return None


def replay_car(info, times, xs, ys, timing_positions, sample_time):
    """Create one visual GPS sample with its latest official timing update."""
    if sample_time < times[0] or sample_time > times[-1]:
        return None
    timing = timing_position_at(
        timing_positions.get(info['driver'], []),
        sample_time,
        info.get('gridPosition'),
    )
    return {
        'driver': info['driver'],
        'x': round(float(np.interp(sample_time, times, xs)), 2),
        'y': round(float(np.interp(sample_time, times, ys)), 2),
        # Position and gap are timestamped official timing data. Coordinates
        # are only for drawing the car on the circuit.
        'timingPosition': timing['position'] if timing else None,
        'timingLap': timing['lapNumber'] if timing else None,
        'timingGapS': timing['gapS'] if timing else None,
        'timingSource': timing['source'] if timing else None,
    }


def lap_boundaries_by_driver(session, session_start_s):
    """Build jump targets from each driver's real completed-lap boundaries."""
    boundaries = {}
    for _, lap in session.laps.iterrows():
        driver = clean(lap.get('Driver'))
        number = clean(lap.get('LapNumber'))
        start_s = seconds(lap.get('LapStartTime'))
        end_s = seconds(lap.get('Time'))
        if not driver or number is None or start_s is None or end_s is None or end_s <= start_s:
            continue
        try:
            lap_number = int(number)
        except (TypeError, ValueError):
            continue
        compound = clean(lap.get('Compound'))
        track_status = clean(lap.get('TrackStatus'))
        boundaries.setdefault(driver, []).append({
            'lap': lap_number,
            'startT': round(start_s - session_start_s, 3),
            'endT': round(end_s - session_start_s, 3),
            'lapTimeS': round(end_s - start_s, 3),
            'compound': str(compound) if compound is not None else None,
            'trackStatus': str(track_status) if track_status is not None else None,
        })
    for entries in boundaries.values():
        entries.sort(key=lambda item: item['lap'])
    return boundaries


def build_full_race_replay(year, round_number, session_name):
    """Build a complete recorded race timeline; it contains no counterfactual.

    The UI may jump to a selected lap for a model branch, but normal playback
    runs continuously from race start through the final recorded timing point.
    A one-Hz compact source stream is interpolated in the browser so the
    full-race payload stays practical while cars still move smoothly.
    """
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.Cache.offline_mode(True)
    event = fastf1.get_event(year, round_number)
    session = event.get_session(session_name)
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    usable_laps = session.laps.dropna(subset=['LapStartTime', 'Time'])
    if usable_laps.empty:
        raise ValueError('no completed laps available for a full recorded replay')
    starts = [seconds(value) for value in usable_laps['LapStartTime']]
    ends = [seconds(value) for value in usable_laps['Time']]
    starts = [value for value in starts if value is not None]
    ends = [value for value in ends if value is not None]
    if not starts or not ends:
        raise ValueError('no usable session-time bounds for a full recorded replay')
    start_s, end_s = min(starts), max(ends)
    if end_s <= start_s:
        raise ValueError('invalid full replay session-time bounds')

    metadata = driver_metadata(session)
    streams = prepared_position_streams(session, metadata)
    timing_positions = prepared_timing_positions(session, metadata)
    if not streams:
        raise ValueError('no public position streams available for this session')

    duration_s = end_s - start_s
    frame_count = min(MAX_FULL_RACE_FRAMES, max(2, int(math.ceil(duration_s * FULL_RACE_SAMPLES_PER_SECOND)) + 1))
    sample_times = np.linspace(start_s, end_s, frame_count)
    frames = []
    for sample_time in sample_times:
        cars = [
            car for car in (
                replay_car(info, times, xs, ys, timing_positions, sample_time)
                for info, times, xs, ys in streams
            ) if car is not None
        ]
        frames.append({'t': round(float(sample_time - start_s), 3), 'cars': cars})

    lap_boundaries = lap_boundaries_by_driver(session, start_s)
    total_laps = int(session.laps['LapNumber'].max()) if not session.laps.empty else 0
    return {
        'schemaVersion': 'recorded-full-race-replay.v1',
        'provenance': 'PUBLIC_FASTF1_POSITION_AND_TIMESTAMPED_TIMING_DATA',
        'mode': 'RECORDED_FULL_RACE_REPLAY_ONLY',
        'event': {
            'year': int(year),
            'round': int(round_number),
            'name': event.EventName,
            'location': event.Location,
        },
        'session': session_name,
        'totalLaps': total_laps,
        'window': {
            'sessionStartS': round(start_s, 3),
            'durationS': round(duration_s, 3),
            'frameCount': len(frames),
            'sampleRateHz': FULL_RACE_SAMPLES_PER_SECOND,
        },
        'lapBoundariesByDriver': lap_boundaries,
        'drivers': sorted(metadata.values(), key=lambda item: (item['classifiedPosition'] or '999', item['driver'])),
        'frames': frames,
    }


def build_replay_window(year, round_number, session_name, driver, lap_number):
    # Keep this in the reusable builder as well as ``main`` so API tests and
    # future replay services cannot accidentally attempt a network schedule
    # fetch for a cache-backed historical session.
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.Cache.offline_mode(True)
    event = fastf1.get_event(year, round_number)
    session = event.get_session(session_name)
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    selected_lap, start_s, end_s = lap_window(session, driver, lap_number)
    available_laps = sorted({
        int(value) for value in session.laps.loc[
            (session.laps['Driver'] == driver) & session.laps['LapTime'].notna(),
            'LapNumber',
        ].dropna().tolist()
    })
    duration_s = end_s - start_s
    frame_count = min(MAX_FRAMES, max(2, int(math.ceil(duration_s * SAMPLES_PER_SECOND)) + 1))
    sample_times = np.linspace(start_s, end_s, frame_count)
    metadata = driver_metadata(session)
    streams = prepared_position_streams(session, metadata)
    timing_positions = prepared_timing_positions(session, metadata)
    if not streams:
        raise ValueError('no public position streams available for this session')

    frames = []
    for sample_time in sample_times:
        cars = []
        for info, times, xs, ys in streams:
            car = replay_car(info, times, xs, ys, timing_positions, sample_time)
            if car is not None:
                cars.append(car)
        frames.append({'t': round(float(sample_time - start_s), 3), 'cars': cars})

    total_laps = int(session.laps['LapNumber'].max()) if not session.laps.empty else int(lap_number)
    return {
        'schemaVersion': 'recorded-replay-window.v4',
        'provenance': 'PUBLIC_FASTF1_POSITION_AND_TIMESTAMPED_TIMING_DATA',
        'mode': 'RECORDED_REPLAY_ONLY',
        'event': {
            'year': int(year),
            'round': int(round_number),
            'name': event.EventName,
            'location': event.Location,
        },
        'session': session_name,
        'selectedDriver': driver,
        'selectedLap': int(lap_number),
        'availableLaps': available_laps,
        'totalLaps': total_laps,
        'window': {
            'sessionStartS': round(start_s, 3),
            'durationS': round(duration_s, 3),
            'frameCount': len(frames),
        },
        'selectedLapContext': {
            'lapTimeS': duration_s,
            'compound': str(compound) if (compound := clean(selected_lap.get('Compound'))) is not None else None,
            'trackStatus': str(status) if (status := clean(selected_lap.get('TrackStatus'))) is not None else None,
        },
        'drivers': sorted(metadata.values(), key=lambda item: (item['classifiedPosition'] or '999', item['driver'])),
        'frames': frames,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--driver')
    parser.add_argument('--lap', type=int)
    parser.add_argument('--full-race', action='store_true')
    args = parser.parse_args()

    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.Cache.offline_mode(True)
    if args.full_race:
        print(json.dumps(build_full_race_replay(args.year, args.round, args.session)))
    elif args.driver and args.lap:
        print(json.dumps(build_replay_window(args.year, args.round, args.session, args.driver.upper(), args.lap)))
    else:
        parser.error('--driver and --lap are required unless --full-race is used')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'error': str(error)}), file=__import__('sys').stderr)
        raise SystemExit(1)
