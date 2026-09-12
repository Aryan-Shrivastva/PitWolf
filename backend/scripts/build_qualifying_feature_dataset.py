"""Create the reproducible qualifying feature table used by pace models.

Each row is one clean, completed 2026 qualifying lap.  The inputs are public
FastF1 timing/telemetry only; ERS commands and battery SoC are deliberately
not features because public data does not contain them.
"""

import argparse
import json
from pathlib import Path

import fastf1
import numpy as np

from fetch_f1_session import CACHE_DIR, td_s


SESSION_DIR = CACHE_DIR.parent / 'sessions'
MODEL_DIR = CACHE_DIR.parent / 'models'
FEATURE_VERSION = 'qualifying-features.v1'


def clean_lap(item):
    return item.get('lapTimeS') is not None and not item.get('isOutlier', False)


def compound_code(value):
    return {'SOFT': 3, 'MEDIUM': 2, 'HARD': 1, 'INTERMEDIATE': 0, 'WET': -1}.get(str(value).upper(), -2)


def telemetry_features(lap):
    telemetry = lap.get_telemetry()
    if len(telemetry) < 20:
        return None
    time = telemetry['Time'].dt.total_seconds().to_numpy(dtype=float)
    dt = np.clip(np.diff(time, prepend=time[0] - 0.02), 0.01, 0.5)
    speed = telemetry['Speed'].to_numpy(dtype=float)
    throttle = telemetry['Throttle'].to_numpy(dtype=float)
    brake = telemetry['Brake'].to_numpy(dtype=bool)
    acceleration = np.gradient(speed / 3.6, np.maximum.accumulate(time) + np.arange(len(time)) * 1e-4)
    full_throttle = (throttle >= 95.0) & ~brake
    coast = (throttle < 15.0) & ~brake
    high_speed = speed >= np.percentile(speed, 70)
    brake_starts = np.count_nonzero(brake[1:] & ~brake[:-1])
    return {
        'telemetrySamples': int(len(telemetry)),
        'topSpeedKph': round(float(np.percentile(speed, 99)), 2),
        'meanSpeedKph': round(float(np.mean(speed)), 2),
        'p90SpeedKph': round(float(np.percentile(speed, 90)), 2),
        'meanThrottlePct': round(float(np.mean(throttle)), 2),
        'fullThrottleS': round(float(np.sum(dt[full_throttle])), 3),
        'highSpeedFullThrottleS': round(float(np.sum(dt[full_throttle & high_speed])), 3),
        'brakingS': round(float(np.sum(dt[brake])), 3),
        'coastS': round(float(np.sum(dt[coast])), 3),
        'brakeEvents': int(brake_starts),
        'peakDecelMs2': round(float(abs(min(0.0, np.percentile(acceleration, 1)))), 3),
        'peakAccelMs2': round(float(max(0.0, np.percentile(acceleration, 99))), 3),
    }


def qualifying_rows(year):
    rows = []
    coverage = []
    for source in sorted((SESSION_DIR / str(year)).glob('*_qualifying.json'), key=lambda path: int(path.name.split('_')[0])):
        payload = json.loads(source.read_text(encoding='utf8'))
        clean = [item for item in payload.get('laps', []) if clean_lap(item)]
        ranked = sorted(payload.get('drivers', []), key=lambda driver: driver.get('position') or 99)
        if not clean or not ranked:
            continue
        pole_driver = ranked[0].get('abbr')
        pole_candidates = [item['lapTimeS'] for item in clean if item.get('driver') == pole_driver]
        if not pole_candidates:
            continue
        pole_time = min(pole_candidates)
        round_number = int(payload['event']['round'])
        session = fastf1.get_event(year, round_number).get_session('Qualifying')
        session.load(laps=True, telemetry=True, weather=True, messages=False)
        lookup = {(str(item.get('Driver')), int(item.get('LapNumber'))): item for _, item in session.laps.iterrows() if item.get('Driver') is not None}
        team_for_driver = {str(item.get('Abbreviation')): str(item.get('TeamName') or 'UNKNOWN') for _, item in session.results.iterrows()}
        weather = {}
        try:
            weather_row = session.weather_data.iloc[0]
            weather = {
                'airTempC': round(float(weather_row['AirTemp']), 2),
                'trackTempC': round(float(weather_row['TrackTemp']), 2),
                'humidityPct': round(float(weather_row['Humidity']), 2),
                'pressureMbar': round(float(weather_row['Pressure']), 2),
            }
        except Exception:
            weather = {'airTempC': None, 'trackTempC': None, 'humidityPct': None, 'pressureMbar': None}
        event_rows = 0
        for item in clean:
            key = (str(item.get('driver')), int(item['lapNumber']))
            lap = lookup.get(key)
            if lap is None:
                continue
            try:
                features = telemetry_features(lap)
            except Exception:
                features = None
            if not features:
                continue
            lap_time = float(item['lapTimeS'])
            row = {
                'featureVersion': FEATURE_VERSION,
                'year': int(year),
                'round': round_number,
                'eventName': payload['event'].get('name'),
                'driver': key[0],
                'team': team_for_driver.get(key[0], 'UNKNOWN'),
                'lapNumber': key[1],
                'compound': str(item.get('compound') or 'UNKNOWN').upper(),
                'compoundCode': compound_code(item.get('compound')),
                'lapTimeS': round(lap_time, 3),
                'deltaToPoleS': round(lap_time - pole_time, 3),
                'poleTimeS': round(pole_time, 3),
                **weather,
                **features,
            }
            rows.append(row)
            event_rows += 1
        coverage.append({'round': round_number, 'event': payload['event'].get('name'), 'usableRows': event_rows, 'cleanTimingRows': len(clean)})
    return rows, coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    args = parser.parse_args()
    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.Cache.offline_mode(True)
    rows, coverage = qualifying_rows(args.year)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODEL_DIR / f'qualifying_features_{args.year}_v1.json'
    payload = {
        'featureVersion': FEATURE_VERSION,
        'year': args.year,
        'label': 'OBSERVED_PUBLIC_TELEMETRY_FEATURES',
        'target': 'deltaToPoleS',
        'observedChannels': ['timing', 'speed', 'throttle', 'brake', 'RPM', 'position', 'weather'],
        'notObserved': ['ERS deployment command', 'battery SoC', 'energy-management mode'],
        'rows': rows,
        'coverage': coverage,
        'coverageSummary': {
            'completedQualifyingSessions': len(coverage),
            'sessionsWithTelemetry': sum(1 for item in coverage if item['usableRows'] > 0),
            'cleanTimingRows': sum(item['cleanTimingRows'] for item in coverage),
            'usableTelemetryRows': len(rows),
        },
    }
    destination.write_text(json.dumps(payload, separators=(',', ':')), encoding='utf8')
    print(json.dumps({'file': str(destination), 'featureVersion': FEATURE_VERSION, 'rows': len(rows), 'rounds': len(coverage), 'coverage': coverage}))


if __name__ == '__main__':
    main()
