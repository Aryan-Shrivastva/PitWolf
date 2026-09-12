"""Build a reproducible 2026 qualifying telemetry training set for PitWolf.

FastF1 exposes public speed, throttle, brake, RPM and position channels. It
does not expose the teams' ERS commands or battery state.  This builder saves
only the observable inputs and descriptive coverage metrics, so the optimiser
can be calibrated on more than a single reference lap without implying that it
has learned private deployment data.
"""

import argparse
import json
from pathlib import Path

import fastf1
import numpy as np

from fetch_f1_session import CACHE_DIR, td_s


SESSION_DIR = CACHE_DIR.parent / 'sessions'
MODEL_DIR = CACHE_DIR.parent / 'models'


def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def samples_for_lap(lap):
    telemetry = lap.get_telemetry()
    if len(telemetry) < 20:
        return None
    time = telemetry['Time'].dt.total_seconds().to_numpy(dtype=float)
    dt = np.clip(np.diff(time, prepend=time[0] - 0.02), 0.01, 0.5)
    speed = telemetry['Speed'].to_numpy(dtype=float)
    throttle = telemetry['Throttle'].to_numpy(dtype=float)
    brake = telemetry['Brake'].to_numpy(dtype=bool)
    full_throttle = (throttle >= 95.0) & ~brake
    coast = (throttle < 15.0) & ~brake
    return {
        'samples': int(len(telemetry)),
        'lapTimeS': round(float(td_s(lap.get('LapTime')) or time[-1]), 3),
        'topSpeedKph': round(float(np.percentile(speed, 99)), 1),
        'meanSpeedKph': round(float(np.mean(speed)), 1),
        'fullThrottleS': round(float(np.sum(dt[full_throttle])), 2),
        'brakingS': round(float(np.sum(dt[brake])), 2),
        'coastS': round(float(np.sum(dt[coast])), 2),
        'highSpeedFullThrottleS': round(float(np.sum(dt[full_throttle & (speed >= np.percentile(speed, 70))])), 2),
    }


def fastest_clean_laps(session, payload):
    """Use the cached session summary's clean-lap flags as the source of truth."""
    clean = [lap for lap in payload.get('laps', []) if lap.get('lapTimeS') is not None and not lap.get('isOutlier', False)]
    fastest = {}
    for item in clean:
        driver = item.get('driver')
        if driver and (driver not in fastest or item['lapTimeS'] < fastest[driver]['lapTimeS']):
            fastest[driver] = item
    result = []
    for driver, item in fastest.items():
        matches = session.laps[(session.laps['Driver'] == driver) & (session.laps['LapNumber'] == item['lapNumber'])]
        if not matches.empty:
            result.append((driver, item, matches.iloc[0]))
    return result


def build(year):
    source_dir = SESSION_DIR / str(year)
    outputs = []
    for source in sorted(source_dir.glob('*_qualifying.json'), key=lambda item: int(item.name.split('_')[0])):
        try:
            payload = json.loads(source.read_text(encoding='utf8'))
            event = payload['event']
            round_number = int(event['round'])
            session = fastf1.get_event(year, round_number).get_session('Qualifying')
            session.load(laps=True, telemetry=True, weather=False, messages=False)
        except Exception as error:
            outputs.append({'round': source.name.split('_')[0], 'status': 'unavailable', 'reason': str(error)[:180]})
            continue
        records = []
        for driver, item, lap in fastest_clean_laps(session, payload):
            try:
                record = samples_for_lap(lap)
            except Exception:
                record = None
            if record:
                record.update({'driver': driver, 'lapNumber': int(item['lapNumber']), 'compound': item.get('compound')})
                records.append(record)
        outputs.append({'round': round_number, 'event': event.get('name'), 'status': 'ready', 'laps': records})

    ready = [round_data for round_data in outputs if round_data.get('status') == 'ready']
    laps = [lap for round_data in ready for lap in round_data['laps']]
    summary = {
        'year': year,
        'dataset': 'public FastF1 qualifying telemetry',
        'label': 'OBSERVED INPUTS ONLY',
        'completedSessions': len(ready),
        'fastestCleanDriverLaps': len(laps),
        'telemetrySamples': int(sum(lap['samples'] for lap in laps)),
        'channels': ['speed', 'throttle', 'brake', 'RPM', 'position'],
        'notObserved': ['ERS deployment command', 'battery SoC', 'energy management mode'],
        'rounds': outputs,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODEL_DIR / f'qualifying_telemetry_{year}_v1.json'
    destination.write_text(json.dumps(summary, separators=(',', ':')), encoding='utf8')
    return destination, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    args = parser.parse_args()
    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    # Training must be reproducible from the local project cache.  In
    # particular, do not let a background cache revalidation turn this into a
    # slow or inconsistent network-dependent job.
    fastf1.Cache.offline_mode(True)
    destination, summary = build(args.year)
    print(json.dumps({'file': str(destination), 'sessions': summary['completedSessions'], 'laps': summary['fastestCleanDriverLaps'], 'samples': summary['telemetrySamples']}))


if __name__ == '__main__':
    main()
