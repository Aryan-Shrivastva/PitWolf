"""Download/cache missing public FastF1 qualifying telemetry for one season.

This fills only sessions whose locally cached qualifying summary has clean lap
timing but the feature dataset has no usable car telemetry. It intentionally
does not create synthetic traces when FastF1 cannot supply car/position data.
"""

import argparse
import json
from pathlib import Path

import fastf1

from fetch_f1_session import CACHE_DIR


SESSION_DIR = CACHE_DIR.parent / 'sessions'


def incomplete_rounds(year):
    rounds = []
    for source in sorted((SESSION_DIR / str(year)).glob('*_qualifying.json'), key=lambda path: int(path.name.split('_')[0])):
        payload = json.loads(source.read_text(encoding='utf8'))
        clean_laps = sum(1 for lap in payload.get('laps', []) if lap.get('lapTimeS') is not None and not lap.get('isOutlier', False))
        if clean_laps:
            rounds.append(int(payload['event']['round']))
    return rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--rounds', nargs='*', type=int)
    args = parser.parse_args()
    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.Cache.offline_mode(False)
    requested = args.rounds or incomplete_rounds(args.year)
    results = []
    for round_number in requested:
        try:
            session = fastf1.get_event(args.year, round_number).get_session('Qualifying')
            session.load(laps=True, telemetry=True, weather=True, messages=False)
            car_drivers = len(getattr(session, 'car_data', {}) or {})
            position_drivers = len(getattr(session, 'pos_data', {}) or {})
            status = 'READY' if car_drivers and position_drivers else 'UNAVAILABLE'
            results.append({'round': round_number, 'status': status, 'carDataDrivers': car_drivers, 'positionDataDrivers': position_drivers})
        except Exception as error:
            results.append({'round': round_number, 'status': 'FAILED', 'error': str(error)[:240]})
    print(json.dumps({'year': args.year, 'results': results}))


if __name__ == '__main__':
    main()
