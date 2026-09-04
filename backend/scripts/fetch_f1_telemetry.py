import argparse
import json

import fastf1
import numpy as np
import pandas as pd

from fetch_f1_session import CACHE_DIR, clean, td_s


def build_telemetry_payload(year, round_number, session_name, driver, lap_number):
    event = fastf1.get_event(year, round_number)
    session = event.get_session(session_name)
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    match = session.laps[(session.laps['Driver'] == driver) & (session.laps['LapNumber'] == lap_number)]
    if match.empty:
        raise ValueError(f'no lap {lap_number} for {driver} in {year} round {round_number} {session_name}')
    lap = match.iloc[0]
    tel = lap.get_telemetry()

    if len(tel) > 600:
        keep = np.unique(np.linspace(0, len(tel) - 1, 600).round().astype(int))
        tel = tel.iloc[keep]

    times = tel['Time'].dt.total_seconds().to_numpy()
    distances = tel['Distance'].to_numpy()

    # Keep the car coordinates with the lap trace so replay views can use the
    # actual GPS path for each lap instead of repeating one reference lap.
    has_position = 'X' in tel.columns and 'Y' in tel.columns

    markers = {}
    s1 = clean(lap.get('Sector1Time'))
    s2 = clean(lap.get('Sector2Time'))
    if s1 is not None:
        markers['s2Distance'] = round(float(distances[np.argmin(np.abs(times - s1.total_seconds()))]), 1)
    if s1 is not None and s2 is not None:
        cumulative = s1.total_seconds() + s2.total_seconds()
        markers['s3Distance'] = round(float(distances[np.argmin(np.abs(times - cumulative))]), 1)

    result = session.results[session.results['Abbreviation'] == driver]
    team_color = None
    team = None
    if not result.empty:
        team_color = clean(result.iloc[0].get('TeamColor'))
        team = clean(result.iloc[0].get('TeamName'))

    return {
        'driver': driver,
        'team': team,
        'teamColorHex': f'#{team_color}' if team_color else None,
        'lapNumber': int(lap_number),
        'lapTimeS': td_s(lap.get('LapTime')),
        'compound': str(c) if (c := clean(lap.get('Compound'))) is not None else None,
        'sectorMarkers': markers,
        'trace': {
            'distance': [round(float(v), 1) for v in distances],
            'time': [round(float(v), 3) for v in times],
            'x': [round(float(v), 2) for v in tel['X']] if has_position else [],
            'y': [round(float(v), 2) for v in tel['Y']] if has_position else [],
            'speed': [round(float(v), 1) for v in tel['Speed']],
            'throttle': [round(float(v), 1) for v in tel['Throttle']],
            'brake': [bool(v) for v in tel['Brake']],
            'rpm': [round(float(v), 1) for v in tel['RPM']],
            'drs': [int(v) for v in tel['DRS']],
            'gear': [int(v) for v in tel['nGear']],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--driver', required=True)
    parser.add_argument('--lap', type=int, required=True)
    args = parser.parse_args()

    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    print(json.dumps(build_telemetry_payload(args.year, args.round, args.session, args.driver, args.lap)))


if __name__ == '__main__':
    import sys
    try:
        main()
    except Exception as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        raise SystemExit(1)
