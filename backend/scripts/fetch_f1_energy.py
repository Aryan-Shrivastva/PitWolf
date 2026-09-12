import argparse
import json

import fastf1
import numpy as np

from energy_model import compute_lap_energy
from fia_compliance import event_context, regulation_payload
from fetch_f1_session import CACHE_DIR, td_s
from qualifying_straight_mode import qualifying_straight_mode_policy

MAX_POINTS = 360


def weather_at(weather_data, lap_start_s):
    if weather_data is None or weather_data.empty:
        return None
    times = weather_data['Time'].dt.total_seconds().to_numpy()
    row = weather_data.iloc[int(np.argmin(np.abs(times - lap_start_s)))]
    return {
        'airTempC': float(row['AirTemp']),
        'trackTempC': float(row['TrackTemp']),
        'pressureMbar': float(row['Pressure']),
        'humidityPct': float(row['Humidity']),
    }


def build_energy_payload(year, round_number, session_name, driver, lap_number):
    event = fastf1.get_event(year, round_number)
    session = event.get_session(session_name)
    session.load(laps=True, telemetry=True, weather=True, messages=False)

    match = session.laps[(session.laps['Driver'] == driver) & (session.laps['LapNumber'] == lap_number)]
    if match.empty:
        raise ValueError(f'no lap {lap_number} for {driver} in {year} round {round_number} {session_name}')
    lap = match.iloc[0]
    tel = lap.get_telemetry()

    total_laps = session.total_laps or lap_number
    lap_fraction = max(0.0, min(1.0, (lap_number - 1) / float(total_laps)))

    weather = None
    try:
        weather = weather_at(session.weather_data, td_s(lap.get('LapStartTime')) or 0.0)
    except Exception:
        weather = None

    compliance = event_context(year, round_number, session_name)
    qualifying_policy = None
    if str(session_name).lower() in {'qualifying', 'sprint qualifying'}:
        qualifying_policy = qualifying_straight_mode_policy(
            session,
            year=year,
            round_number=round_number,
            trace_length_m=float(tel['Distance'].max()),
        )
    result = compute_lap_energy(
        {
            'time': tel['Time'].dt.total_seconds().to_numpy(),
            'speed': tel['Speed'].to_numpy(dtype=float),
            'throttle': tel['Throttle'].to_numpy(dtype=float),
            'brake': tel['Brake'].to_numpy(dtype=bool),
            'rpm': tel['RPM'].to_numpy(dtype=float),
            'distance': tel['Distance'].to_numpy(dtype=float),
        },
        year=year,
        round_number=round_number,
        session_name=session_name,
        lap_fraction=lap_fraction,
        weather=weather,
        high_speed_kph=max(80.0, 0.4 * float(tel['Speed'].max())),
        compliance_context=compliance,
        qualifying_straight_mode_policy=qualifying_policy,
    )

    trace = result['trace']
    n = len(trace['time'])
    if n > MAX_POINTS:
        keep = np.unique(np.linspace(0, n - 1, MAX_POINTS).round().astype(int))
    else:
        keep = np.arange(n)

    soc_window = result['summary']['socWindowMj']
    return {
        'label': ('PUBLIC_TELEMETRY_CONSTRAINED_ERS_INFERENCE'
                  if str(session_name).lower() in {'qualifying', 'sprint qualifying'} else 'MODELLED'),
        'driver': driver,
        'lapNumber': int(lap_number),
        'lapTimeS': td_s(lap.get('LapTime')),
        'year': year,
        'round': round_number,
        'session': session_name,
        'lapFraction': round(lap_fraction, 3),
        'weather': weather,
        'regulation': result['regulation'],
        'summary': result['summary'],
        'assumptions': result['assumptions'],
        'citations': result['citations'],
        'trace': {
            'distance': [round(float(trace['distance'][i]), 1) for i in keep],
            'time': [round(float(trace['time'][i]), 3) for i in keep],
            'speed': [round(float(trace['speedKph'][i]), 1) for i in keep],
            'pElecKw': [round(float(trace['pElecKw'][i]), 1) for i in keep],
            'pIceKw': [round(float(trace['pIceKw'][i]), 1) for i in keep],
            'pHarvestKw': [round(float(trace['pHarvestKw'][i]), 1) for i in keep],
            'socMj': [round(float(trace['socMj'][i]), 4) for i in keep],
            'socPct': [round(float(trace['socMj'][i] / soc_window * 100.0), 1) for i in keep],
            'clipping': [bool(trace['clipping'][i]) for i in keep],
            'ersInferenceScore': [round(float(trace['ersInferenceScore'][i]), 3) for i in keep],
            'straightModeEligible': [bool(trace['straightModeEligible'][i]) for i in keep],
            'straightModeWindowIndex': [int(trace['straightModeWindowIndex'][i]) for i in keep],
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
    print(json.dumps(build_energy_payload(args.year, args.round, args.session, args.driver, args.lap)))


if __name__ == '__main__':
    import sys
    try:
        main()
    except Exception as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        raise SystemExit(1)
