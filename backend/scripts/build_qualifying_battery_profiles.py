"""Build one normalised qualifying battery reference per 2026 driver/event.

Each reference is the driver's fastest clean Soft-tyre qualifying lap.  It is
not measured team SoC: the output records PitWolf's FIA-bounded 4 MJ usable
window, normalised to 100% at the timing line and 0% at lap end when physically
reachable from the public telemetry-derived energy model.
"""

import argparse
import json
from pathlib import Path

import fastf1
import numpy as np

from energy_model import compute_lap_energy
from fetch_f1_energy import weather_at
from fetch_f1_session import CACHE_DIR, td_s
from fia_compliance import event_context
from qualifying_straight_mode import qualifying_straight_mode_policy


SESSION_DIR = CACHE_DIR.parent / 'sessions'
MODEL_DIR = CACHE_DIR.parent / 'models'
PROFILE_VERSION = 'qualifying-battery-profiles.v6'


def clean_soft_reference(laps, driver):
    candidates = [
        item for item in laps
        if item.get('driver') == driver
        and str(item.get('compound') or '').upper() == 'SOFT'
        and item.get('lapTimeS') is not None
        and not item.get('isOutlier', False)
    ]
    return min(candidates, key=lambda item: item['lapTimeS']) if candidates else None


def profile_for_lap(lap, *, session, year, round_number, weather, compliance):
    telemetry = lap.get_telemetry()
    if telemetry is None or len(telemetry) < 20:
        return None
    qualifying_policy = qualifying_straight_mode_policy(
        session,
        year=year,
        round_number=round_number,
        trace_length_m=float(telemetry['Distance'].max()),
    )
    result = compute_lap_energy(
        {
            'time': telemetry['Time'].dt.total_seconds().to_numpy(dtype=float),
            'speed': telemetry['Speed'].to_numpy(dtype=float),
            'throttle': telemetry['Throttle'].to_numpy(dtype=float),
            'brake': telemetry['Brake'].to_numpy(dtype=bool),
            'rpm': telemetry['RPM'].to_numpy(dtype=float),
            'distance': telemetry['Distance'].to_numpy(dtype=float),
        },
        year=year,
        round_number=round_number,
        session_name='Qualifying',
        weather=weather,
        high_speed_kph=max(80.0, 0.4 * float(telemetry['Speed'].max())),
        compliance_context=compliance,
        qualifying_straight_mode_policy=qualifying_policy,
    )
    summary = result['summary']
    return {
        'lapTimeS': round(float(td_s(lap.get('LapTime'))), 3),
        'socStartPercent': summary.get('socStartPercent'),
        'socEndPercent': summary.get('socEndPercent'),
        'calibratedToLapEnd': bool(summary.get('qualifyingBatteryCalibrated')),
        'deploymentScale': summary.get('qualifyingDeploymentScale'),
        'deployMj': summary['deployMj'],
        'harvestMj': summary['harvestMj'],
        'rechargeCapMj': summary['harvestCapMj'],
        'socWindowMj': summary['socWindowMj'],
        'ersInference': summary.get('qualifyingErsInference'),
        'deploymentPolicy': summary.get('qualifyingBatteryDeploymentPolicy'),
        'inferredFull350KwSeconds': summary.get('inferredFull350KwSeconds'),
        'inferredHighPriorityDeploySeconds': summary.get('inferredHighPriorityDeploySeconds'),
        'samples': int(len(telemetry)),
    }


def build(year):
    profiles, unavailable, rounds = [], [], []
    source_files = sorted((SESSION_DIR / str(year)).glob('*_qualifying.json'), key=lambda path: int(path.name.split('_')[0]))
    for source in source_files:
        payload = json.loads(source.read_text(encoding='utf8'))
        round_number = int(payload['event']['round'])
        event_name = payload['event'].get('name')
        session = fastf1.get_event(year, round_number).get_session('Qualifying')
        session.load(laps=True, telemetry=True, weather=True, messages=False)
        lookup = {(str(row.get('Driver')), int(row.get('LapNumber'))): row for _, row in session.laps.iterrows() if row.get('Driver') is not None}
        teams = {str(row.get('Abbreviation')): str(row.get('TeamName') or 'UNKNOWN') for _, row in session.results.iterrows()}
        compliance = event_context(year, round_number, 'Qualifying')
        loaded = 0
        for driver in payload.get('drivers', []):
            abbreviation = str(driver.get('abbr') or '')
            reference = clean_soft_reference(payload.get('laps', []), abbreviation)
            if reference is None:
                unavailable.append({'round': round_number, 'event': event_name, 'driver': abbreviation, 'reason': 'NO_CLEAN_SOFT_QUALIFYING_LAP'})
                continue
            lap = lookup.get((abbreviation, int(reference['lapNumber'])))
            if lap is None:
                unavailable.append({'round': round_number, 'event': event_name, 'driver': abbreviation, 'reason': 'TELEMETRY_REFERENCE_NOT_FOUND'})
                continue
            try:
                weather = weather_at(session.weather_data, td_s(lap.get('LapStartTime')) or 0.0)
                profile = profile_for_lap(lap, session=session, year=year, round_number=round_number, weather=weather, compliance=compliance)
            except Exception as error:
                unavailable.append({'round': round_number, 'event': event_name, 'driver': abbreviation, 'reason': f'PROFILE_FAILED: {error}'})
                continue
            if profile is None:
                unavailable.append({'round': round_number, 'event': event_name, 'driver': abbreviation, 'reason': 'INSUFFICIENT_TELEMETRY'})
                continue
            profiles.append({
                'year': year,
                'round': round_number,
                'event': event_name,
                'driver': abbreviation,
                'team': teams.get(abbreviation, 'UNKNOWN'),
                'lapNumber': int(reference['lapNumber']),
                'compound': 'SOFT',
                'source': 'RECORDED_PUBLIC_FASTF1_TELEMETRY',
                'battery': profile,
            })
            loaded += 1
        rounds.append({'round': round_number, 'event': event_name, 'profiles': loaded})
    return {
        'profileVersion': PROFILE_VERSION,
        'year': year,
        'label': 'MODELLED_QUALIFYING_USABLE_SOC_REFERENCES',
        'definition': 'Each row is a driver\'s fastest clean Soft qualifying lap. 100–0% maps to the FIA 4 MJ usable on-track SoC window, not private team battery telemetry. Qualifying ERS deployment is modelled only inside the user-supplied Straight Mode circuit ranges; absent maps deliberately produce no qualifying deployment.',
        'profiles': profiles,
        'unavailable': unavailable,
        'coverage': rounds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2026)
    args = parser.parse_args()
    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.Cache.offline_mode(True)
    output = build(args.year)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f'qualifying_battery_profiles_{args.year}_v6.json'
    path.write_text(json.dumps(output, separators=(',', ':')), encoding='utf8')
    print(json.dumps({'file': str(path), 'profiles': len(output['profiles']), 'unavailable': len(output['unavailable']), 'rounds': len(output['coverage'])}))


if __name__ == '__main__':
    main()
