"""Race-wide energy trace for one driver: continuous battery SoC across all
laps (pit stops do not recharge the on-track state) plus the three validation
gates required by the PitWolf energy-engine spec:

1. zone alignment  — harvest concentrates at high-speed braking, deployment at
                     full throttle;
2. ceilings        — per-lap harvest <= 8.5 MJ (Art. 5.4.10), SoC swing within
                     the 4 MJ window (Art. 5.4.9);
3. cross-track consistency — per-circuit totals, to be compared across events
                     (the comparison itself is reported by the caller).
"""

import argparse
import json

import fastf1
import numpy as np

from energy_model import ASSUMPTIONS, compute_lap_energy
from fetch_f1_session import CACHE_DIR, td_s
import fia_2026_regs as regs


def weather_row(weather_data, session_seconds):
    if weather_data is None or weather_data.empty:
        return None
    times = weather_data['Time'].dt.total_seconds().to_numpy()
    row = weather_data.iloc[int(np.argmin(np.abs(times - session_seconds)))]
    return {
        'airTempC': float(row['AirTemp']),
        'trackTempC': float(row['TrackTemp']),
        'pressureMbar': float(row['Pressure']),
        'humidityPct': float(row['Humidity']),
    }


def build_race_energy_payload(year, round_number, session_name, driver):
    event = fastf1.get_event(year, round_number)
    session = event.get_session(session_name)
    session.load(laps=True, telemetry=True, weather=True, messages=False)

    laps = session.laps[session.laps['Driver'] == driver].sort_values('LapNumber')
    if laps.empty:
        raise ValueError(f'no laps for {driver} in {year} round {round_number} {session_name}')

    weather_data = session.weather_data
    total_laps = session.total_laps or int(laps['LapNumber'].max())
    soc_window = regs.CONSTANTS['es_soc_window_mj']['value']

    soc = 0.7 * soc_window
    lap_rows = []
    harvests, deploys, fuels, swings, clip_total = [], [], [], [], 0.0
    deploy_ft_mj = deploy_all_mj = harvest_hs_mj = harvest_all_mj = 0.0

    usable = []
    race_max_speed = 0.0
    for _, lap in laps.iterrows():
        tel = lap.get_telemetry()
        if tel is None or len(tel) < 5:
            continue
        usable.append((lap, tel))
        race_max_speed = max(race_max_speed, float(tel['Speed'].max()))
    # Street circuits brake from far below 140 kph, so judge "high speed"
    # harvest relative to this driver's own top speed on the day.
    high_speed_kph = max(80.0, 0.4 * race_max_speed)

    for lap, tel in usable:
        lap_number = int(lap['LapNumber'])
        lap_fraction = max(0.0, min(1.0, (lap_number - 1) / float(total_laps)))
        weather = None
        try:
            weather = weather_row(weather_data, td_s(lap.get('LapStartTime')) or 0.0)
        except Exception:
            weather = None

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
            lap_fraction=lap_fraction,
            weather=weather,
            soc_start_mj=soc,
            high_speed_kph=high_speed_kph,
        )
        summary = result['summary']

        pit_in = bool(lap.get('PitInLane', False))
        pit_out = bool(lap.get('PitOutLane', False))
        soc = summary['socEndMj']

        lap_rows.append({
            'lap': lap_number,
            'lapTimeS': td_s(lap.get('LapTime')),
            'compound': str(c) if (c := lap.get('Compound')) is not None and not (isinstance(c, float) and np.isnan(c)) else None,
            'pit': pit_in or pit_out,
            'deployMj': summary['deployMj'],
            'harvestMj': summary['harvestMj'],
            'fuelEnergyMj': summary['fuelEnergyMj'],
            'socStartMj': summary['socStartMj'],
            'socEndMj': round(soc, 3),
            'clipSeconds': summary['clipSeconds'],
        })

        harvests.append(summary['harvestMj'])
        deploys.append(summary['deployMj'])
        fuels.append(summary['fuelEnergyMj'])
        swings.append(summary['socEndMj'] - summary['socStartMj'])
        clip_total += summary['clipSeconds']
        deploy_ft_mj += summary['deployFullThrottleMj']
        deploy_all_mj += summary['deployMj']
        harvest_hs_mj += summary['harvestHighSpeedMj']
        harvest_all_mj += summary['harvestMj']

    harvest_cap = regs.CONSTANTS['harvest_max_mj_per_lap']['value']
    gates = {
        'zoneAlignment': {
            'deployAtFullThrottlePct': round(100.0 * deploy_ft_mj / deploy_all_mj, 1) if deploy_all_mj else None,
            'harvestAtHighSpeedPct': round(100.0 * harvest_hs_mj / harvest_all_mj, 1) if harvest_all_mj else None,
            'highSpeedKph': round(high_speed_kph, 0),
            'pass': bool((deploy_all_mj == 0 or deploy_ft_mj / deploy_all_mj >= 0.35)
                         and (harvest_all_mj == 0 or harvest_hs_mj / harvest_all_mj >= 0.5)),
            'description': 'Deployment should concentrate at full throttle (street circuits legitimately deploy under traction-limited partial throttle, hence the 35% floor); harvest at braking entries above 40% of the driver top speed.',
        },
        'ceilings': {
            'maxHarvestPerLapMj': round(float(np.max(harvests)), 3) if harvests else 0.0,
            'harvestCapMj': harvest_cap,
            'maxSocSwingMj': round(float(np.max(np.abs(swings))) if swings else 0.0, 3),
            'socWindowMj': soc_window,
            'pass': bool(harvests) and float(np.max(harvests)) <= harvest_cap + 1e-6
                    and float(np.max(np.abs(swings))) <= soc_window + 1e-6,
            'description': 'Per-lap harvest cap (PU TR 5.4.10) and 4 MJ SoC window (PU TR 5.4.9).',
        },
        'crossTrackConsistency': {
            'meanDeployMjPerLap': round(float(np.mean(deploys)), 3) if deploys else None,
            'meanHarvestMjPerLap': round(float(np.mean(harvests)), 3) if harvests else None,
            'meanFuelEnergyMjPerLap': round(float(np.mean(fuels)), 2) if fuels else None,
            'totalClipSeconds': round(clip_total, 1),
            'description': 'Per-circuit totals; consistency is judged by comparing these across events of different character.',
        },
    }

    return {
        'label': 'MODELLED',
        'driver': driver,
        'year': year,
        'round': round_number,
        'session': session_name,
        'totalLaps': total_laps,
        'laps': lap_rows,
        'gates': gates,
        'assumptions': ASSUMPTIONS,
        'pitDoesNotRechargeEnergy': True,
        'citations': {
            'harvestCap': regs.CONSTANTS['harvest_max_mj_per_lap']['citation'],
            'socWindow': regs.CONSTANTS['es_soc_window_mj']['citation'],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--driver', required=True)
    args = parser.parse_args()

    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    print(json.dumps(build_race_energy_payload(args.year, args.round, args.session, args.driver)))


if __name__ == '__main__':
    import sys
    try:
        main()
    except Exception as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        raise SystemExit(1)
