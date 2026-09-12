"""Fit a 2026 qualifying calibration and produce one constrained optimum lap.

The public FastF1 feed exposes car telemetry, not team battery commands. This
therefore keeps the recorded lap as the pace seed and uses the regulation-bound
energy model to redistribute the same electrical budget toward higher-value
full-throttle sections. Every returned value is labelled MODELLED by the UI.
"""

import argparse
import json
from pathlib import Path

import fastf1
import numpy as np

from energy_model import compute_lap_energy
from fetch_f1_session import CACHE_DIR, clean, td_s
from qualifying_straight_mode import qualifying_straight_mode_policy


SESSION_CACHE = CACHE_DIR.parent / 'sessions'
MODEL_CACHE = CACHE_DIR.parent / 'models'
MAX_TELEMETRY_POINTS = 600
MAX_ENERGY_POINTS = 360


def clean_lap(lap):
    return lap.get('lapTimeS') is not None and not lap.get('isOutlier', False)


def training_summary(year):
    """Learn a conservative improvement ceiling from all cached 2026 Q data."""
    qualifying_dir = SESSION_CACHE / str(year)
    sessions = []
    field_deltas = []
    clean_laps = 0
    for path in sorted(qualifying_dir.glob('*_qualifying.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf8'))
        except (OSError, json.JSONDecodeError):
            continue
        laps = [lap for lap in payload.get('laps', []) if clean_lap(lap)]
        drivers = sorted(payload.get('drivers', []), key=lambda driver: driver.get('position') or 99)
        if not laps or not drivers:
            continue
        p1 = drivers[0].get('abbr')
        pole_laps = [lap for lap in laps if lap.get('driver') == p1]
        if not pole_laps:
            continue
        pole = min(lap['lapTimeS'] for lap in pole_laps)
        sessions.append({'round': payload.get('event', {}).get('round'), 'poleTimeS': pole})
        clean_laps += len(laps)
        field_deltas.extend(lap['lapTimeS'] - pole for lap in laps if lap['lapTimeS'] > pole)

    # This is not a claim that the field gap is recoverable. It is a learned,
    # conservative cap on how much a one-lap energy reallocation may claim.
    median_delta = float(np.median(field_deltas)) if field_deltas else 0.5
    p20_delta = float(np.percentile(field_deltas, 20)) if field_deltas else 0.2
    telemetry_training = load_telemetry_training(year)
    return {
        'year': int(year),
        'sessions': len(sessions),
        'rounds': [item['round'] for item in sessions],
        'cleanLaps': int(clean_laps),
        'medianFieldDeltaS': round(median_delta, 3),
        'lowDeltaS': round(p20_delta, 3),
        'telemetrySessions': telemetry_training.get('completedSessions', 0),
        'telemetryDriverLaps': telemetry_training.get('fastestCleanDriverLaps', 0),
        'telemetrySamples': telemetry_training.get('telemetrySamples', 0),
        'observedChannels': telemetry_training.get('channels', []),
        'unobservedChannels': telemetry_training.get('notObserved', ['ERS deployment command', 'battery SoC']),
        'validation': {
            'status': 'LIMITED',
            'reason': 'Public timing and telemetry contain no ground-truth ERS deployment or battery state. The pace calibration is data-backed; the energy plan remains a constrained model.',
        },
        'model': '2026 qualifying pace calibration + constrained energy optimiser',
    }


def load_telemetry_training(year):
    path = MODEL_CACHE / f'qualifying_telemetry_{year}_v1.json'
    try:
        payload = json.loads(path.read_text(encoding='utf8'))
        if payload.get('year') == int(year):
            return payload
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def sampled_indices(length, limit):
    if length <= limit:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, limit).round().astype(int))


def telemetry_payload(lap, tel, times, speeds, scale):
    keep = sampled_indices(len(tel), MAX_TELEMETRY_POINTS)
    distance = tel['Distance'].to_numpy(dtype=float)
    has_position = 'X' in tel.columns and 'Y' in tel.columns
    return {
        'driver': clean(lap.get('Driver')),
        'lapNumber': int(lap['LapNumber']),
        'lapTimeS': round(float(times[-1]), 3),
        'compound': str(compound) if (compound := clean(lap.get('Compound'))) is not None else None,
        'trace': {
            'distance': [round(float(distance[i]), 1) for i in keep],
            'time': [round(float(times[i]), 3) for i in keep],
            'x': [round(float(tel['X'].iloc[i]), 2) for i in keep] if has_position else [],
            'y': [round(float(tel['Y'].iloc[i]), 2) for i in keep] if has_position else [],
            # The time-compressed trace is intentionally modest. It only
            # visualises the energy optimiser's predicted delta, not new car
            # telemetry or a fabricated recorded speed sample.
            'speed': [round(float(min(380.0, speeds[i] / scale)), 1) for i in keep],
            'throttle': [round(float(tel['Throttle'].iloc[i]), 1) for i in keep],
            'brake': [bool(tel['Brake'].iloc[i]) for i in keep],
            'rpm': [round(float(tel['RPM'].iloc[i]), 1) for i in keep],
            'drs': [int(tel['DRS'].iloc[i]) for i in keep],
            'gear': [int(tel['nGear'].iloc[i]) for i in keep],
        },
    }


def optimise_energy(base, speeds, throttle, brake, times, training, reference_time):
    """Redistribute, rather than create, electrical energy across the lap."""
    electric = np.asarray(base['trace']['pElecKw'], dtype=float).copy()
    harvest = np.asarray(base['trace']['pHarvestKw'], dtype=float)
    soc = np.asarray(base['trace']['socMj'], dtype=float)
    distances = np.asarray(base['trace']['distance'], dtype=float)
    straight_mode_eligible = np.asarray(
        base['trace'].get('straightModeEligible', np.ones(len(electric), dtype=bool)),
        dtype=bool,
    )
    # The baseline already has no deployment outside the user-supplied
    # Straight Mode ranges. Keep that hard boundary during optimisation too:
    # redistribution may improve allocation *within* a range, never create
    # battery use in another part of the track.
    electric[~straight_mode_eligible] = 0.0
    dt = np.diff(times, prepend=times[0] - 0.02)
    dt = np.clip(dt, 0.01, 0.5)

    high_speed = np.percentile(speeds, 68)
    target = (throttle >= 92) & (~brake) & (speeds >= high_speed) & straight_mode_eligible
    donor = (throttle >= 65) & (~brake) & (speeds < high_speed) & (electric > 18) & straight_mode_eligible
    # The field-distribution calibration caps the claimed improvement. More
    # observed sessions make the calibration more representative, never the
    # improvement larger. There is deliberately no synthetic "full ERS"
    # target because team deployment commands are not public.
    session_strength = min(1.0, training['sessions'] / 13.0)
    learned_ceiling = max(0.06, min(0.38, training['lowDeltaS'] * 0.28))
    gain = min(learned_ceiling, max(0.06, reference_time * 0.00125 + target.mean() * 0.035))
    gain *= 0.72 + 0.28 * session_strength
    gain = round(float(min(gain, reference_time * 0.0045)), 3)

    # Move 45% of low-value deployment to the highest-speed throttle zones.
    transfer_mj = float(np.sum(electric[donor] * dt[donor]) / 1000.0 * 0.45)
    remaining = transfer_mj
    for index in np.where(target)[0]:
        capacity_kw = max(0.0, 350.0 - electric[index])
        use_mj = min(remaining, capacity_kw * dt[index] / 1000.0)
        electric[index] += use_mj * 1000.0 / dt[index]
        remaining -= use_mj
        if remaining <= 1e-6:
            break
    if transfer_mj > 0:
        scale = max(0.0, 1.0 - (transfer_mj - remaining) / max(1e-6, np.sum(electric[donor] * dt[donor]) / 1000.0))
        electric[donor] *= scale

    delta_power = electric - np.asarray(base['trace']['pElecKw'], dtype=float)
    # Maintain the same energy budget: higher-speed deployment is offset by
    # lower-value deployment, leaving the end-of-lap SoC physically bounded.
    soc_opt = np.clip(soc - np.cumsum(delta_power * dt / 1000.0 / 0.92), 0.0, base['summary']['socWindowMj'])
    optimum_time = max(0.001, reference_time - gain)
    scale_time = optimum_time / max(reference_time, 1e-6)
    summary = dict(base['summary'])
    summary.update({
        'referenceLapTimeS': round(float(reference_time), 3),
        'modelledLapTimeS': round(float(optimum_time), 3),
        'expectedGainS': gain,
        'socEndMj': round(float(soc_opt[-1]), 3),
        'netSocDeltaMj': round(float(soc_opt[-1] - summary['socStartMj']), 3),
        'trainingSessions': training['sessions'],
        'trainingCleanLaps': training['cleanLaps'],
        'trainingTelemetryLaps': training.get('telemetryDriverLaps', 0),
        'trainingTelemetrySamples': training.get('telemetrySamples', 0),
        'predictionConfidence': 'LIMITED — no public ERS/SoC labels',
    })
    return {
        'timeScale': scale_time,
        'gain': gain,
        'summary': summary,
        'trace': {
            'time': times * scale_time,
            'distance': distances,
            'speed': speeds / scale_time,
            'pElecKw': electric,
            'pIceKw': np.asarray(base['trace']['pIceKw'], dtype=float),
            'pHarvestKw': harvest,
            'socMj': soc_opt,
            'clipping': np.asarray(base['trace']['clipping'], dtype=bool),
            'ersInferenceScore': np.asarray(base['trace'].get('ersInferenceScore', np.zeros(len(electric))), dtype=float),
            'straightModeEligible': straight_mode_eligible,
            'straightModeWindowIndex': np.asarray(base['trace'].get('straightModeWindowIndex', np.full(len(electric), -1)), dtype=int),
        },
    }


def build_payload(year, round_number, driver, lap_number):
    training = training_summary(year)
    if not training['sessions']:
        raise ValueError('no cached qualifying sessions available for training')

    event = fastf1.get_event(year, round_number)
    session = event.get_session('Qualifying')
    session.load(laps=True, telemetry=True, weather=True, messages=False)
    match = session.laps[(session.laps['Driver'] == driver) & (session.laps['LapNumber'] == lap_number)]
    if match.empty:
        raise ValueError(f'no qualifying lap {lap_number} for {driver}')
    lap = match.iloc[0]
    tel = lap.get_telemetry()
    raw_times = tel['Time'].dt.total_seconds().to_numpy(dtype=float)
    raw_speeds = tel['Speed'].to_numpy(dtype=float)
    raw_distance = tel['Distance'].to_numpy(dtype=float)
    reference_time = td_s(lap.get('LapTime')) or float(raw_times[-1])
    weather = None
    try:
        weather_row = session.weather_data.iloc[0]
        weather = {'airTempC': float(weather_row['AirTemp']), 'trackTempC': float(weather_row['TrackTemp']), 'pressureMbar': float(weather_row['Pressure']), 'humidityPct': float(weather_row['Humidity'])}
    except Exception:
        pass
    qualifying_policy = qualifying_straight_mode_policy(
        session,
        year=year,
        round_number=round_number,
        trace_length_m=float(raw_distance.max()),
    )
    baseline = compute_lap_energy({
        'time': raw_times, 'speed': raw_speeds, 'throttle': tel['Throttle'].to_numpy(dtype=float),
        'brake': tel['Brake'].to_numpy(dtype=bool), 'rpm': tel['RPM'].to_numpy(dtype=float), 'distance': raw_distance,
    }, year=year, round_number=round_number, session_name='Qualifying', lap_fraction=0.0, weather=weather, high_speed_kph=max(80.0, 0.4 * float(raw_speeds.max())), qualifying_straight_mode_policy=qualifying_policy)
    optimum = optimise_energy(baseline, raw_speeds, tel['Throttle'].to_numpy(dtype=float), tel['Brake'].to_numpy(dtype=bool), raw_times, training, reference_time)
    keep = sampled_indices(len(raw_times), MAX_ENERGY_POINTS)
    trace = optimum['trace']
    return {
        'label': 'MODELLED',
        'kind': 'PITWOLF_OPTIMAL_QUALIFYING_LAP',
        'driver': driver,
        'lapNumber': int(lap_number),
        'training': training,
        'regulation': baseline['regulation'],
        'telemetry': telemetry_payload(lap, tel, trace['time'], raw_speeds, optimum['timeScale']),
        'energy': {
            'label': 'MODELLED', 'summary': optimum['summary'], 'assumptions': baseline['assumptions'], 'citations': baseline['citations'],
            'trace': {key: ([bool(trace[key][i]) for i in keep] if key in {'clipping', 'straightModeEligible'} else [round(float(trace[key][i]), 4 if key == 'socMj' else 1) for i in keep]) for key in trace},
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--driver', required=True)
    parser.add_argument('--lap', type=int, required=True)
    args = parser.parse_args()
    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    print(json.dumps(build_payload(args.year, args.round, args.driver.upper(), args.lap)))


if __name__ == '__main__':
    main()
