"""Fast deterministic checks for the FIA energy-compliance layer."""

import numpy as np

from energy_model import compute_lap_energy


def trace(speed=350.0, brake=False):
    return {
        'time': np.linspace(0.0, 2.0, 21),
        'speed': np.full(21, speed),
        'throttle': np.full(21, 100.0),
        'brake': np.full(21, brake),
        'rpm': np.full(21, 10000.0),
        'distance': np.linspace(0.0, 200.0, 21),
    }


def complete_qualifying_context():
    return {
        'status': 'FIA_EVENT_LIMITS_LOADED',
        'eventSpecificDataLoaded': True,
        'sessionKind': 'QUALIFYING',
        'rechargeLimitMj': 4.0,
        'powerLimitProfile': {
            'normal': [{'speedKph': 0, 'maxKw': 350}, {'speedKph': 345, 'maxKw': 0}],
        },
        'alternativePowerSectors': [],
        'source': {'url': 'https://www.fia.com/example', 'published': '2026-01-01'},
    }


def main():
    qualifying = compute_lap_energy(trace(), year=2026, round_number=1, session_name='Qualifying')
    summary = qualifying['summary']
    assert summary['carMassKg'] == 770.0, summary['carMassKg']  # 726 + assumed 44 kg nominal tyres
    assert summary['standardEcuEfficiencyCorrection'] == 0.97
    assert qualifying['regulation']['status'] == 'FIA_EVENT_LIMITS_LOADED'
    assert qualifying['regulation']['rechargeLimitMj'] == 7.0
    assert np.max(qualifying['trace']['pElecKw']) <= 350.0 + 1e-9
    assert np.max(qualifying['trace']['pElecKw']) <= 100.0 + 1e-6  # B7.2 selected Overtake curve at 350 kph
    assert np.min(qualifying['trace']['socMj']) >= -1e-9
    assert np.max(qualifying['trace']['socMj']) <= 4.0 + 1e-9
    assert summary['qualifyingErsInference']['status'] == 'PUBLIC_TELEMETRY_CONSTRAINED_INFERENCE'
    assert np.max(qualifying['trace']['ersInferenceScore']) > 0.0

    # User-provided Straight Mode ranges are an explicit *placement policy*,
    # not an FIA source. With no supplied range, qualifying must never invent
    # ERS deployment elsewhere on the lap.
    no_straight_mode = compute_lap_energy(
        trace(speed=180.0), year=2026, round_number=1, session_name='Qualifying',
        qualifying_battery_calibration=False, soc_start_mj=4.0,
        qualifying_straight_mode_policy={
            'policy': 'USER_STRAIGHT_MODE_ZONES_ONLY',
            'status': 'USER_STRAIGHT_MODE_REFERENCE_MISSING',
            'windows': [],
        },
    )
    assert np.max(no_straight_mode['trace']['pElecKw']) == 0.0
    assert not np.any(no_straight_mode['trace']['straightModeEligible'])

    mapped_straight_mode = compute_lap_energy(
        trace(speed=180.0), year=2026, round_number=1, session_name='Qualifying',
        qualifying_battery_calibration=False, soc_start_mj=4.0,
        qualifying_straight_mode_policy={
            'policy': 'USER_STRAIGHT_MODE_ZONES_ONLY',
            'status': 'USER_STRAIGHT_MODE_POLICY_LOADED',
            'windows': [{'startDistanceM': 0.0, 'endDistanceM': 100.0}],
        },
    )
    eligible = mapped_straight_mode['trace']['straightModeEligible']
    mapped_electric = mapped_straight_mode['trace']['pElecKw']
    assert np.max(mapped_electric[eligible]) > 0.0
    assert np.max(mapped_electric[~eligible]) == 0.0

    no_event_document = compute_lap_energy(trace(), year=2026, round_number=14, session_name='Qualifying')
    assert no_event_document['regulation']['status'] == 'FIA_GLOBAL_LIMITS_ONLY'

    configured = compute_lap_energy(
        trace(speed=180.0), year=2026, round_number=1, session_name='Qualifying',
        compliance_context=complete_qualifying_context(),
    )
    assert configured['summary']['harvestCapMj'] == 4.0
    assert configured['summary']['eventRechargeLimitLoaded'] is True
    assert configured['regulation']['status'] == 'FIA_EVENT_LIMITS_LOADED'
    print('FIA compliance layer checks passed')


if __name__ == '__main__':
    main()
