"""Small regression checks for the LAP_START feature cutoff.

These checks intentionally use synthetic rows. They verify that future lap
values cannot change an earlier decision-point feature before the full FastF1
cache is rebuilt.
"""

import pandas as pd

from energy_surrogate import add_surrogate_energy
from extract_decision_points import tyre_degradation_proxy


def main():
    timeline = {
        1: {'tyreLife': 4, 'lapTimeS': 90.0},
        2: {'tyreLife': 5, 'lapTimeS': 91.0},
        3: {'tyreLife': 6, 'lapTimeS': 140.0},
    }
    baseline = tyre_degradation_proxy(timeline, 3)
    changed_future = dict(timeline)
    changed_future[3] = {'tyreLife': 40, 'lapTimeS': 10.0}
    assert tyre_degradation_proxy(changed_future, 3) == baseline, (
        'tyre degradation at lap 3 changed when only lap 3 changed'
    )

    columns = {
        'year': [2023, 2023], 'round': [1, 1], 'session': ['R', 'R'],
        'lap': [2, 3], 'driver': ['AAA', 'AAA'],
        'defender': ['BBB', 'BBB'], 'gapS': [0.8, 0.7],
        'closingRateS': [0.1, 0.2], 'speedDeltaKph': [4.0, 30.0],
        'tyreAgeDiff': [1.0, 2.0], 'attackerLapTimeS': [90.0, 91.0],
        'defenderLapTimeS': [90.5, 91.5],
    }
    original = add_surrogate_energy(pd.DataFrame(columns))
    future_changed = dict(columns)
    future_changed['speedDeltaKph'] = [4.0, -80.0]
    future_changed['attackerLapTimeS'] = [90.0, 5.0]
    future_changed['defenderLapTimeS'] = [90.5, 5.5]
    changed = add_surrogate_energy(pd.DataFrame(future_changed))

    for column in ('attackerSoCMj', 'defenderSoCMj', 'energyDeltaMj'):
        assert original.loc[0, column] == changed.loc[0, column], (
            f'{column} at lap 2 changed when only lap 3 changed'
        )

    print('CAUSAL_FEATURES_OK cutoff=LAP_START future-lap-invariance=PASS')


if __name__ == '__main__':
    main()
