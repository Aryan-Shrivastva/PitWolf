"""Validate the shared modelled energy transition and its SoC sensitivity.

This is a deterministic safety check, not a claim about private team battery
telemetry. It verifies that the same transition used by the surrogate and
replay respects the public-model capacity/clipping assumptions across both
attacking and defending cars.
"""

from __future__ import annotations

import json

from energy_transition import (ACTION_PROFILE, CAPACITY_MJ, ENERGY_MODEL_VERSION,
                               get_era_config, transition_soc)


ROW = {
    'gapS': 0.7,
    'closingRateS': 0.18,
    'tyreAgeDiff': 4,
    'attackerTyreDegProxy': 0.25,
    'defenderTyreDegProxy': 0.35,
    'pace': 0.7,
}


def check_case(soc: float, action: str, defending: bool, era: str) -> dict:
    next_soc, deploy, harvest = transition_soc(soc, action, ROW, defending, era=era)
    available = max(0.0, soc) + harvest
    assert 0.0 <= next_soc <= CAPACITY_MJ, (soc, action, next_soc)
    assert deploy >= 0.0 and deploy <= available + 1e-9, (soc, action, deploy, available)
    assert harvest >= 0.0, (soc, action, harvest)
    return {
        'soc': soc,
        'action': action,
        'defending': defending,
        'era': era,
        'nextSocMj': round(next_soc, 3),
        'deployMj': deploy,
        'harvestMj': harvest,
        'clippedToAvailable': deploy < ACTION_PROFILE[action]['deploy'],
    }


def main() -> None:
    cases = [
        check_case(soc, action, defending, era)
        for era in ('2018_2025', '2026')
        for soc in (0.0, 0.08, 0.25, 0.8, 2.0, CAPACITY_MJ)
        for action in ('ATTACK', 'SAVE', 'DELAY')
        for defending in (False, True)
    ]
    depleted_attack = next(item for item in cases if item['era'] == '2026' and item['soc'] == 0.0 and item['action'] == 'ATTACK' and not item['defending'])
    full_save = next(item for item in cases if item['era'] == '2018_2025' and item['soc'] == CAPACITY_MJ and item['action'] == 'SAVE' and not item['defending'])
    assert depleted_attack['deployMj'] <= depleted_attack['harvestMj'] + 1e-9
    assert full_save['nextSocMj'] <= CAPACITY_MJ
    print(json.dumps({
        'status': 'PASS',
        'energyModelVersion': ENERGY_MODEL_VERSION,
        'eraConfigs': {
            era: {
                'id': get_era_config(era)['id'],
                'capacityMj': get_era_config(era)['capacityMj'],
                'calibrationStatus': get_era_config(era)['calibrationStatus'],
            }
            for era in ('2018_2025', '2026')
        },
        'capacityMj': CAPACITY_MJ,
        'cases': len(cases),
        'attackAtZeroSoc': depleted_attack,
        'saveAtCapacity': full_save,
        'notes': [
            'Deployment is clipped to current SoC plus same-lap harvest.',
            'Capacity is enforced for both attacker and defender transitions.',
            'SoC values remain modelled surrogates, not measured team telemetry.',
        ],
    }, indent=2))


if __name__ == '__main__':
    main()
