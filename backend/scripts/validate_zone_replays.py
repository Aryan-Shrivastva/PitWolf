"""Deterministic contract checks for zone replay enrichment."""

from build_zone_replays import build_replay


def main() -> None:
    zone = {
        'year': 2026, 'round': 2, 'session': 'R', 'opportunities': [{
            'lap': 4, 'driver': 'AAA', 'defender': 'BBB', 'gapAtDetectionS': 0.7,
        }],
    }
    decision = {'analysisRows': [{
        'lap': 4, 'driver': 'AAA', 'defender': 'BBB', 'featureCutoff': 'LAP_START',
        'attackerSoCMj': 2.2, 'defenderSoCMj': 2.0, 'energyDeltaMj': 0.2,
        'outcomeLabelEligible': True, 'label': 'ATTACK', 'passedNow': True,
        'held': True, 'observedPassLap': 4, 'observedLeadLaps': 6,
    }]}
    payload = build_replay(zone, decision)
    assert payload['lapStartAssociatedCount'] == 1
    replay = payload['replays'][0]
    assert replay['liveCommandEligible'] is False
    assert replay['replayState']['causalStatus'] == 'OBSERVATIONAL_ASSOCIATION_ONLY'
    assert replay['replayState']['energyState']['source'].startswith('MODELLED')
    print('ZONE_REPLAY_OK association=PASS causalBoundary=PASS energyDisclosure=PASS')


if __name__ == '__main__':
    main()
