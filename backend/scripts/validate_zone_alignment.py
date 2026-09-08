"""Small deterministic checks for FIA-zone parsing and telemetry interpolation."""

from __future__ import annotations

import pandas as pd

from extract_zone_opportunities import (crossing_at_cumulative_distance,
                                        crossing_at_distance,
                                        reference_track_length)
from zone_alignment import zone_alignment_status


class FakeLap(dict):
    def __init__(self, telemetry, **values):
        super().__init__(values)
        self._telemetry = telemetry

    def get_telemetry(self):
        return self._telemetry

    def get_car_data(self):
        return self._telemetry


def main():
    source = {'url': 'https://www.fia.com/example.pdf', 'published': '2026-01-01',
              'documentTitle': 'Test', 'articleOrSection': 'B7.2'}
    partial = {'zoneEvidence': {
        'zoneEvidenceLoaded': True,
        'officialDetectionGapS': 1.0,
        'detectionLineDistanceM': 400.0,
        'activationLineDistanceM': 500.0,
        'eventAppendixSource': source,
    }}
    status = zone_alignment_status(partial, 1000.0)
    assert status['status'] == 'RETROSPECTIVE_ZONE_EVIDENCE_ONLY'
    assert status['telemetryAlignmentAvailable'] is True
    assert status['liveCommandEligible'] is False

    complete = {**partial, 'eventSpecificDataLoaded': True,
                'officialDetectionGapS': 1.0, 'detectionLineDistanceM': 400.0,
                'activationLineDistanceM': 500.0, 'rechargeLimitMj': 8.5,
                'powerLimitProfile': {'kind': 'digitised FIA curve'},
                'eventAppendixSource': source}
    assert zone_alignment_status(complete, 1000.0)['liveCommandEligible'] is True

    telemetry = pd.DataFrame({
        'Distance': [0.0, 500.0, 1000.0],
        'Time': [0.0, 5.0, 10.0],
        'Speed': [200.0, 250.0, 300.0],
    })
    crossing = crossing_at_distance(FakeLap(telemetry, LapStartTime=pd.Timedelta(seconds=100)), 250.0)
    assert crossing == {'sessionTimeS': 102.5, 'speedKph': 225.0}
    cumulative = telemetry.assign(SessionTime=[100.0, 105.0, 110.0])
    assert crossing_at_cumulative_distance(cumulative, 250.0) == {'sessionTimeS': 102.5, 'speedKph': 225.0}
    track_length, source = reference_track_length(FakeLap(cumulative, LapStartTime=pd.Timedelta(seconds=100)))
    assert track_length == 1000.0 and source == 'MERGED_TELEMETRY_DISTANCE'
    print('ZONE_ALIGNMENT_OK retrospective=PASS live-gate=PASS interpolation=PASS')


if __name__ == '__main__':
    main()
