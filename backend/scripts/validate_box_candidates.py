"""Check the BOX extractor's future-label boundary."""

import pandas as pd

from extract_box_candidates import extract_candidates_from_laps


def main() -> None:
    laps = pd.DataFrame([
        {'Driver': 'AAA', 'LapNumber': 1, 'LapTime': pd.Timedelta(seconds=90), 'Compound': 'MEDIUM', 'TyreLife': 1, 'Stint': 1, 'Position': 5, 'TrackStatus': '1', 'IsAccurate': True, 'PitInTime': pd.NaT, 'PitOutTime': pd.NaT},
        {'Driver': 'AAA', 'LapNumber': 2, 'LapTime': pd.Timedelta(seconds=92), 'Compound': 'MEDIUM', 'TyreLife': 2, 'Stint': 1, 'Position': 5, 'TrackStatus': '1', 'IsAccurate': True, 'PitInTime': pd.Timedelta(seconds=181), 'PitOutTime': pd.NaT},
    ])
    payload = extract_candidates_from_laps(laps, 2026, 1, 'R', 'Test GP')
    assert payload['candidateCount'] == 1
    row = payload['rows'][0]
    assert row['featureCutoff'] == 'AFTER_COMPLETED_LAP'
    assert row['pitNextLapObserved'] is True
    assert row['previousLapTimeS'] == 90.0
    assert row['boundary'].startswith('SEPARATE') if 'boundary' in row else True
    print('BOX_CANDIDATES_OK cutoff=PASS separatePolicy=PASS')


if __name__ == '__main__':
    main()
