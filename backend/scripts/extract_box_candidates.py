"""Extract causal-cutoff, per-driver pit-window candidates for a separate BOX model.

This is intentionally independent from the battle-action classifier. A BOX
candidate is recorded just after a completed lap; its target is whether the
car enters the pit lane on the following lap. It is an observed historical
pit-window label, not a claimed radio command or an optimal tyre strategy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd

from fetch_f1_session import CACHE_DIR


ROOT = Path(CACHE_DIR).parent
OUT_ROOT = ROOT / 'box-candidates'
SCHEMA_VERSION = 'box-candidate.v1'


def seconds(value: Any) -> float | None:
    try:
        value = float(value.total_seconds()) if hasattr(value, 'total_seconds') else float(value)
        return value if value == value else None
    except (TypeError, ValueError):
        return None


def clean(value: Any) -> Any:
    return None if pd.isna(value) else value


def extract_candidates_from_laps(laps: Any, year: int, round_number: int, session: str, event_name: str) -> dict[str, Any]:
    """Build one prior-lap state per driver without using future-lap inputs."""
    rows = []
    drivers = []
    for driver, driver_laps in laps.groupby('Driver', sort=False):
        if not isinstance(driver, str) or not driver:
            continue
        driver_laps = driver_laps.sort_values('LapNumber').reset_index(drop=True)
        if len(driver_laps) < 2:
            continue
        drivers.append(driver)
        for index in range(len(driver_laps) - 1):
            observed, next_lap = driver_laps.iloc[index], driver_laps.iloc[index + 1]
            lap_number = clean(observed.get('LapNumber'))
            next_number = clean(next_lap.get('LapNumber'))
            if lap_number is None or next_number is None or int(next_number) != int(lap_number) + 1:
                continue
            # The candidate point is after ``observed`` has completed. Values
            # from next_lap are used only for the target / exclusions.
            current_pit = pd.notna(observed.get('PitInTime')) or pd.notna(observed.get('PitOutTime'))
            next_pit_in = pd.notna(next_lap.get('PitInTime'))
            next_pit_out = pd.notna(next_lap.get('PitOutTime'))
            if current_pit or pd.isna(observed.get('LapTime')):
                continue
            track_status = str(clean(observed.get('TrackStatus')) or '')
            rows.append({
                'year': int(year),
                'round': int(round_number),
                'session': session,
                'eventName': event_name,
                'driver': driver,
                'lap': int(lap_number),
                'featureCutoff': 'AFTER_COMPLETED_LAP',
                'compound': str(clean(observed.get('Compound')) or 'UNKNOWN'),
                'tyreLifeLaps': clean(observed.get('TyreLife')),
                'stint': clean(observed.get('Stint')),
                'position': clean(observed.get('Position')),
                'previousLapTimeS': seconds(observed.get('LapTime')),
                'trackStatus': track_status,
                'isAccurate': bool(clean(observed.get('IsAccurate')) is not False),
                'pitNextLapObserved': bool(next_pit_in),
                'nextLapPitOutObserved': bool(next_pit_out),
                'outcomeEligible': bool(track_status == '1' and not next_pit_out),
                'labelSource': 'OBSERVED_NEXT_LAP_PIT_IN',
                'interpretation': 'Historical pit-window observation; not an optimal or authorised BOX command.',
            })
    return {
        'schemaVersion': SCHEMA_VERSION,
        'year': int(year),
        'round': int(round_number),
        'session': session,
        'eventName': event_name,
        'drivers': sorted(set(drivers)),
        'candidateCount': len(rows),
        'eligibleCount': sum(row['outcomeEligible'] for row in rows),
        'observedPitNextLapCount': sum(row['pitNextLapObserved'] for row in rows),
        'rows': rows,
        'boundary': 'SEPARATE_FROM_ATTACK_SAVE_DELAY_AND_REPLAY_ONLY',
    }


def build_candidates(year: int, round_number: int, session: str = 'R') -> dict[str, Any]:
    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.Cache.offline_mode(True)
    event = fastf1.get_event(year, round_number)
    race = event.get_session(session)
    race.load(laps=True, telemetry=False, weather=False, messages=False)
    return extract_candidates_from_laps(race.laps, year, round_number, session, str(event.get('EventName', '')))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--session', default='R')
    parser.add_argument('--output', action='store_true')
    args = parser.parse_args()
    payload = build_candidates(args.year, args.round, args.session)
    if args.output:
        target = OUT_ROOT / str(args.year) / f'{args.round}_{args.session.lower()}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
