import argparse
import json
from pathlib import Path

import fastf1

from fetch_f1_session import CACHE_DIR, clean, session_names


def has_recorded_race_data(year, round_number, event_date):
    """Check whether this local install has enough cached Race data to replay.

    The schedule lists every 2026 round ahead of time. A scheduled event is
    not automatically replayable: the visual replay needs cached FastF1 Race
    timing and position data. Keep that distinction explicit so the UI does
    not try to construct a simulation from a future (or not-yet-ingested)
    event.
    """
    session_json = CACHE_DIR.parent / 'sessions' / str(year) / f'{int(round_number)}_race.json'
    if session_json.exists():
        return True
    if event_date is None:
        return False
    season_dir = CACHE_DIR / str(year)
    if not season_dir.exists():
        return False
    event_prefix = f'{event_date.date().isoformat()}_'
    for event_dir in season_dir.glob(f'{event_prefix}*'):
        if not event_dir.is_dir():
            continue
        for session_dir in event_dir.glob('*_Race'):
            if not session_dir.is_dir():
                continue
            has_timing = (session_dir / '_extended_timing_data.ff1pkl').exists()
            has_positions = (session_dir / 'position_data.ff1pkl').exists()
            if has_timing and has_positions:
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    args = parser.parse_args()

    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    schedule = fastf1.get_event_schedule(args.year, include_testing=False)

    events = []
    for _, row in schedule.iterrows():
        event_date = clean(row.get('EventDate'))
        events.append({
            'round': int(row['RoundNumber']),
            'name': row['EventName'],
            'officialName': row['OfficialEventName'],
            'country': row['Country'],
            'location': row['Location'],
            'date': str(event_date.date()) if event_date is not None else None,
            'sessions': session_names(row),
            # This tells the simulation whether it can load a recorded Race
            # replay, not whether an event merely appears on the calendar.
            'raceDataAvailable': has_recorded_race_data(args.year, int(row['RoundNumber']), event_date),
        })

    print(json.dumps({'year': args.year, 'events': events}))


if __name__ == '__main__':
    import sys
    try:
        main()
    except Exception as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        raise SystemExit(1)
