"""Clip C5.2 energy against an already-fetched FastF1 race JSON.

Uses only the session cache we already have (no new API download):
REAL  — lap times, compounds, pit flags, classified order
DERIVED — race-time gap and closing rate from those lap times
MODELLED — 0-4 MJ ES window after C5.2 clips

The compact session files do not store trap speed, so the 345 km/h cut
usually will not fire. Gap/closing still come from actual laps.

Default race is the first 2026 file on disk (Australian GP).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_energy_rollout import rollout
from fetch_f1_session import CACHE_DIR

SESSIONS = CACHE_DIR.parent / 'sessions'
SESSION_ALIASES = {
    'r': 'race', 'q': 'qualifying', 's': 'sprint', 'sq': 'sprint_qualifying',
    'ss': 'sprint_shootout', 'p': 'practice_1', 'fp1': 'practice_1',
    'fp2': 'practice_2', 'fp3': 'practice_3',
}


def session_slug(session_name: str) -> str:
    raw = (session_name or 'Race').lower().replace(' ', '_')
    return SESSION_ALIASES.get(raw, raw)


def session_path(year: int, round_number: int, session_name: str = 'Race') -> Path:
    return SESSIONS / str(year) / f'{round_number}_{session_slug(session_name)}.json'


def first_cached_race() -> Path | None:
    for year in range(2018, 2027):
        folder = SESSIONS / str(year)
        if not folder.exists():
            continue
        races = sorted(folder.glob('*_race.json'), key=lambda path: int(path.name.split('_')[0]))
        if races:
            return races[0]
    return None


def load_session(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not payload.get('laps'):
        raise SystemExit(json.dumps({'error': f'no laps in {path}'}))
    return payload


def pick_pair(payload: dict, driver: str | None, defender: str | None) -> tuple[str, str]:
    classified = sorted(
        (item for item in payload.get('drivers') or [] if item.get('abbr') and item.get('position')),
        key=lambda item: item['position'],
    )
    if driver and defender:
        return driver.upper(), defender.upper()
    if driver:
        code = driver.upper()
        index = next((i for i, item in enumerate(classified) if item['abbr'] == code), None)
        if index is None and classified:
            return classified[0]['abbr'], classified[1]['abbr']
        neighbour = classified[index + 1] if index is not None and index + 1 < len(classified) else classified[index - 1]
        return code, neighbour['abbr']
    if len(classified) < 2:
        raise SystemExit(json.dumps({'error': 'need two classified drivers in the cached race'}))
    return classified[0]['abbr'], classified[1]['abbr']


def laps_for(payload: dict, driver: str) -> dict[int, dict]:
    out = {}
    for row in payload.get('laps') or []:
        if row.get('driver') != driver:
            continue
        lap = row.get('lapNumber')
        if lap is None:
            continue
        out[int(lap)] = row
    return out


def build_actual_laps(payload: dict, driver: str, defender: str) -> list[dict]:
    ours = laps_for(payload, driver)
    theirs = laps_for(payload, defender)
    shared = sorted(set(ours) & set(theirs))
    our_time = 0.0
    their_time = 0.0
    rows = []
    for lap in shared:
        ours_lap = ours[lap]
        theirs_lap = theirs[lap]
        our_s = ours_lap.get('lapTimeS')
        their_s = theirs_lap.get('lapTimeS')
        if our_s is None or their_s is None:
            continue
        our_time += float(our_s)
        their_time += float(their_s)
        gap = round(our_time - their_time, 3)
        closing = round(float(their_s) - float(our_s), 3)
        rows.append({
            'lap': lap,
            'driver': driver,
            'defender': defender,
            'gapS': gap,
            'closingRateS': closing,
            'ourLapTimeS': our_s,
            'defenderLapTimeS': their_s,
            'attackerCompound': ours_lap.get('compound'),
            'defenderCompound': theirs_lap.get('compound'),
            'pitDistorted': bool(ours_lap.get('isPitLap') or theirs_lap.get('isPitLap')),
            'source': 'CACHED_SESSION_LAP_TIMES',
        })
    return rows


def clip_cached(year: int | None, round_number: int | None, session_name: str,
                driver: str | None, defender: str | None, policy: str) -> dict:
    if year and round_number:
        path = session_path(year, round_number, session_name)
        if not path.exists():
            return {'error': f'no cached {session_name} for {year} R{round_number} at {path}'}
    else:
        path = first_cached_race()
        if path is None:
            return {'error': 'no cached race JSON under data/f1-cache/sessions'}
    payload = load_session(path)
    event = payload.get('event') or {}
    year = int(event.get('year') or year or 0)
    driver, defender = pick_pair(payload, driver, defender)
    laps = build_actual_laps(payload, driver, defender)
    racing = [row for row in laps if not row['pitDistorted']]
    if not racing:
        return {'error': f'no shared timed laps for {driver} vs {defender}'}
    result = rollout({
        'year': year,
        'policy': policy,
        'laps': racing,
        'startSocMj': 2.8,
        'overtakeActive': False,
    })
    result['actualRace'] = {
        'path': str(path),
        'event': event,
        'session': payload.get('session'),
        'driver': driver,
        'defender': defender,
        'sharedTimedLaps': len(racing),
        'clippedPitLaps': len(laps) - len(racing),
        'dataUsed': [
            'REAL lapTimeS, compound, pit flags from cached FastF1 session JSON',
            'DERIVED gapS = cumulative race-time difference',
            'DERIVED closingRateS = defender lap time minus driver lap time',
            'MODELLED SoC under C5.2 (not team battery)',
        ],
        'notInThisCache': [
            'trap / max speed (so C5.2.8 345 km/h cut rarely applies on this file)',
            'official timing gap to car ahead',
            'measured ES state of charge',
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int)
    parser.add_argument('--round', type=int)
    parser.add_argument('--session', default='Race')
    parser.add_argument('--driver')
    parser.add_argument('--defender')
    parser.add_argument('--policy', default='AUTO')
    parser.add_argument('--first', action='store_true', help='use the earliest cached race on disk')
    args = parser.parse_args()
    year = None if args.first else args.year
    round_number = None if args.first else args.round
    print(json.dumps(clip_cached(year, round_number, args.session, args.driver, args.defender, args.policy)))


if __name__ == '__main__':
    if not sys.stdin.isatty() and not sys.argv[1:]:
        payload = json.loads(sys.stdin.read() or '{}')
        print(json.dumps(clip_cached(
            payload.get('year'), payload.get('round'),
            payload.get('session') or 'Race',
            payload.get('driver'), payload.get('defender'),
            payload.get('policy') or 'AUTO',
        )))
    else:
        main()
