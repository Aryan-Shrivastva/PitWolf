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
from energy_transition import CAPACITY_MJ, era_for_year, transition_soc
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


def build_field_story(payload: dict) -> dict:
    """Running order, overtakes, and slow laps from cached FastF1 times only.

    No team battery is present in this cache. Do not invent SoC.
    """
    by_driver: dict[str, dict[int, dict]] = {}
    for row in payload.get('laps') or []:
        driver = row.get('driver')
        lap = row.get('lapNumber')
        if not driver or lap is None or row.get('lapTimeS') is None:
            continue
        by_driver.setdefault(driver, {})[int(lap)] = row

    classified = {
        item['abbr']: item
        for item in (payload.get('drivers') or [])
        if item.get('abbr')
    }
    race_time = {driver: 0.0 for driver in by_driver}
    traces: dict[str, list[dict]] = {driver: [] for driver in by_driver}
    all_laps = sorted({lap for laps in by_driver.values() for lap in laps})

    for lap in all_laps:
        present = [driver for driver in by_driver if lap in by_driver[driver]]
        if not present:
            continue
        for driver in present:
            race_time[driver] += float(by_driver[driver][lap]['lapTimeS'])
        order = sorted((driver for driver in race_time if race_time[driver] > 0), key=lambda driver: race_time[driver])
        for index, driver in enumerate(order):
            if lap not in by_driver[driver]:
                continue
            row = by_driver[driver][lap]
            lap_s = float(row['lapTimeS'])
            ahead = order[index - 1] if index else None
            behind = order[index + 1] if index + 1 < len(order) else None
            gap = 0.0 if ahead is None else round(race_time[driver] - race_time[ahead], 3)
            gap_behind = 0.0 if behind is None else round(race_time[behind] - race_time[driver], 3)
            closing = 0.0
            if ahead and lap in by_driver.get(ahead, {}):
                closing = round(float(by_driver[ahead][lap]['lapTimeS']) - lap_s, 3)
            traces[driver].append({
                'lap': lap,
                'lapTimeS': lap_s,
                'compound': row.get('compound'),
                'isPitLap': bool(row.get('isPitLap')),
                'timingPosition': index + 1,
                'gapToAheadS': gap,
                'gapToBehindS': gap_behind,
                'closingRateS': closing,
                'ahead': ahead,
                'behind': behind,
            })

    drivers_out = []
    for item in sorted(classified.values(), key=lambda row: row.get('position') or 99):
        code = item['abbr']
        laps = traces.get(code) or []
        if not laps:
            continue
        prev_pos = None
        prev_time = None
        for step in laps:
            pos = step['timingPosition']
            pos_change = 0 if prev_pos is None else prev_pos - pos
            delta = None if prev_time is None else round(step['lapTimeS'] - prev_time, 3)
            if step['isPitLap']:
                event = 'PIT'
            elif pos_change > 0:
                event = 'OVERTAKE'
            elif delta is not None and (delta >= 0.25 or (step['closingRateS'] <= -0.2 and step['gapToAheadS'] > 0)):
                event = 'SLOW'
            else:
                event = 'HOLD'
            step['event'] = event
            step['posChange'] = pos_change
            step['deltaToPrevS'] = delta
            prev_pos = pos
            if not step['isPitLap']:
                prev_time = step['lapTimeS']

        era = era_for_year((payload.get('event') or {}).get('year'))
        soc = float(CAPACITY_MJ)
        used = 0.0
        for step in laps:
            if step['event'] == 'OVERTAKE' or step['closingRateS'] >= 0.15:
                action = 'ATTACK'
            elif step['event'] == 'SLOW' or step['isPitLap']:
                action = 'SAVE'
            else:
                action = 'DELAY'
            start = soc
            soc, deploy, harvest = transition_soc(
                soc, action,
                {'gapS': step['gapToAheadS'], 'closingRateS': step['closingRateS'], 'pace': 0.55},
                era=era,
            )
            used += deploy
            step['modelled'] = {
                'label': 'MODELLED',
                'action': action,
                'startPct': round((start / CAPACITY_MJ) * 100.0, 1),
                'endPct': round((soc / CAPACITY_MJ) * 100.0, 1),
                'consumedMj': deploy,
                'harvestedMj': harvest,
                'usedMj': round(used, 3),
                'note': 'Started at 100% of the 4 MJ window. Not team telemetry.',
            }

        overtake_laps = [step['lap'] for step in laps if step['event'] == 'OVERTAKE']
        slow_laps = [step['lap'] for step in laps if step['event'] == 'SLOW']
        after_pass_slow = 0
        for step in laps:
            if step['event'] != 'OVERTAKE':
                continue
            nxt = next((row for row in laps if row['lap'] == step['lap'] + 1), None)
            if nxt and (nxt['event'] == 'SLOW' or (nxt['deltaToPrevS'] or 0) > 0.15):
                after_pass_slow += 1
        drivers_out.append({
            'driver': code,
            'name': item.get('name'),
            'classifiedPosition': item.get('position'),
            'overtakeLaps': overtake_laps,
            'slowLaps': slow_laps,
            'overtakes': len(overtake_laps),
            'slows': len(slow_laps),
            'afterPassNextLapSlower': after_pass_slow,
            'laps': laps,
        })
    return {
        'provenance': 'REAL lap times, order, gaps · MODELLED ES starts at 100% of the 4 MJ window and is not team battery',
        'drivers': drivers_out,
    }


def clip_cached(year: int | None, round_number: int | None, session_name: str,
                driver: str | None, defender: str | None, policy: str,
                start_soc_mj: float | None = None) -> dict:
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
    # Lights-out is a full 4 MJ C5.2 window (100%). The model then
    # consumes and harvests from actual timed laps — not team SoC.
    start = 4.0 if start_soc_mj is None else float(start_soc_mj)
    result = rollout({
        'year': year,
        'policy': policy,
        'laps': racing,
        'startSocMj': start,
        'overtakeActive': False,
    })
    result['field'] = build_field_story(payload)
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
    parser.add_argument('--start-soc', type=float, default=4.0, help='modelled lights-out SoC in MJ; 4.0 is a full C5.2 window')
    parser.add_argument('--first', action='store_true', help='use the earliest cached race on disk')
    args = parser.parse_args()
    year = None if args.first else args.year
    round_number = None if args.first else args.round
    print(json.dumps(clip_cached(year, round_number, args.session, args.driver, args.defender, args.policy, args.start_soc)))


if __name__ == '__main__':
    if not sys.stdin.isatty() and not sys.argv[1:]:
        payload = json.loads(sys.stdin.read() or '{}')
        print(json.dumps(clip_cached(
            payload.get('year'), payload.get('round'),
            payload.get('session') or 'Race',
            payload.get('driver'), payload.get('defender'),
            payload.get('policy') or 'AUTO',
            payload.get('startSocMj'),
        )))
    else:
        main()
