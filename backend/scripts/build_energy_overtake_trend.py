"""Build leftover / consume trends at DRS convert vs miss.

Walks cached 2018–2025 races only. 2014 is not in the FastF1 session store.
The JSON is a historical trend, not a draw and not team battery telemetry.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from clip_cached_race import build_field_story
from drs_recipe import HISTORY_YEARS, extract_windows
from train_driver_recommend import iter_races

OUT = Path(__file__).resolve().parents[1] / 'data' / 'f1-cache' / 'models' / 'energy_overtake_trend.json'
HIGH_LEFT = 70.0
CHASE_GAP_S = 1.2


def _median(values) -> float | None:
    clean = [float(item) for item in values if item is not None]
    if not clean:
        return None
    return round(float(np.median(clean)), 2)


def _rate(wins: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(wins / total, 3)


def _bucket() -> dict:
    return {
        'windows': 0,
        'converts': 0,
        'leftConvert': [],
        'leftMiss': [],
        'earlyLeftConvert': [],
        'earlyLeftMiss': [],
        'consumeConvert': [],
        'consumeMiss': [],
        'highLeftWindows': 0,
        'highLeftConverts': 0,
        'lowLeftWindows': 0,
        'lowLeftConverts': 0,
        'chaseWindows': 0,
        'chaseConverts': 0,
    }


def _record(store: dict, leftover, consume, converted: bool, chased: bool, early: bool) -> None:
    store['windows'] += 1
    store['converts'] += int(converted)
    if leftover is not None:
        if converted:
            store['leftConvert'].append(leftover)
        else:
            store['leftMiss'].append(leftover)
        if early:
            if converted:
                store['earlyLeftConvert'].append(leftover)
            else:
                store['earlyLeftMiss'].append(leftover)
        if leftover >= HIGH_LEFT:
            store['highLeftWindows'] += 1
            store['highLeftConverts'] += int(converted)
        else:
            store['lowLeftWindows'] += 1
            store['lowLeftConverts'] += int(converted)
    if consume is not None:
        if converted:
            store['consumeConvert'].append(consume)
        else:
            store['consumeMiss'].append(consume)
    if chased:
        store['chaseWindows'] += 1
        store['chaseConverts'] += int(converted)


def _pack(store: dict) -> dict:
    return {
        'windows': store['windows'],
        'converts': store['converts'],
        'efficiency': _rate(store['converts'], store['windows']),
        'medianLeftPctConvert': _median(store['leftConvert']),
        'medianLeftPctMiss': _median(store['leftMiss']),
        'medianLeftPctEarlyConvert': _median(store['earlyLeftConvert']),
        'medianLeftPctEarlyMiss': _median(store['earlyLeftMiss']),
        'medianConsumeMjConvert': _median(store['consumeConvert']),
        'medianConsumeMjMiss': _median(store['consumeMiss']),
        'highLeftConvertRate': _rate(store['highLeftConverts'], store['highLeftWindows']),
        'lowLeftConvertRate': _rate(store['lowLeftConverts'], store['lowLeftWindows']),
        'highLeftWindows': store['highLeftWindows'],
        'lowLeftWindows': store['lowLeftWindows'],
        'chaseConvertRate': _rate(store['chaseConverts'], store['chaseWindows']),
        'chaseWindows': store['chaseWindows'],
    }


def build_trend() -> dict:
    global_store = _bucket()
    drivers = defaultdict(_bucket)
    tracks = defaultdict(_bucket)
    years = set()
    races = 0
    for (year, _round), payload in iter_races():
        if year not in HISTORY_YEARS:
            continue
        event = payload.get('event') or {}
        field = build_field_story(payload)
        location = str(event.get('location') or event.get('name') or '')
        windows = extract_windows(field, event)
        if not windows:
            continue
        races += 1
        years.add(year)
        laps_by_driver = {
            row['driver']: {step['lap']: step for step in (row.get('laps') or [])}
            for row in field.get('drivers') or []
        }
        for window in windows:
            start = (laps_by_driver.get(window['driver']) or {}).get(window['startLap']) or {}
            modelled = start.get('modelled') or {}
            leftover = modelled.get('startPct')
            consume = modelled.get('consumedMj')
            gap_behind = start.get('gapToBehindS')
            chased = bool(start.get('behind') and gap_behind is not None and float(gap_behind) <= CHASE_GAP_S)
            converted = bool(window.get('converted'))
            early = int(window.get('startLap') or 0) <= 15
            _record(global_store, leftover, consume, converted, chased, early)
            _record(drivers[window['driver']], leftover, consume, converted, chased, early)
            if location:
                _record(tracks[location], leftover, consume, converted, chased, early)
    packed_years = sorted(years)
    return {
        'ok': True,
        'schema': 'energy-overtake-trend.v1',
        'years': f'{packed_years[0]}-{packed_years[-1]}' if packed_years else None,
        'yearList': packed_years,
        'races': races,
        'note': 'FastF1 session cache starts in 2018. 2014 is not present. Trend is leftover/consume at 1.0s DRS windows, modelled ES only.',
        'global': _pack(global_store),
        'drivers': {code: _pack(store) for code, store in sorted(drivers.items()) if store['windows'] >= 4},
        'tracks': {name: _pack(store) for name, store in sorted(tracks.items()) if store['windows'] >= 4},
        'labels': {
            'leftover': 'MODELLED end-of-lap leftover of the 4 MJ window at DRS entry',
            'consume': 'MODELLED deploy on the DRS-entry lap, not team battery',
            'drs': 'Timing gap ≤ 1.0s, not official DRS telemetry',
        },
    }


if __name__ == '__main__':
    payload = build_trend()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload), encoding='utf-8')
    print(json.dumps({
        'wrote': str(OUT),
        'years': payload.get('years'),
        'races': payload.get('races'),
        'windows': (payload.get('global') or {}).get('windows'),
        'efficiency': (payload.get('global') or {}).get('efficiency'),
    }))
