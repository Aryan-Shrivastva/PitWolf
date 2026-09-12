"""FIA 2026 energy-compliance context.

This module separates universally published C5 ceilings from values which the
FIA supplies per Competition under Sporting Regulation B7.2.  An absent event
entry is deliberately *not* treated as a legal event configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import fia_2026_regs as regs

ROOT = Path(__file__).resolve().parents[1]
EVENT_LIMITS_PATH = ROOT / 'data' / 'fia-energy-event-limits.json'


def session_kind(session_name: str | None) -> str:
    name = (session_name or '').strip().lower()
    return 'QUALIFYING' if name in {'qualifying', 'sprint qualifying'} else 'RACE_OR_RUNNING'


def _load_registry() -> dict:
    try:
        return json.loads(EVENT_LIMITS_PATH.read_text(encoding='utf8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {'schemaVersion': 'fia-energy-event-limits.v1', 'events': {}}


def _resolve_profile(registry: dict, entry: dict) -> dict:
    """Resolve named, officially digitised curves without duplicating them per GP."""
    curves = registry.get('curves') or {}
    profile = entry.get('powerLimitProfile') or {}
    return {
        key: curves.get(value, value) if isinstance(value, str) else value
        for key, value in profile.items()
    }


def _complete_event_entry(entry: dict, profile: dict, kind: str) -> bool:
    # B7.2.1 requires the FIA Competition information; a source URL alone is
    # not proof that the required values were copied and checked.
    required = ['eventSpecificDataLoaded', 'rechargeLimitsMj', 'powerLimitProfile', 'source']
    if not all(entry.get(key) not in (None, '', False) for key in required):
        return False
    source = entry.get('source') or {}
    if not source.get('url') or not source.get('published'):
        return False
    applied_curve = 'qualifying' if kind == 'QUALIFYING' else 'raceDefault'
    if not isinstance(profile.get(applied_curve), list) or not profile[applied_curve]:
        return False
    recharge = entry.get('rechargeLimitsMj') or {}
    recharge_key = 'qualifying' if kind == 'QUALIFYING' else 'raceOvertakeInactive'
    if recharge.get(recharge_key) in (None, ''):
        return False
    if kind == 'RACE_OR_RUNNING' and not isinstance(entry.get('alternativePowerSectors'), list):
        return False
    return True


def event_context(year: int, round_number: int | None, session_name: str | None) -> dict:
    """Resolve event-specific limits without ever inventing FIA appendix data."""
    kind = session_kind(session_name)
    if int(year) < 2026:
        return {
            'status': 'HISTORICAL_SURROGATE_NOT_2026_COMPLIANCE',
            'eventSpecificDataLoaded': False,
            'sessionKind': kind,
            'rechargeLimitMj': None,
            'powerLimitProfile': {},
            'powerLimitedSectors': [],
            'source': None,
        }
    registry = _load_registry()
    entry = (registry.get('events') or {}).get(f'{year}:{round_number}', {})
    profile = _resolve_profile(registry, entry)
    complete = _complete_event_entry(entry, profile, kind)
    return {
        'status': 'FIA_EVENT_LIMITS_LOADED' if complete else 'FIA_GLOBAL_LIMITS_ONLY',
        'eventSpecificDataLoaded': complete,
        'sessionKind': kind,
        # B7.2 documents publish separate caps by session and, for races,
        # whether Overtake is active.  The single resolved value prevents a
        # qualifying lap from accidentally consuming a race allowance.
        'rechargeLimitMj': float((entry.get('rechargeLimitsMj') or {}).get(
            'qualifying' if kind == 'QUALIFYING' else 'raceOvertakeInactive'
        )) if complete else None,
        'rechargeLimitsMj': entry.get('rechargeLimitsMj') if complete else {},
        'powerLimitProfile': {
            'normal': profile.get(
                'qualifying' if kind == 'QUALIFYING' else 'raceDefault', []
            ),
            'overtake': profile.get('raceOvertake', []),
            'powerLimited': profile.get('alternativeRace', []),
        } if complete else {},
        'alternativePowerSectors': entry.get('alternativePowerSectors', []) if complete else [],
        'overtake': entry.get('overtake') if complete else None,
        'source': entry.get('source') if complete else None,
        'missing': [] if complete else [
            f'official B7.2 {"qualifying" if kind == "QUALIFYING" else "race"} recharge limit',
            f'official B7.2 {"qualifying" if kind == "QUALIFYING" else "race"} power-limit profile',
            'official event source and publication date',
            *([] if kind == 'QUALIFYING' else ['official alternative-power-sector list']),
        ],
    }


def interpolate_power_curve(points: list[dict], speed_kph: float) -> float | None:
    """Interpolate a digitised FIA event curve, if a complete entry provides one."""
    parsed = sorted(
        ({'speed': float(p['speedKph']), 'power': float(p['maxKw'])} for p in points),
        key=lambda point: point['speed'],
    )
    if not parsed:
        return None
    if speed_kph <= parsed[0]['speed']:
        return parsed[0]['power']
    if speed_kph >= parsed[-1]['speed']:
        return parsed[-1]['power']
    for low, high in zip(parsed, parsed[1:]):
        if low['speed'] <= speed_kph <= high['speed']:
            ratio = (speed_kph - low['speed']) / max(1e-9, high['speed'] - low['speed'])
            return low['power'] + ratio * (high['power'] - low['power'])
    return None


def propulsion_ceiling_kw(speed_kph: float, context: dict, *, overtake: bool = False, power_limited: bool = False) -> float:
    profile = context.get('powerLimitProfile') or {}
    key = 'powerLimited' if power_limited else ('overtake' if overtake else 'normal')
    points = profile.get(key)
    if isinstance(points, list) and points:
        curve_value = interpolate_power_curve(points, speed_kph)
        if curve_value is not None:
            return max(0.0, curve_value)
    # The published C5.2.8 default is a legal ceiling, but not an assertion
    # that the FIA did not publish a lower event-specific curve.
    return regs.propulsion_envelope_kw(speed_kph, override=overtake)


def recharge_ceiling_mj(context: dict) -> float:
    return float(context['rechargeLimitMj']) if context.get('eventSpecificDataLoaded') else regs.CONSTANTS['harvest_max_mj_per_lap']['value']


def regulation_payload(context: dict) -> dict:
    return {
        'status': context['status'],
        'eventSpecificLimitsLoaded': context['eventSpecificDataLoaded'],
        'sessionKind': context['sessionKind'],
        'rechargeLimitMj': context.get('rechargeLimitMj'),
        'rechargeLimitsMj': context.get('rechargeLimitsMj', {}),
        'overtake': context.get('overtake'),
        'source': context.get('source'),
        'missing': context.get('missing', []),
        'note': (
            'Global C5 limits plus the official B7.2 Competition configuration are applied.'
            if context['eventSpecificDataLoaded'] else
            'Global C5 ceilings are applied. This is not a Competition-specific FIA compliance claim until the official B7.2 event limits are imported.'
        ),
    }
