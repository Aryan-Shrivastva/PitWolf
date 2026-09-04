"""Single versioned energy transition shared by extraction and replay.

The transition is deliberately modelled: public timing and telemetry do not
expose a team's battery state or deployment commands.  Keeping the action
profiles and state update in one module prevents the historical surrogate and
the tactical replay from quietly using different battery rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np


ENERGY_MODEL_VERSION = 'energy-transition.v2'
CAPACITY_MJ = 4.0
ACTION_PROFILE = {
    'ATTACK': {'deploy': 0.88, 'harvest': 0.06},
    'SAVE': {'deploy': 0.16, 'harvest': 0.34},
    'DELAY': {'deploy': 0.43, 'harvest': 0.16},
}

# These are explicit model configurations, not claims of private team battery
# data. The current tactical surrogate keeps the same provisional action
# calibration in both eras while recording the era and its evidence boundary.
ERA_CONFIGS = {
    '2018_2025': {
        'id': 'pu-era-2018-2025-surrogate-v1',
        'capacityMj': 4.0,
        'actionProfile': ACTION_PROFILE,
        'calibrationStatus': 'PROVISIONAL_SHARED_SURROGATE',
        'source': 'MODELLED from public lap timing; team SoC/deployment is unavailable.',
    },
    '2026': {
        'id': 'pu-era-2026-surrogate-v1',
        'capacityMj': 4.0,
        'actionProfile': ACTION_PROFILE,
        'calibrationStatus': 'PROVISIONAL_2026_REGULATION_ENVELOPE',
        'source': 'MODELLED under the configured 2026 usable energy window; event-specific activation data may be unavailable.',
    },
}


def era_for_year(year: Any) -> str:
    try:
        return '2026' if int(year) >= 2026 else '2018_2025'
    except (TypeError, ValueError):
        return '2026'


def get_era_config(era: str | None = None) -> dict[str, Any]:
    key = era if era in ERA_CONFIGS else '2026'
    return ERA_CONFIGS[key]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def clamp_soc(value: Any, capacity_mj: float = CAPACITY_MJ) -> float:
    return float(np.clip(_number(value), 0.0, capacity_mj))


def transition_soc(soc: float, action: str, row: dict[str, Any] | None = None,
                   defending: bool = False, era: str = '2026') -> tuple[float, float, float]:
    """Apply one tactical action and return (next_soc, deploy, harvest)."""
    config = get_era_config(era)
    profile = config['actionProfile'].get(action, config['actionProfile']['SAVE'])
    capacity_mj = float(config['capacityMj'])
    row = row or {}
    tyre_load = min(0.18, abs(_number(row.get('tyreAgeDiff'))) / 100.0)
    tyre_key = 'defenderTyreDegProxy' if defending else 'attackerTyreDegProxy'
    tyre_load += 0.08 * float(np.clip(_number(row.get(tyre_key)), 0.0, 1.0))
    pressure = min(0.16, abs(_number(row.get('closingRateS'))) / 4.0)
    pace = float(np.clip(_number(row.get('pace'), 0.5), 0.1, 0.9))
    desired_deploy = profile['deploy'] + (pressure if defending else 0.0) + tyre_load + (0.10 * pace)
    harvest = profile['harvest'] + (0.04 if not defending and action == 'SAVE' else 0.0) + (0.04 * (1.0 - pace))
    # A depleted car cannot spend a notional deployment budget it does not
    # have. Harvest may support a small same-lap amount, but the reported
    # deployment is clipped to the energy actually available.
    available = max(0.0, _number(soc)) + harvest
    deploy = min(desired_deploy, available)
    return clamp_soc(_number(soc) - deploy + harvest, capacity_mj), round(deploy, 3), round(harvest, 3)


def observed_action(row: dict[str, Any] | None) -> str:
    """Choose a feature-only surrogate mode without reading the target label."""
    row = row or {}
    gap = _number(row.get('gapS'), 1.2)
    closing = _number(row.get('closingRateS'))
    if gap <= 0.85 and closing >= 0.12:
        return 'ATTACK'
    if gap <= 1.2:
        return 'DELAY'
    return 'SAVE'
