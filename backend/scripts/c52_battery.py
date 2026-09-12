"""C5.2-constrained modelled battery box.

FastF1 does not publish team ES state of charge. This module only applies the
published 2026 technical envelope to a modelled 0-4 MJ window so a later UI
can show consumed / remaining without claiming measured telemetry.

Citations are 2026 FIA F1 Regulations Section C [Technical], Issue 20:
C5.2.7  ERS-K DC power <= 350 kW
C5.2.8  propulsion power vs speed (Overtake on/off)
C5.2.9  ES max-min SoC <= 4 MJ on track
C5.2.10 Recharge <= 8.5 MJ per lap
"""

from __future__ import annotations

from typing import Any

from energy_transition import CAPACITY_MJ, clamp_soc, transition_soc
from fia_2026_regs import CONSTANTS, TECHNICAL_C, propulsion_envelope_kw

BATTERY_ENGINE_VERSION = 'c52-battery.v1'
ERS_K_MAX_KW = float(CONSTANTS['ers_k_dc_power_max_kw']['value'])
RECHARGE_CAP_MJ = float(CONSTANTS['harvest_max_mj_per_lap']['value'])
SOC_WINDOW_MJ = float(CONSTANTS['es_soc_window_mj']['value'])

CITATIONS = [
    {'article': 'C5.2.7', 'document': TECHNICAL_C, 'rule': 'ERS-K DC power <= 350 kW'},
    {'article': 'C5.2.8', 'document': TECHNICAL_C, 'rule': 'Propulsion power falls with speed; 0 kW at/above 345 km/h (355 with Overtake)'},
    {'article': 'C5.2.9', 'document': TECHNICAL_C, 'rule': 'ES max minus min SoC <= 4 MJ on track'},
    {'article': 'C5.2.10', 'document': TECHNICAL_C, 'rule': 'Recharge <= 8.5 MJ per lap at the CU-K HV DC bus'},
]


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if number == number and abs(number) != float('inf') else default
    except (TypeError, ValueError):
        return default


def legal_propulsion_kw(speed_kph: Any, overtake_active: bool = False) -> dict[str, Any]:
    speed = max(0.0, _num(speed_kph))
    envelope = propulsion_envelope_kw(speed, override=bool(overtake_active))
    allowed = min(ERS_K_MAX_KW, envelope)
    cut = allowed <= 0.0
    return {
        'speedKph': round(speed, 1),
        'overtakeActive': bool(overtake_active),
        'ersKCapKw': ERS_K_MAX_KW,
        'speedEnvelopeKw': round(envelope, 1),
        'legalDeployKw': round(allowed, 1),
        'speedCut': cut,
        'citation': 'C5.2.7 + C5.2.8',
        'note': (
            'Electric propulsion is legally zero at this speed.'
            if cut else
            f'Legal ERS-K propulsion ceiling is {allowed:.0f} kW at {speed:.0f} km/h.'
        ),
    }


def apply_legal_lap(
    soc_mj: float,
    action: str,
    row: dict[str, Any] | None = None,
    *,
    overtake_active: bool = False,
    harvest_used_mj: float = 0.0,
    defending: bool = False,
    era: str = '2026',
) -> dict[str, Any]:
    """One modelled lap: requested ATTACK/SAVE/DELAY, then C5.2 clips."""
    row = dict(row or {})
    start = clamp_soc(soc_mj, SOC_WINDOW_MJ)
    legal = legal_propulsion_kw(row.get('speedKph') or row.get('raceMeanSpeedKph'), overtake_active)
    next_soc, deploy, harvest = transition_soc(
        start, action, row, defending=defending, era=era,
    )

    if legal['speedCut']:
        # C5.2.8: no propulsion credit; harvest may still refill toward 8.5 MJ.
        deploy = 0.0
        next_soc = clamp_soc(start + harvest, SOC_WINDOW_MJ)
        clip_reason = 'C5.2.8 speed cut — deploy forced to 0 kW'
    else:
        scale = legal['legalDeployKw'] / ERS_K_MAX_KW if ERS_K_MAX_KW else 0.0
        legal_deploy = round(min(deploy, deploy * scale if scale < 1 else deploy), 3)
        if legal_deploy < deploy:
            next_soc = clamp_soc(start - legal_deploy + harvest, SOC_WINDOW_MJ)
            deploy = legal_deploy
            clip_reason = 'C5.2.7/C5.2.8 power envelope reduced requested deploy'
        else:
            clip_reason = None

    recharge_left = max(0.0, RECHARGE_CAP_MJ - max(0.0, harvest_used_mj))
    if harvest > recharge_left:
        next_soc = clamp_soc(next_soc - (harvest - recharge_left), SOC_WINDOW_MJ)
        harvest = round(recharge_left, 3)
        clip_reason = (clip_reason + ' · ' if clip_reason else '') + 'C5.2.10 lap recharge cap 8.5 MJ'

    consumed = round(deploy, 3)
    harvested = round(harvest, 3)
    end = clamp_soc(next_soc, SOC_WINDOW_MJ)
    return {
        'action': action,
        'socStartMj': round(start, 3),
        'consumedMj': consumed,
        'harvestedMj': harvested,
        'socEndMj': round(end, 3),
        'socLeftPct': round((end / SOC_WINDOW_MJ) * 100.0, 1),
        'harvestUsedAfterLapMj': round(harvest_used_mj + harvested, 3),
        'legal': legal,
        'clipReason': clip_reason,
        'capacityMj': SOC_WINDOW_MJ,
        'provenance': 'MODELLED',
    }


def battery_box(start_mj: float, end_mj: float, consumed_mj: float, harvested_mj: float) -> dict[str, Any]:
    return {
        'label': 'MODELLED ES WINDOW',
        'capacityMj': SOC_WINDOW_MJ,
        'startMj': round(_num(start_mj), 3),
        'endMj': round(_num(end_mj), 3),
        'consumedMj': round(_num(consumed_mj), 3),
        'harvestedMj': round(_num(harvested_mj), 3),
        'leftMj': round(clamp_soc(end_mj, SOC_WINDOW_MJ), 3),
        'leftPct': round((clamp_soc(end_mj, SOC_WINDOW_MJ) / SOC_WINDOW_MJ) * 100.0, 1),
        'citation': 'C5.2.9 usable window 4 MJ · C5.2.10 recharge 8.5 MJ/lap',
        'notTeamTelemetry': True,
    }
