"""Future-blind AI energy rollout for one battle pair.

The frontend is unchanged. This script is the extra backend: given FastF1-like
lap rows (gap, speed, closing), it chooses ATTACK/SAVE/DELAY, updates a
modelled C5.2 battery, and says whether a pass is still legal/affordable
on this lap or later.

It does not drive a real car and does not read team SoC.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from c52_battery import (
    BATTERY_ENGINE_VERSION,
    CITATIONS,
    RECHARGE_CAP_MJ,
    SOC_WINDOW_MJ,
    apply_legal_lap,
    battery_box,
    legal_propulsion_kw,
)
from energy_transition import observed_action

START_SOC_MJ = 2.8
MIN_ATTACK_SOC_MJ = 0.25
MIN_DELAY_SOC_MJ = 0.08

ENGINE_VERSION = 'ai-energy-rollout.v1'


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if number == number and abs(number) != float('inf') else default
    except (TypeError, ValueError):
        return default


def choose_action(row: dict[str, Any], soc_mj: float, policy: str) -> tuple[str, str]:
    """How the AI 'pushes' on this lap.

    AUTO: close + closing + enough SoC -> ATTACK
          close but not enough energy or no speed-legal deploy -> DELAY
          else SAVE (rebuild toward 8.5 MJ lap harvest)
    Explicit ATTACK/SAVE/DELAY in the row or policy overrides AUTO.
    """
    forced = str(row.get('action') or '').upper()
    if forced in {'ATTACK', 'SAVE', 'DELAY'}:
        return forced, 'row-specified action'
    if str(policy).upper() in {'ATTACK', 'SAVE', 'DELAY'}:
        return str(policy).upper(), 'fixed policy for the whole rollout'

    base = observed_action(row)
    gap = _num(row.get('gapS'), 1.2)
    legal_kw = _num((row.get('legal') or {}).get('legalDeployKw'), 1.0)
    if base == 'ATTACK' and soc_mj < MIN_ATTACK_SOC_MJ:
        return 'SAVE', f'wanted ATTACK but modelled SoC {soc_mj:.2f} MJ is below {MIN_ATTACK_SOC_MJ} MJ floor'
    if base == 'DELAY' and soc_mj < MIN_DELAY_SOC_MJ:
        return 'SAVE', f'wanted DELAY but modelled SoC {soc_mj:.2f} MJ is below {MIN_DELAY_SOC_MJ} MJ floor'
    if base == 'ATTACK' and legal_kw <= 0:
        return 'DELAY', 'C5.2.8 speed cut — wait for a slower sector before pushing'
    if gap > 1.2:
        return 'SAVE', 'gap outside the close-battle window; harvest under the 8.5 MJ lap cap'
    return base, f'AUTO from gap {_num(row.get("gapS"), 1.2):.2f}s and closing {_num(row.get("closingRateS")):.2f}s/lap'


def pass_possible(row: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    gap = _num(row.get('gapS'), 9.0)
    closing = _num(row.get('closingRateS'))
    speed_cut = bool(step['legal']['speedCut'])
    enough = step['socEndMj'] >= MIN_ATTACK_SOC_MJ or step['action'] != 'ATTACK'
    immediate = gap <= 0.85 and closing >= 0.12 and not speed_cut and step['consumedMj'] > 0
    later = gap <= 1.2 and not speed_cut and step['socEndMj'] >= MIN_DELAY_SOC_MJ
    if speed_cut:
        verdict = 'NOT THIS STRAIGHT'
        proof = 'C5.2.8 sets legal deploy to 0 kW at this speed; a push cannot create a pass here.'
    elif immediate and enough:
        verdict = 'THIS LAP'
        proof = (
            f'REAL gap {gap:.2f}s and closing {closing:.2f}s/lap; MODELLED spend '
            f'{step["consumedMj"]:.2f} MJ leaving {step["socEndMj"]:.2f}/{SOC_WINDOW_MJ:.0f} MJ.'
        )
    elif later:
        verdict = 'WAIT 1-2 LAPS'
        proof = (
            f'Window is close (gap {gap:.2f}s) but AUTO chose {step["action"]}; '
            f'{step["socEndMj"]:.2f} MJ left after C5.2 clips.'
        )
    else:
        verdict = 'NOT YET'
        proof = f'Gap {gap:.2f}s or energy/speed envelope does not support a durable push.'
    return {'verdict': verdict, 'proof': proof, 'immediate': immediate}


def rollout(payload: dict[str, Any]) -> dict[str, Any]:
    laps = payload.get('laps') or payload.get('rows') or []
    if not isinstance(laps, list) or not laps:
        return {'error': 'laps[] with FastF1-like gap/speed/closing fields is required'}

    policy = str(payload.get('policy') or 'AUTO')
    soc = _num(payload.get('startSocMj'), START_SOC_MJ)
    start = soc
    overtake = bool(payload.get('overtakeActive'))
    era = '2026' if int(_num(payload.get('year'), 2026)) >= 2026 else '2018_2025'
    harvest_used = 0.0
    consumed_total = 0.0
    harvested_total = 0.0
    steps = []

    for raw in laps:
        row = dict(raw or {})
        row['legal'] = legal_propulsion_kw(row.get('speedKph') or row.get('raceMeanSpeedKph'), overtake)
        action, reason = choose_action(row, soc, policy)
        step = apply_legal_lap(
            soc, action, row,
            overtake_active=overtake,
            harvest_used_mj=harvest_used,
            era=era,
        )
        harvest_used = step['harvestUsedAfterLapMj']
        if harvest_used >= RECHARGE_CAP_MJ:
            harvest_used = 0.0  # next lap gets a fresh 8.5 MJ recharge budget
        possible = pass_possible(row, step)
        soc = step['socEndMj']
        consumed_total += step['consumedMj']
        harvested_total += step['harvestedMj']
        steps.append({
            'lap': row.get('lap'),
            'driver': row.get('driver'),
            'defender': row.get('defender'),
            'gapS': row.get('gapS'),
            'speedKph': row['legal']['speedKph'],
            'action': step['action'],
            'whyAiPushed': reason,
            'consumedMj': step['consumedMj'],
            'harvestedMj': step['harvestedMj'],
            'socStartMj': step['socStartMj'],
            'socEndMj': step['socEndMj'],
            'socLeftPct': step['socLeftPct'],
            'legalDeployKw': step['legal']['legalDeployKw'],
            'speedCut': step['legal']['speedCut'],
            'clipReason': step['clipReason'],
            'passWindow': possible,
            'box': battery_box(step['socStartMj'], step['socEndMj'], step['consumedMj'], step['harvestedMj']),
        })

    last = steps[-1]
    call = (
        f"AI {last['action']} on lap {last.get('lap')}: consumed {last['consumedMj']:.2f} MJ, "
        f"{last['socEndMj']:.2f} MJ left in the 4 MJ C5.2 window. "
        f"Pass window: {last['passWindow']['verdict']}. {last['passWindow']['proof']}"
    )
    return {
        'ok': True,
        'engine': ENGINE_VERSION,
        'batteryEngine': BATTERY_ENGINE_VERSION,
        'provenance': 'MODELLED under FIA C5.2; FastF1 supplies gap/speed only; SoC is not team telemetry.',
        'citations': CITATIONS,
        'year': payload.get('year'),
        'policy': policy,
        'overtakeActive': overtake,
        'box': battery_box(start, soc, consumed_total, harvested_total),
        'laps': steps,
        'call': call,
    }


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({'error': 'JSON body required on stdin'}))
        sys.exit(1)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        print(json.dumps({'error': f'invalid JSON: {error}'}))
        sys.exit(1)
    print(json.dumps(rollout(payload)))


if __name__ == '__main__':
    main()
