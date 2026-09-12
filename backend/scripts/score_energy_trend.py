"""Energy + recommendation story for one selected driver.

Per-lap modelled ES from the cached field story. DRS / chase / overtake marks
come from timing. Counterfactual ATTACK uses the 2018–2025 forests already
trained — leftover after one extra ATTACK vs DELAY deploy, not a random draw.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

from clip_cached_race import build_field_story, clip_cached, load_session, session_path
from drs_recipe import DRS_GAP_S, extract_windows, in_drs, laps_already_in_drs
from energy_transition import ACTION_PROFILE, CAPACITY_MJ
from score_recommend import MODELS, score

TREND = Path(__file__).resolve().parents[1] / 'data' / 'f1-cache' / 'models' / 'energy_overtake_trend.json'
CHASE_GAP_S = 1.2
ATTACK_EXTRA_MJ = round(
    float(ACTION_PROFILE['ATTACK']['deploy']) - float(ACTION_PROFILE['DELAY']['deploy']),
    3,
)
SAVE_KEEP_MJ = round(
    float(ACTION_PROFILE['DELAY']['deploy']) - float(ACTION_PROFILE['SAVE']['deploy']),
    3,
)
HIGH_LEFT = 70.0


def _num(value, default=None):
    try:
        number = float(value)
        return number if number == number else default
    except (TypeError, ValueError):
        return default


def _being_chased(step: dict) -> bool:
    if not step.get('behind') or step.get('isPitLap'):
        return False
    gap = _num(step.get('gapToBehindS'))
    return gap is not None and gap <= CHASE_GAP_S


def _battle_payload(row: dict, event: dict, step: dict, leftover_pct=None) -> dict:
    last = max(1, step.get('lap') or 1)
    modelled = step.get('modelled') or {}
    held = laps_already_in_drs(row.get('laps') or [], step.get('lap') or 1)
    return {
        'year': event.get('year'),
        'location': event.get('location'),
        'driver': row['driver'],
        'ahead': step.get('ahead'),
        'behind': step.get('behind'),
        'lap': step.get('lap'),
        'gapS': step.get('gapToAheadS'),
        'closingRateS': step.get('closingRateS'),
        'position': step.get('timingPosition'),
        'lapFraction': (step.get('lap') or 1) / last,
        'deltaToPrevS': step.get('deltaToPrevS'),
        'modelledLeftPct': leftover_pct if leftover_pct is not None else modelled.get('endPct'),
        'modelledUsedMj': modelled.get('usedMj'),
        'lapsInDrs': held,
        'behindClose': _being_chased(step),
    }


def _window_step(laps: list[dict], start_lap: int) -> dict:
    return next((step for step in laps if step.get('lap') == start_lap), laps[0] if laps else {})


def _driver_row(field: dict | None, code: str | None) -> dict | None:
    if not field or not code:
        return None
    return next((item for item in (field.get('drivers') or []) if item.get('driver') == code), None)


def _leftover_of(step: dict | None):
    modelled = (step or {}).get('modelled') or {}
    return _num(modelled.get('startPct'), modelled.get('endPct'))


def _opponent_odds(field, event, step: dict, artifact, hunter: str) -> dict | None:
    """Forest view from the car ahead: will they hold or yield if hunted."""
    code = step.get('ahead')
    opp = _driver_row(field, code)
    if not opp or not artifact:
        return None
    opp_step = _window_step(opp.get('laps') or [], step.get('lap'))
    if not opp_step:
        return None
    leftover = _leftover_of(opp_step)
    scored = score({
        'year': event.get('year'),
        'location': event.get('location'),
        'driver': code,
        'ahead': opp_step.get('ahead'),
        'behind': hunter,
        'lap': step.get('lap'),
        'gapS': step.get('gapToAheadS'),
        'closingRateS': step.get('closingRateS'),
        'position': opp_step.get('timingPosition'),
        'lapFraction': (step.get('lap') or 1) / max(1, step.get('lap') or 1),
        'deltaToPrevS': opp_step.get('deltaToPrevS'),
        'modelledLeftPct': leftover,
        'modelledUsedMj': (opp_step.get('modelled') or {}).get('usedMj'),
        'lapsInDrs': 0,
        'behindClose': True,
    }, artifact=artifact)
    p_yield = _num((scored.get('outcome') or {}).get('pLoseSoon'))
    p_attack = _num((scored.get('probabilities') or {}).get('pushHelps'))
    they = _map_action((opp_step.get('modelled') or {}).get('action'))
    call = _map_action(scored.get('action'))
    return {
        'driver': code,
        'leftPct': leftover,
        'theyDid': they,
        'call': call,
        'pYield': None if p_yield is None else round(float(p_yield), 3),
        'pHold': None if p_yield is None else round(max(0.0, min(1.0, 1.0 - float(p_yield))), 3),
        'pAttack': None if p_attack is None else round(float(p_attack), 3),
        'pSave': None if p_attack is None else round(max(0.0, min(1.0, 1.0 - float(p_attack))), 3),
    }


def _net_pass(p_pass, opponent: dict | None) -> float | None:
    if p_pass is None:
        return None
    hold = 0.0 if not opponent or opponent.get('pHold') is None else float(opponent['pHold'])
    net = float(p_pass) * (1.0 - 0.5 * hold)
    if opponent and opponent.get('theyDid') == 'ATTACK' and (opponent.get('leftPct') or 0) >= 40:
        net *= 0.85
    return round(max(0.0, min(1.0, net)), 3)


def _counterfactual(row: dict, event: dict, step: dict, artifact, trend_slice: dict, converted: bool, windows=None, field=None) -> dict:
    modelled = step.get('modelled') or {}
    leftover = _num(modelled.get('startPct'), modelled.get('endPct') or 100.0)
    extra_pct = (ATTACK_EXTRA_MJ / CAPACITY_MJ) * 100.0
    push_left = max(0.0, leftover - extra_pct)
    clips = leftover < extra_pct + 8
    choices = simulate_choices(row, event, step, artifact, trend_slice, windows, field) if artifact else None
    hold = None if not choices else choices.get('hold')
    attack = None if not choices else choices.get('attack')
    actual_p2 = None if not hold else hold.get('convertIn2')
    push_p2 = None if not attack else attack.get('pGain')
    hold_gain = None if not hold else hold.get('pGain')
    lift = None if choices is None else choices.get('convertLift')
    high_rate = trend_slice.get('highLeftConvertRate')
    low_rate = trend_slice.get('lowLeftConvertRate')
    bucket = 'high' if leftover >= HIGH_LEFT else 'low'
    early_convert = trend_slice.get('medianLeftPctEarlyConvert')
    if converted:
        verdict = 'TAKEN'
        note = (
            f'They already passed here with {leftover:.0f}% of the 4 MJ energy store still unused'
            f'{f" (their early-race 2018–2025 passes typically had about {early_convert:.0f}% left)" if isinstance(early_convert, (int, float)) else ""}.'
        )
    elif clips:
        verdict = 'NO HEADROOM'
        note = (
            f'Only {leftover:.0f}% of the 4 MJ energy store is left, so an extra {ATTACK_EXTRA_MJ:.2f} MJ ATTACK would empty it.'
        )
    elif leftover >= HIGH_LEFT and lift is not None and lift >= 0.04:
        verdict = 'SPEND HERE'
        note = (
            f'Spending {ATTACK_EXTRA_MJ:.2f} MJ extra here ({leftover:.0f}% left) raises the chance of passing in the next 2 laps from '
            f'{hold_gain:.0%} to {push_p2:.0%}, so this fight’s place moves '
            f'{_fmt_pos(None if not hold else hold.get("afterPosition"))} → '
            f'{_fmt_pos(None if not attack else attack.get("afterPosition"))}.'
        )
    elif leftover >= HIGH_LEFT:
        verdict = 'HEADROOM UNUSED'
        note = (
            f'They had {leftover:.0f}% energy left and did not pass; in 2018–2025, a 1.0s window with ≥70% energy left became a pass '
            f'{f"{high_rate:.0%}" if high_rate is not None else "—"} of the time'
            f'{f" (already-spent windows: {low_rate:.0%})" if low_rate is not None else ""}.'
        )
    else:
        verdict = 'ALREADY SPENT'
        note = (
            f'Only {leftover:.0f}% of the 4 MJ energy store remains, so extra ATTACK here is spending after the cheap window is gone.'
        )
    return {
        'actualLeftPct': leftover,
        'pushLeftPct': None if clips else round(push_left, 1),
        'extraMj': ATTACK_EXTRA_MJ,
        'clips': clips,
        'actualConvertIn2': actual_p2,
        'pushConvertIn2': push_p2,
        'convertLift': lift,
        'actualAction': None if not hold else hold.get('action'),
        'pushAction': None if not attack else 'ATTACK',
        'actualPushHelps': None if not hold else hold.get('pushHelps'),
        'pushPushHelps': None if not attack else attack.get('pushHelps'),
        'leftoverBand': bucket,
        'verdict': verdict,
        'note': note,
        'afterIfAttack': None if not attack else attack.get('afterPosition'),
        'afterIfHold': None if not hold else hold.get('afterPosition'),
        'endIfAttack': None if not choices else choices.get('raceEndIfAttack'),
        'endIfHold': None if not choices else choices.get('raceEndIfHold'),
        'observedFinish': None if not choices else choices.get('observedFinish'),
        'avoidedLose': None if not choices else choices.get('avoidedLose'),
        'call': None if not choices else choices.get('call'),
        'raceEndIfAttack': None if not choices else choices.get('raceEndIfAttack'),
        'raceEndIfHold': None if not choices else choices.get('raceEndIfHold'),
        'takesIfAttack': None if not choices else (choices.get('attack') or {}).get('takes'),
        'lines': [],
    }


def _choice_row(scored: dict, leftover, extra_mj=0.0) -> dict:
    drs = scored.get('drs') or {}
    probs = scored.get('probabilities') or {}
    outcome = scored.get('outcome') or {}
    return {
        'leftPct': None if leftover is None else round(float(leftover), 1),
        'extraMj': extra_mj,
        'action': scored.get('action'),
        'convertIn1': drs.get('pNext1'),
        'convertIn2': drs.get('pNext2'),
        'pushHelps': probs.get('pushHelps'),
        'recoverIfLost': probs.get('recoverIfLost'),
        'overtakeSoon': probs.get('overtakeSoon'),
        'placesToFlag': outcome.get('placesToFlag'),
        'expectedFinish': outcome.get('expectedFinish'),
        'pLoseSoon': outcome.get('pLoseSoon'),
    }


def _clamp_pos(value) -> float:
    return round(max(1.0, min(20.0, float(value))), 2)


def _map_action(action: str | None) -> str:
    if action == 'ATTACK':
        return 'ATTACK'
    if action == 'SAVE':
        return 'SAVE'
    return 'HOLD'


def _fmt_pos(value) -> str:
    if value is None:
        return '—'
    return f'P{int(round(float(value)))}'


def _fmt_int(value) -> str:
    return _fmt_pos(value)


def _lost_soon(laps: list[dict], lap: int) -> bool:
    for step in laps or []:
        if step.get('isPitLap'):
            continue
        if step.get('lap') in {lap + 1, lap + 2} and (step.get('posChange') or 0) < 0:
            return True
    return False


def _first_extra(call: str, step: dict, leftover, clips: bool, converted=False, lost_soon=False, they_did=None, net_pass=None, opponent=None) -> dict | None:
    """Whole place vs the real race from this fight only."""
    if call == 'ATTACK' and in_drs(step) and not converted and not clips and they_did != 'ATTACK':
        if net_pass is not None and net_pass < 0.30:
            return None
        opp = opponent or {}
        return {
            'lap': step.get('lap'),
            'ahead': step.get('ahead'),
            'kind': 'TAKE',
            'note': f'L{step.get("lap")} take {step.get("ahead") or "the car ahead"}',
            'pPass': net_pass,
            'pOppHold': opp.get('pHold'),
            'oppDid': opp.get('theyDid'),
        }
    if call in ('SAVE', 'ATTACK') and lost_soon and leftover is not None and leftover >= 40:
        return {
            'lap': step.get('lap'),
            'ahead': step.get('behind'),
            'kind': 'HOLD',
            'note': f'L{step.get("lap")} hold {step.get("behind") or "the car behind"}',
        }
    return None


def _later_takes(row: dict, event: dict, start_lap: int, leftover, energy_delta, windows: list[dict] | None, artifact, trend_slice, field=None) -> list[dict]:
    """Later missed 1.0s windows that still have energy and a trained pass chance."""
    extra_pct = (ATTACK_EXTRA_MJ / CAPACITY_MJ) * 100.0
    takes = []
    delta = float(energy_delta or 0)
    last_lap = None
    last_ahead = None
    for window in windows or []:
        lap = window.get('startLap')
        if lap is None or lap <= start_lap or window.get('converted'):
            continue
        if last_lap is not None and window.get('ahead') == last_ahead and lap - last_lap <= 2:
            continue
        win_step = _window_step(row.get('laps') or [], lap)
        modelled = win_step.get('modelled') or {}
        recorded = _num(window.get('leftPct'), _num(modelled.get('startPct'), modelled.get('endPct')))
        if recorded is None:
            recorded = leftover
        adj = None if recorded is None else max(0.0, min(100.0, float(recorded) + delta))
        if adj is not None and adj < extra_pct + 8:
            continue
        step = win_step
        if not step or not in_drs(step):
            continue
        scored = score(_battle_payload(row, event, step, adj), artifact=artifact)
        gain, _lose = _gain_lose(scored, step, adj, 'ATTACK', False, trend_slice)
        opponent = _opponent_odds(field, event, step, artifact, row.get('driver'))
        net = _net_pass(gain, opponent)
        if (net is None or net < 0.30) and (adj is None or adj < HIGH_LEFT):
            continue
        if net is not None and net < 0.22:
            continue
        takes.append({
            'lap': lap,
            'ahead': window.get('ahead') or step.get('ahead'),
            'kind': 'LATER',
            'note': f'L{lap} take {window.get("ahead") or step.get("ahead") or "the car ahead"}',
            'pPass': net,
            'pGain': gain,
            'pOppHold': None if not opponent else opponent.get('pHold'),
            'oppDid': None if not opponent else opponent.get('theyDid'),
            'oppLeftPct': None if not opponent else opponent.get('leftPct'),
        })
        delta -= extra_pct
        last_lap = lap
        last_ahead = window.get('ahead')
    return takes


def _race_end(observed_finish, extras: list[dict]) -> int | None:
    if observed_finish is None:
        return None
    gained = len(extras or [])
    return max(1, min(20, int(observed_finish) - gained))


def _gain_lose(scored: dict, step: dict, leftover, mode: str, clips: bool, trend_slice: dict | None):
    """P(gain a place) and P(lose a place) from this lap's fight only."""
    drs = scored.get('drs') or {}
    probs = scored.get('probabilities') or {}
    outcome = scored.get('outcome') or {}
    p1 = _num(drs.get('pNext1'))
    p2 = _num(drs.get('pNext2'), 0.0) or 0.0
    soon = _num(probs.get('overtakeSoon'), 0.0) or 0.0
    efficiency = _num((trend_slice or {}).get('efficiency'), _num(drs.get('efficiency'), 0.35)) or 0.35
    lose = _num(outcome.get('pLoseSoon'), 0.0) or 0.0
    in_range = in_drs(step)
    chased = _being_chased(step)
    if in_range:
        if mode == 'ATTACK' and not clips and leftover >= 25:
            # Commit the window: pull convert-in-1 forward and blend toward
            # this driver's 2018–2025 DRS efficiency (how often they convert
            # when they stay in a 1.0s window).
            attack_now = p1 if p1 is not None else p2
            commit = 0.55 * float(attack_now) + 0.45 * float(efficiency)
            gain = max(p2, commit)
        elif mode == 'SAVE':
            gain = p2 * 0.75
        else:
            gain = p2
    else:
        gain = soon * (0.35 if mode == 'ATTACK' and leftover >= HIGH_LEFT and not clips else 0.15)
    if chased:
        if mode == 'SAVE':
            lose = max(0.0, lose * (1.0 - 0.22 * min(1.0, leftover / 100.0)))
        elif mode == 'ATTACK' and leftover < HIGH_LEFT:
            lose = min(0.95, lose + 0.05)
    else:
        lose = lose * 0.25
    return round(float(gain), 3), round(float(lose), 3)


def simulate_choices(row: dict, event: dict, step: dict, artifact, trend_slice=None, windows=None, field=None) -> dict | None:
    if not artifact:
        return None
    modelled = step.get('modelled') or {}
    leftover = _num(modelled.get('startPct'), modelled.get('endPct') or 100.0)
    extra_pct = (ATTACK_EXTRA_MJ / CAPACITY_MJ) * 100.0
    save_pct = (SAVE_KEEP_MJ / CAPACITY_MJ) * 100.0
    attack_left = max(0.0, leftover - extra_pct)
    save_left = min(100.0, leftover + save_pct)
    clips = leftover < extra_pct + 8
    consumed = _num(modelled.get('consumedMj'), 0.0) or 0.0
    actual = score(_battle_payload(row, event, step, leftover), artifact=artifact)
    # Leftover barely moves the forest (feature weight ~0.004). ATTACK/SAVE
    # change convert/lose through the disclosed commit/defend rules, not by
    # pretending a different SoC would rewrite the same state vector.
    attack = None if clips else actual
    saved = actual
    actual_p2 = (actual.get('drs') or {}).get('pNext2')
    attack_p1 = None if not attack else (attack.get('drs') or {}).get('pNext1')
    recover = (actual.get('probabilities') or {}).get('recoverIfLost') or 0
    chased = _being_chased(step)
    in_range = in_drs(step)
    hold_gain, hold_lose = _gain_lose(actual, step, leftover, 'HOLD', False, trend_slice)
    attack_gain, attack_lose = (None, None) if not attack else _gain_lose(
        attack, step, leftover, 'ATTACK', clips, trend_slice,
    )
    save_gain, save_lose = _gain_lose(saved, step, leftover, 'SAVE', False, trend_slice)
    lift = None if attack_gain is None else round(float(attack_gain) - float(hold_gain), 3)
    if clips:
        call = 'HOLD'
        why = (
            f'Only {leftover:.0f}% of the 4 MJ energy store is left, so extra ATTACK would empty it — SAVE can still keep ~{SAVE_KEEP_MJ:.2f} MJ for later.'
        )
    elif in_range and leftover >= HIGH_LEFT and lift is not None and lift >= 0.04:
        call = 'ATTACK'
        why = (
            f'Gap is under 1.0s and {leftover:.0f}% of the 4 MJ store is left — spending {ATTACK_EXTRA_MJ:.2f} MJ extra raises the chance of passing in the next 2 laps from {hold_gain:.0%} to {attack_gain:.0%}.'
        )
    elif chased and leftover >= HIGH_LEFT and recover >= 0.45:
        call = 'SAVE' if hold_lose - save_lose >= 0.03 else 'HOLD'
        why = (
            f'Someone is 1.2s or closer behind, and {leftover:.0f}% energy is still left — {call} keeps reserve because historically they take the place back {recover:.0%} of the time after losing it.'
        )
    elif leftover >= HIGH_LEFT and not in_range:
        call = 'SAVE'
        why = (
            f'Nobody is inside 1.0s and {leftover:.0f}% energy remains, so SAVE keeps ~{SAVE_KEEP_MJ:.2f} MJ for the next pass or defend.'
        )
    else:
        call = _map_action(actual.get('action'))
        why = 'In this gap and energy state, spending extra megajoules barely changes the 2018–2025 pass chance.'

    now_pos = _num(step.get('timingPosition'))
    try:
        observed_finish = int(row.get('classifiedPosition') or (row.get('laps') or [{}])[-1].get('timingPosition'))
    except (TypeError, ValueError):
        observed_finish = None if now_pos is None else int(now_pos)

    def pack(scored, left, extra_mj, used, gain, lose):
        if scored is None or gain is None:
            return None
        packed = _choice_row(scored, left, extra_mj)
        packed['pGain'] = gain
        packed['pLose'] = lose
        packed['usedMj'] = round(float(used), 3)
        packed['pLoseSoon'] = lose
        if now_pos is None:
            return packed
        after = _clamp_pos(now_pos - gain + lose)
        packed['afterPosition'] = after
        packed['expectedFinish'] = after
        packed['placesToFlag'] = round(now_pos - after, 2)
        return packed

    hold_row = pack(actual, leftover, 0.0, consumed, hold_gain, hold_lose)
    attack_row = pack(attack, attack_left, ATTACK_EXTRA_MJ, consumed + ATTACK_EXTRA_MJ, attack_gain, attack_lose)
    save_row = pack(saved, save_left, -SAVE_KEEP_MJ, max(0.0, consumed - SAVE_KEEP_MJ), save_gain, save_lose)
    they_did = _map_action(modelled.get('action'))
    did_row = {'ATTACK': attack_row, 'SAVE': save_row, 'HOLD': hold_row}.get(they_did) or hold_row
    did_after = None if not did_row else did_row.get('afterPosition')
    extra_pct = (ATTACK_EXTRA_MJ / CAPACITY_MJ) * 100.0
    save_pct = (SAVE_KEEP_MJ / CAPACITY_MJ) * 100.0
    start_lap = step.get('lap') or 0
    converted_here = any(
        item.get('converted') and item.get('startLap') == start_lap
        for item in (windows or [])
    )
    lost_soon = _lost_soon(row.get('laps') or [], start_lap)
    opponent = _opponent_odds(field, event, step, artifact, row.get('driver'))
    net_now = _net_pass(attack_gain, opponent)
    p_our_attack = None if attack_gain is None else round(float(attack_gain), 3)
    paths = {}
    for name, choice, energy_delta in (
        ('HOLD', hold_row, 0.0),
        ('ATTACK', attack_row, -extra_pct if not clips else 0.0),
        ('SAVE', save_row, save_pct),
    ):
        extras = []
        first = None if not choice else _first_extra(
            name, step, leftover, clips, converted=converted_here, lost_soon=lost_soon,
            they_did=they_did, net_pass=net_now, opponent=opponent,
        )
        if first:
            extras.append(first)
        extras.extend(_later_takes(
            row, event, start_lap, leftover, energy_delta, windows, artifact, trend_slice, field,
        ))
        race_end = _race_end(observed_finish, extras)
        paths[name] = extras
        if choice:
            if observed_finish is not None and did_after is not None and choice.get('afterPosition') is not None:
                choice['endPosition'] = _clamp_pos(observed_finish - (did_after - choice['afterPosition']))
            choice['raceEnd'] = race_end
            choice['takes'] = extras
            choice['extraPlaces'] = len(extras)
    def _end_of(name):
        choice = {'ATTACK': attack_row, 'SAVE': save_row, 'HOLD': hold_row}.get(name)
        return None if not choice else choice.get('raceEnd')
    what_if_call = call
    if they_did == call or _end_of(call) == observed_finish:
        alts = [name for name in ('SAVE', 'ATTACK', 'HOLD') if name != they_did]
        alts.sort(key=lambda name: 99 if _end_of(name) is None else _end_of(name))
        if alts and observed_finish is not None and _end_of(alts[0]) is not None and _end_of(alts[0]) < observed_finish:
            what_if_call = alts[0]
    picked = {'ATTACK': attack_row, 'SAVE': save_row, 'HOLD': hold_row}.get(what_if_call) or hold_row
    hold_end = _end_of('HOLD')
    pick_end = None if not picked else picked.get('raceEnd')
    hold_lose_p = None if not hold_row else hold_row.get('pLose')
    pick_lose = None if not picked else picked.get('pLose')
    place_delta = None
    if observed_finish is not None and pick_end is not None:
        place_delta = int(observed_finish) - int(pick_end)
    lose_drop = None
    if hold_lose_p is not None and pick_lose is not None:
        lose_drop = round(float(hold_lose_p) - float(pick_lose), 3)
    pick_takes = paths.get(what_if_call) or []
    could = bool(place_delta)
    later = [item for item in pick_takes if item.get('kind') == 'LATER']
    key = max(later, key=lambda item: item.get('pPass') or 0) if later else None
    bits = []
    now_bit = f'P{int(now_pos)}' if now_pos is not None else 'this place'
    opp_bit = ''
    if opponent and opponent.get('pHold') is not None:
        opp_bit = (
            f' {opponent["driver"]} holds {opponent["pHold"]:.0%} if he spends {opponent.get("theyDid") or "energy"}'
            f' ({0 if opponent.get("leftPct") is None else round(opponent["leftPct"])}% left).'
        )
    if observed_finish and pick_end is not None and place_delta and they_did == 'ATTACK' and what_if_call == 'SAVE' and key:
        bits.append(
            f'They were {now_bit} and finished {_fmt_int(observed_finish)}: they already ATTACKed here, so SAVE now banks energy and the important lap is {key["note"]} — classified finish {_fmt_int(pick_end)}.{opp_bit}'
        )
    elif observed_finish and pick_end is not None and place_delta:
        later_bit = f', then later {", ".join(item["note"] for item in later[:4])}' if later else ''
        bits.append(
            f'They were {now_bit} and finished {_fmt_int(observed_finish)}: {what_if_call} here{later_bit} puts the classified finish at {_fmt_int(pick_end)}.{opp_bit}'
        )
    elif observed_finish:
        bits.append(
            f'They finished {_fmt_int(observed_finish)}: {what_if_call} here does not add another whole place before the flag.{opp_bit}'
        )
    if lose_drop is not None and lose_drop >= 0.04:
        bits.append(
            f'{call} cuts the chance of being passed in the next 2 laps from {hold_lose_p:.0%} to {pick_lose:.0%}.'
        )
    elif hold_lose_p is not None and pick_lose is not None and lose_drop is not None and lose_drop <= -0.04:
        bits.append(f'{call} raises the chance of being passed in the next 2 laps from {hold_lose_p:.0%} to {pick_lose:.0%}.')
    return {
        'actual': did_row,
        'attack': attack_row,
        'save': save_row,
        'hold': hold_row,
        'call': call,
        'whatIfCall': what_if_call,
        'theyDid': they_did,
        'why': why,
        'resultIf': bits[0] if bits else None,
        'avoidedIf': bits[1] if len(bits) > 1 else None,
        'couldDiffer': could,
        'clips': clips,
        'attackExtraMj': ATTACK_EXTRA_MJ,
        'saveKeepMj': SAVE_KEEP_MJ,
        'convertLift': lift,
        'observedFinish': observed_finish,
        'expectedFinishIfCall': pick_end,
        'expectedFinishIfHold': hold_end,
        'endPositionIfCall': pick_end,
        'endPositionIfHold': hold_end,
        'placesVsObserved': place_delta,
        'placesVsHold': place_delta,
        'avoidedLose': lose_drop,
        'runningPosition': None if now_pos is None else int(now_pos),
        'takesIfCall': pick_takes,
        'extraPlacesIfCall': len(pick_takes),
        'raceEndIfCall': pick_end,
        'raceEndIfHold': hold_end,
        'raceEndIfAttack': None if not attack_row else attack_row.get('raceEnd'),
        'raceEndIfSave': None if not save_row else save_row.get('raceEnd'),
        'pPassIfAttack': p_our_attack,
        'netPassIfAttack': net_now,
        'opponent': opponent,
        'keyLap': None if not key else key.get('lap'),
        'keyTake': key,
        'story': {
            'theyDid': they_did,
            'modelCall': call,
            'whatIfCall': what_if_call,
            'realFinish': observed_finish,
            'whatIfFinish': pick_end,
            'thisLap': start_lap,
            'pPass': p_our_attack,
            'netPass': net_now,
            'opponent': opponent,
            'later': later,
            'keyTake': key,
            'takes': pick_takes,
        },
    }


def _problem(kind: str, step: dict, window: dict | None) -> str:
    ahead = step.get('ahead')
    behind = step.get('behind')
    lap = step.get('lap')
    if kind == 'TOOK':
        return f'Lap {lap}: they passed {ahead or "the car ahead"} — this is the real overtake, used to check the model.'
    if kind == 'MISS_ATTACK':
        wait = (window or {}).get('waitLaps')
        return (
            f'Lap {lap}: they sat within 1.0s of {ahead} for {wait} laps (close enough for DRS) and did not pass.'
        )
    if kind == 'DEFEND':
        return (
            f'Lap {lap}: {behind or "the car behind"} is within 1.2s — spend energy to hold position, or SAVE and try to take it back later.'
        )
    return (
        f'Lap {lap}: they spent a lot of modelled energy with nobody inside 1.0s, so that deploy likely bought no pass.'
    )


def build_incidents(row: dict, event: dict, series: list[dict], windows: list[dict], artifact, trend_slice=None, field=None) -> list[dict]:
    laps = row.get('laps') or []
    by_lap = {item['lap']: item for item in laps}
    incidents = []
    seen = set()

    def add(kind: str, lap: int, window=None, priority=0):
        if lap in seen or lap is None:
            return
        step = by_lap.get(lap)
        if not step:
            return
        point = next((item for item in series if item['lap'] == lap), None)
        choices = simulate_choices(row, event, step, artifact, trend_slice, windows, field)
        modelled = step.get('modelled') or {}
        leftover = _num(modelled.get('startPct'), modelled.get('endPct'))
        if kind == 'MISS_ATTACK' and (choices or {}).get('clips'):
            priority = 4
        seen.add(lap)
        incidents.append({
            'id': f'{kind}-L{lap}',
            'kind': kind,
            'priority': priority,
            'lap': lap,
            'problem': _problem(kind, step, window),
            'live': {
                'position': step.get('timingPosition'),
                'ahead': step.get('ahead'),
                'behind': step.get('behind'),
                'gapToAheadS': step.get('gapToAheadS'),
                'gapToBehindS': step.get('gapToBehindS'),
                'event': step.get('event'),
                'inDrs': in_drs(step),
                'chased': _being_chased(step),
                'action': modelled.get('action'),
                'leftPct': leftover,
                'consumedMj': modelled.get('consumedMj'),
                'usedMj': modelled.get('usedMj'),
            },
            'window': None if not window else {
                'ahead': window.get('ahead'),
                'converted': window.get('converted'),
                'waitLaps': window.get('waitLaps'),
            },
            'model': choices,
            'otherDriver': step.get('ahead') if kind != 'DEFEND' else step.get('behind'),
        })

    for window in windows:
        kind = 'TOOK' if window.get('converted') else 'MISS_ATTACK'
        add(kind, window.get('startLap'), window, priority=0 if kind != 'TOOK' else 2)
    for point in series:
        if point['chased'] and point['lap'] not in seen and (point.get('gapToBehindS') or 9) <= 1.0:
            add('DEFEND', point['lap'], priority=1)
    for point in series:
        if point['lap'] in seen or point.get('inDrs') or point.get('event') == 'OVERTAKE':
            continue
        if (point.get('consumedMj') or 0) >= 0.75 and (point.get('leftPct') or 0) >= 40:
            add('SAVE_CHANCE', point['lap'], priority=3)
            if sum(1 for item in incidents if item['kind'] == 'SAVE_CHANCE') >= 2:
                break
    incidents.sort(key=lambda item: (item['priority'], item['lap']))
    return incidents[:10]


def _pick_trend(trend: dict, driver: str, location: str) -> dict:
    drivers = (trend or {}).get('drivers') or {}
    tracks = (trend or {}).get('tracks') or {}
    global_ = (trend or {}).get('global') or {}
    return {
        'years': trend.get('years'),
        'races': trend.get('races'),
        'driver': drivers.get(driver) or {},
        'track': tracks.get(location) or {},
        'global': global_,
        'highLeftConvertRate': (drivers.get(driver) or {}).get('highLeftConvertRate')
            or global_.get('highLeftConvertRate'),
        'lowLeftConvertRate': (drivers.get(driver) or {}).get('lowLeftConvertRate')
            or global_.get('lowLeftConvertRate'),
        'medianLeftPctConvert': (drivers.get(driver) or {}).get('medianLeftPctConvert')
            or global_.get('medianLeftPctConvert'),
        'medianLeftPctMiss': (drivers.get(driver) or {}).get('medianLeftPctMiss')
            or global_.get('medianLeftPctMiss'),
        'medianLeftPctEarlyConvert': (drivers.get(driver) or {}).get('medianLeftPctEarlyConvert')
            or global_.get('medianLeftPctEarlyConvert'),
        'medianLeftPctEarlyMiss': (drivers.get(driver) or {}).get('medianLeftPctEarlyMiss')
            or global_.get('medianLeftPctEarlyMiss'),
        'medianConsumeMjConvert': (drivers.get(driver) or {}).get('medianConsumeMjConvert')
            or global_.get('medianConsumeMjConvert'),
        'chaseConvertRate': (drivers.get(driver) or {}).get('chaseConvertRate')
            or global_.get('chaseConvertRate'),
        'efficiency': (drivers.get(driver) or {}).get('efficiency') or global_.get('efficiency'),
    }


def build_energy_trend(year, round_number, session_name, driver: str | None) -> dict:
    if not year or not round_number:
        first = clip_cached(None, None, 'Race', driver, None, 'AUTO', 4.0)
        if first.get('error'):
            return first
        event = (first.get('actualRace') or {}).get('event') or {}
        year = event.get('year')
        round_number = event.get('round')
        session_name = 'Race'
    path = session_path(int(year), int(round_number), session_name or 'Race')
    if not path.exists():
        return {'error': f'no cached race for {year} R{round_number}'}
    payload = load_session(path)
    event = payload.get('event') or {}
    field = build_field_story(payload)
    location = str(event.get('location') or '')
    selected_code = (driver or '').upper()
    row = next((item for item in field.get('drivers') or [] if item['driver'] == selected_code), None)
    if not row:
        row = (field.get('drivers') or [None])[0]
        selected_code = row['driver'] if row else None
    if not row:
        return {'error': f'no laps for {driver or "selected driver"}'}
    laps = row.get('laps') or []
    windows = [item for item in extract_windows(field, event) if item['driver'] == selected_code]
    trend_file = {}
    if TREND.exists():
        try:
            trend_file = json.loads(TREND.read_text(encoding='utf-8'))
        except Exception:
            trend_file = {}
    trend = _pick_trend(trend_file, selected_code, location)
    artifact = joblib.load(MODELS) if MODELS.exists() else None

    series = []
    peak = []
    for step in laps:
        modelled = step.get('modelled') or {}
        consume = _num(modelled.get('consumedMj'), 0.0) or 0.0
        chased = _being_chased(step)
        flags = []
        if step.get('event') == 'OVERTAKE':
            flags.append('OVERTAKE')
        if in_drs(step):
            flags.append('DRS')
        if chased:
            flags.append('CHASED')
        if step.get('isPitLap'):
            flags.append('PIT')
        if consume >= 0.75 and not step.get('isPitLap'):
            flags.append('HIGH_SPEND')
        point = {
            'lap': step.get('lap'),
            'position': step.get('timingPosition'),
            'event': step.get('event'),
            'ahead': step.get('ahead'),
            'behind': step.get('behind'),
            'gapToAheadS': step.get('gapToAheadS'),
            'gapToBehindS': step.get('gapToBehindS'),
            'closingRateS': step.get('closingRateS'),
            'consumedMj': modelled.get('consumedMj'),
            'harvestedMj': modelled.get('harvestedMj'),
            'usedMj': modelled.get('usedMj'),
            'leftPct': modelled.get('endPct'),
            'action': modelled.get('action'),
            'inDrs': in_drs(step),
            'chased': chased,
            'flags': flags,
        }
        series.append(point)
        if not step.get('isPitLap'):
            peak.append(point)
    peak.sort(key=lambda item: -(item.get('consumedMj') or 0))
    hotspots = peak[:6]

    window_rows = []
    for window in windows[:12]:
        step = _window_step(laps, window['startLap'])
        converted = bool(window.get('converted'))
        counter = _counterfactual(row, event, step, artifact, trend, converted, windows, field) if artifact else None
        window_rows.append({
            'startLap': window['startLap'],
            'ahead': window['ahead'],
            'converted': converted,
            'waitLaps': window.get('waitLaps'),
            'duration': window.get('duration'),
            'consumedMj': (step.get('modelled') or {}).get('consumedMj'),
            'leftPct': (step.get('modelled') or {}).get('startPct')
                if (step.get('modelled') or {}).get('startPct') is not None
                else (step.get('modelled') or {}).get('endPct'),
            'chased': _being_chased(step),
            'counterfactual': counter,
        })

    chase_laps = [item for item in series if item['chased']]
    overtake_laps = [item['lap'] for item in series if item['event'] == 'OVERTAKE']
    missed = [item for item in window_rows if not item['converted']]
    converted = [item for item in window_rows if item['converted']]
    spend_here = [
        item for item in missed
        if (item.get('counterfactual') or {}).get('verdict') == 'SPEND HERE'
    ]
    expected_extra = None
    if trend.get('efficiency') is not None:
        expected_extra = round(float(trend['efficiency']) * len(missed), 2)

    incidents = build_incidents(row, event, series, window_rows, artifact, trend, field)
    try:
        observed_finish = int(row.get('classifiedPosition') or (row.get('laps') or [{}])[-1].get('timingPosition'))
    except (TypeError, ValueError):
        observed_finish = None
    ranked = [
        item for item in incidents
        if (item.get('model') or {}).get('couldDiffer')
    ]
    ranked.sort(key=lambda item: (
        0 if (item.get('model') or {}).get('theyDid') != (item.get('model') or {}).get('whatIfCall') else 1,
        -((item.get('model') or {}).get('extraPlacesIfCall') or 0),
    ))
    best_incident = ranked[0] if ranked else (incidents[0] if incidents else None)
    best_model = None if not best_incident else (best_incident.get('model') or {})
    what_if = {
        'observedFinish': observed_finish,
        'bestIncidentId': None if not best_incident else best_incident.get('id'),
        'bestLap': None if not best_incident else best_incident.get('lap'),
        'bestKind': None if not best_incident else best_incident.get('kind'),
        'bestCall': best_model.get('whatIfCall') or best_model.get('call'),
        'theyDid': best_model.get('theyDid'),
        'afterIfCall': best_model.get('raceEndIfCall'),
        'afterIfHold': best_model.get('raceEndIfHold'),
        'endIfCall': best_model.get('raceEndIfCall'),
        'endIfHold': best_model.get('raceEndIfHold'),
        'placesVsHold': best_model.get('placesVsHold'),
        'avoidedLose': best_model.get('avoidedLose'),
        'takes': best_model.get('takesIfCall') or [],
        'extraPlaces': best_model.get('extraPlacesIfCall') or 0,
        'opponent': best_model.get('opponent'),
        'netPass': best_model.get('netPassIfAttack'),
        'keyTake': best_model.get('keyTake'),
        'story': best_model.get('story'),
        'attackExtraMj': ATTACK_EXTRA_MJ,
        'saveKeepMj': SAVE_KEEP_MJ,
        'note': (
            'What-if finish uses the trained pass chance minus the other car’s hold chance, then later missed 1.0s windows still left with energy.'
        ),
    }
    return {
        'ok': True,
        'schema': 'energy-overtake-story.v2',
        'event': event,
        'driver': selected_code,
        'name': row.get('name'),
        'drsGapS': DRS_GAP_S,
        'attackExtraMj': ATTACK_EXTRA_MJ,
        'series': series,
        'hotspots': hotspots,
        'windows': window_rows,
        'overtakeLaps': overtake_laps,
        'chaseLaps': [item['lap'] for item in chase_laps],
        'highSpendLaps': [item['lap'] for item in hotspots],
        'summary': {
            'laps': len(series),
            'overtakes': len(overtake_laps),
            'drsWindows': len(window_rows),
            'drsConverted': len(converted),
            'drsMissed': len(missed),
            'chaseLaps': len(chase_laps),
            'peakConsumeMj': hotspots[0]['consumedMj'] if hotspots else None,
            'peakConsumeLap': hotspots[0]['lap'] if hotspots else None,
            'expectedExtraPlaces': expected_extra,
            'spendHereWindows': len(spend_here),
        },
        'trend': trend,
        'bestCounterfactual': spend_here[0] if spend_here else (missed[0] if missed else None),
        'incidents': incidents,
        'whatIf': what_if,
        'saveKeepMj': SAVE_KEEP_MJ,
        'provenance': [
            'REAL timing positions, gaps, OVERTAKE flags, and chase gaps from the cached session',
            'MODELLED leftover/consume from the 4 MJ C5.2 window starting at 100% — not team battery',
            'DRS = timing gap ≤ 1.0s, not official DRS telemetry',
            'Trend medians from 2018–2025 cached races (2014 is not in this store)',
            'What-if ATTACK commits the 1.0s window: convert-in-1 blended with that driver\'s 2018–2025 DRS efficiency',
            'What-if SAVE keeps one DELAY-vs-SAVE deploy and lowers modelled 2-lap loss while being chased',
            'End position = classified finish shifted by the after-fight delta vs what they actually did — not a second full-race replay',
        ],
    }


if __name__ == '__main__':
    payload = json.loads(sys.stdin.read() or '{}')
    print(json.dumps(build_energy_trend(
        payload.get('year'),
        payload.get('round'),
        payload.get('session') or 'Race',
        payload.get('driver'),
    )))
