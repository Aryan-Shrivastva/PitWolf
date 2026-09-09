"""Backend tactical replay for the ATTACK/SAVE/DELAY decision tree.

This is intentionally a small, deterministic state-transition engine.  It does
not claim to reconstruct private team battery telemetry.  It consumes observed
race rows plus model probabilities, carries a modelled SoC for both cars, and
evaluates a selected driver's three available actions at every lap in a short
look-ahead horizon.

The external tree has three children per lap.  The opponent is represented by a
calibrated best-response policy inside each child, so a pass changes the roles:
the selected driver becomes the defender and the former defender can spend its
remaining energy to retake the position.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

try:
    from energy_transition import (CAPACITY_MJ, ENERGY_MODEL_VERSION,
                                   era_for_year, get_era_config, transition_soc)
except ImportError:  # allow package-style unit tests as well as direct scripts
    from .energy_transition import (CAPACITY_MJ, ENERGY_MODEL_VERSION,
                                    era_for_year, get_era_config, transition_soc)


TREE_VERSION = "tactical-tree.v7"
SCHEMA_VERSION = "replay-state.v7"
START_SOC_MJ = 2.8
MAX_HORIZON = 6
ACTIONS = ("ATTACK", "SAVE", "DELAY")
MIN_ATTACK_SOC_MJ = 0.25
MIN_DELAY_SOC_MJ = 0.08
RESERVE_WEIGHT = 25.0

def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.0, high: float = CAPACITY_MJ) -> float:
    return max(low, min(high, number(value)))


def probability(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, number(value, default)))


def row_for(rows: list[dict[str, Any]], lap: int, driver: str, defender: str) -> dict[str, Any] | None:
    for row in rows:
        if (number(row.get("lap")) == lap and row.get("driver") == driver
                and row.get("defender") == defender):
            return row
    return None


def soc_from_laps(energy_laps: Any, driver: str, lap: int) -> float:
    """Read the last observed pre-decision SoC for a driver when available."""
    if isinstance(energy_laps, dict):
        values = energy_laps.get(driver, [])
    else:
        values = energy_laps if isinstance(energy_laps, list) else []
    best = None
    for item in values:
        if not isinstance(item, dict) or number(item.get("lap"), -1) > lap:
            continue
        candidate = item.get("socEndMj")
        if candidate is not None:
            best = number(candidate, best if best is not None else START_SOC_MJ)
    return clamp(best if best is not None else START_SOC_MJ)


def model_signal(row: dict[str, Any] | None, label: str, default: float) -> float:
    probabilities = (row or {}).get("pred", {}).get("probabilities", {})
    if not isinstance(probabilities, dict):
        probabilities = {}
    return probability(probabilities.get(label), default)


def context(row: dict[str, Any] | None) -> dict[str, float]:
    row = row or {}
    weather = row.get("weather") if isinstance(row.get("weather"), dict) else {}
    rainfall = weather.get("rainfall", 0.0)
    wet = 1.0 if str(rainfall).lower() in ("true", "on", "yes", "rain") else number(rainfall)
    return {
        "gap": max(0.0, number(row.get("gapS"), 1.2)),
        "closing": number(row.get("closingRateS")),
        "speed": number(row.get("speedDeltaKph")),
        "tyre": number(row.get("tyreAgeDiff")),
        "slipstream": probability(row.get("slipstreamProxy")),
        "dirty_air": probability(row.get("dirtyAirRisk")),
        "traffic_ahead": max(0.0, number(row.get("trafficAheadCount"))),
        "traffic_behind": max(0.0, number(row.get("trafficBehindCount"))),
        "our_tyre_deg": probability(row.get("attackerTyreDegProxy")),
        "opponent_tyre_deg": probability(row.get("defenderTyreDegProxy")),
        "wet": probability(wet),
        "track_clear": row.get("attackerTrackStatus") in (None, "1") and row.get("defenderTrackStatus") in (None, "1"),
        "pit_distorted": bool(row.get("pitDistorted")),
    }


def observed_pit_tyre_context(rows: list[dict[str, Any]], focus: dict[str, Any],
                               selected: str, defender: str, start_lap: int,
                               horizon: int) -> dict[str, Any]:
    """Describe observed pit/tyre context without inventing a BOX simulation.

    The tactical tree has exactly ATTACK/SAVE/DELAY branches. Public timing can
    tell us that a real pit cycle occurred, but cannot identify the
    counterfactual pit timing, rejoin traffic, or tyre warm-up of a different
    strategy. Those events are therefore disclosed and held fixed, not scored
    as a hidden fourth tactical choice.
    """
    end_lap = start_lap + max(0, horizon - 1)
    events = []
    seen = set()
    for row in [focus, *rows]:
        lap = int(number(row.get("lap"), -1))
        if lap < start_lap or lap > end_lap:
            continue
        for driver, role in ((selected, "SELECTED"), (defender, "OPPONENT")):
            if row.get("driver") == driver:
                prefix = "attacker"
            elif row.get("defender") == driver:
                prefix = "defender"
            else:
                continue
            pit_in = bool(row.get(f"{prefix}PitIn"))
            pit_out = bool(row.get(f"{prefix}PitOut"))
            if not pit_in and not pit_out:
                continue
            key = (driver, lap, pit_in, pit_out)
            if key in seen:
                continue
            seen.add(key)
            event = "PIT IN/OUT" if pit_in and pit_out else ("PIT IN" if pit_in else "PIT OUT")
            events.append({"lap": lap, "driver": driver, "role": role, "event": event})

    def tyre_state(prefix: str) -> dict[str, Any]:
        return {
            "compound": focus.get(f"{prefix}Compound"),
            "ageDifferenceLaps": number(focus.get("tyreAgeDiff")) if prefix == "attacker" else -number(focus.get("tyreAgeDiff")),
            "degradationProxy": round(probability(focus.get(f"{prefix}TyreDegProxy")), 3),
        }

    return {
        "mode": "OBSERVED_CONTEXT_HELD_FIXED",
        "boxActionSimulated": False,
        "horizonLaps": horizon,
        "observedPitEvents": sorted(events, key=lambda item: (item["lap"], item["driver"])),
        "futurePitCycleWithinHorizon": bool(events),
        "initialTyres": {
            "selected": tyre_state("attacker"),
            "opponent": tyre_state("defender"),
        },
        "handling": (
            "Observed pit cycles gate normal overtake claims. Alternative BOX timing, rejoin traffic, "
            "tyre warm-up and undercut/overcut outcomes are not simulated in this three-action tree."
        ),
    }


def attack_chance(row: dict[str, Any] | None, action: str, our_soc: float,
                  opponent_soc: float, defending: bool) -> float:
    """Estimate the chance that the attacking side changes position this lap."""
    values = context(row)
    # A yellow/SC/VSC/red-flag or pit-cycle state is not a normal overtake
    # window. Keep the branch in the tree for auditability, but prevent it from
    # claiming a pass opportunity from that state.
    if not values["track_clear"] or values["pit_distorted"]:
        return 0.0
    gap = values["gap"]
    closing = values["closing"]
    speed_delta = values["speed"]
    tyre_delta = values["tyre"]
    attack_signal = model_signal(row, "ATTACK", 0.25)
    delay_signal = model_signal(row, "DELAY", 0.25)
    reserve_edge = (our_soc - opponent_soc) / CAPACITY_MJ
    window = max(0.0, min(1.0, (1.25 - gap) / 1.25))
    closing_signal = max(0.0, min(1.0, closing / 0.35))
    speed_signal = max(0.0, min(1.0, (speed_delta + 8.0) / 16.0))
    tyre_signal = max(0.0, min(1.0, tyre_delta / 10.0))
    raw = (0.10 + (0.22 * attack_signal) + (0.22 * window)
           + (0.16 * closing_signal) + (0.10 * speed_signal)
           + (0.08 * tyre_signal) + (0.12 * reserve_edge))
    if defending:
        our_tyre_deg = values["opponent_tyre_deg"]
        opponent_tyre_deg = values["our_tyre_deg"]
    else:
        our_tyre_deg = values["our_tyre_deg"]
        opponent_tyre_deg = values["opponent_tyre_deg"]
    raw += (0.12 * values["slipstream"])
    raw -= (0.10 * values["dirty_air"])
    raw -= (0.10 * our_tyre_deg)
    raw += (0.06 * opponent_tyre_deg)
    raw -= (0.04 * min(1.0, values["traffic_ahead"] / 3.0))
    raw -= (0.03 * values["wet"])
    action_factor = {"ATTACK": 1.00, "DELAY": 0.42, "SAVE": 0.12}[action]
    if defending:
        # When the selected car is ahead, this function measures the
        # opponent's repass threat.  A defensive action reduces exposure.
        action_factor = {"ATTACK": 0.48, "DELAY": 0.70, "SAVE": 0.92}[action]
        raw = (0.08 + (0.32 * attack_signal) + (0.22 * window)
               + (0.16 * closing_signal) + (0.10 * speed_signal)
               - (0.10 * reserve_edge))
        raw *= action_factor
    elif action == "DELAY":
        raw += 0.05 * delay_signal
    # A car with no remaining modelled SoC cannot receive the same pass
    # probability as a charged car.  Keep a small residual for tyre/pace
    # effects, but make energy exhaustion materially affect ATTACK/DELAY.
    energy_factor = max(0.0, min(1.0, our_soc / 0.8))
    if action == "ATTACK":
        if our_soc < MIN_ATTACK_SOC_MJ:
            return 0.0
        raw *= energy_factor
    elif action == "DELAY":
        if our_soc < MIN_DELAY_SOC_MJ:
            return 0.0
        raw *= 0.35 + (0.65 * energy_factor)
    return max(0.01, min(0.97, raw * action_factor))


def best_response(row: dict[str, Any] | None, opponent_soc: float, our_soc: float,
                  opponent_is_attacker: bool, era: str, deploy_scale: float = 1.0,
                  harvest_scale: float = 1.0) -> tuple[str, float]:
    scores = {}
    for action in ACTIONS:
        next_soc, _, _ = transition_soc(
            opponent_soc, action, row, not opponent_is_attacker, era=era,
            deploy_scale=deploy_scale, harvest_scale=harvest_scale)
        chance = attack_chance(row, action, next_soc, our_soc, not opponent_is_attacker)
        # An attacker values a pass; a defender values survival.
        score = chance if opponent_is_attacker else (1.0 - chance)
        score += 0.05 * (next_soc / CAPACITY_MJ)
        scores[action] = score
    selected = max(ACTIONS, key=lambda action: scores[action])
    return selected, round(scores[selected], 4)


def rollout_to_finish(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a future-blind, two-car policy rollout from one recorded snapshot.

    This deliberately does *not* receive later decision rows.  The classifier
    signal and public context at the chosen lap are carried forward while both
    cars' modelled SoC and their probability of pair-order reversal evolve.
    It is therefore a race-to-flag energy/overtake policy experiment, not a
    reconstructed full-grid race or a claim about private team strategy.
    """
    focus = payload.get("focus") or {}
    selected = str(focus.get("driver") or "")
    opponent = str(focus.get("defender") or "")
    start_lap = int(number(focus.get("lap"), 1))
    total_laps = max(start_lap, int(number(payload.get("totalLaps"), start_lap)))
    regulation_era = str(payload.get("regulationEra") or era_for_year(payload.get("year")))
    energy_config = get_era_config(regulation_era)
    energy_laps = payload.get("energyLaps", {})
    state = {
        "lap": start_lap,
        "ahead": number(focus.get("position"), 99) < number(focus.get("defenderPosition"), 99),
        "aheadProbability": 1.0 if number(focus.get("position"), 99) < number(focus.get("defenderPosition"), 99) else 0.0,
        "ourSoc": soc_from_laps(energy_laps, selected, start_lap),
        "defenderSoc": soc_from_laps(energy_laps, opponent, start_lap),
    }
    initial_ahead = state["ahead"]
    path: list[dict[str, Any]] = []
    action_changes: list[dict[str, Any]] = []
    last_action = None

    while state["lap"] <= total_laps:
        candidates = []
        for action in ACTIONS:
            our_soc, deploy, harvest = transition_soc(
                state["ourSoc"], action, focus, state["ahead"], era=regulation_era)
            opponent_action, response_score = best_response(
                focus, state["defenderSoc"], state["ourSoc"], state["ahead"], regulation_era)
            opponent_soc, opponent_deploy, opponent_harvest = transition_soc(
                state["defenderSoc"], opponent_action, focus, not state["ahead"], era=regulation_era)
            if state["ahead"]:
                repass = attack_chance(focus, opponent_action, opponent_soc, our_soc, True)
                next_ahead = state["aheadProbability"] * max(0.02, min(0.99, 1.0 - repass))
                event_probability = 1.0 - repass
            else:
                pass_probability = attack_chance(focus, action, our_soc, opponent_soc, False)
                next_ahead = state["aheadProbability"] + ((1.0 - state["aheadProbability"]) * pass_probability)
                event_probability = pass_probability
            # This policy score selects the action using only the current
            # carried state.  It has no access to a future lap, result, pit
            # event, race-control event, or future public timing row.
            utility = (100.0 * next_ahead) + (RESERVE_WEIGHT * our_soc) + (4.0 * event_probability)
            candidates.append({
                "action": action,
                "probability": round(event_probability, 4),
                "aheadProbability": round(next_ahead, 4),
                "ourSoc": round(our_soc, 3),
                "defenderSoc": round(opponent_soc, 3),
                "deployMj": deploy,
                "harvestMj": harvest,
                "opponentAction": opponent_action,
                "opponentDeployMj": opponent_deploy,
                "opponentHarvestMj": opponent_harvest,
                "opponentResponseScore": response_score,
                "utility": utility,
            })
        chosen = max(candidates, key=lambda candidate: candidate["utility"])
        entry = {key: chosen[key] for key in (
            "action", "probability", "aheadProbability", "ourSoc", "defenderSoc",
            "deployMj", "harvestMj", "opponentAction", "opponentDeployMj",
            "opponentHarvestMj", "opponentResponseScore")}
        entry.update({
            "lap": state["lap"],
            "role": "DEFENDING" if state["ahead"] else "ATTACKING",
            "contextSource": "START_STATE_CARRIED_FUTURE_BLIND",
            "pitPlan": "NO BOX MODEL · START STATE CARRIED",
        })
        path.append(entry)
        if chosen["action"] != last_action:
            action_changes.append({
                "lap": state["lap"],
                "action": chosen["action"],
                "opponentAction": chosen["opponentAction"],
                "ourSoc": chosen["ourSoc"],
                "defenderSoc": chosen["defenderSoc"],
                "aheadProbability": chosen["aheadProbability"],
            })
            last_action = chosen["action"]
        state = {
            "lap": state["lap"] + 1,
            "ahead": chosen["aheadProbability"] >= 0.5,
            "aheadProbability": chosen["aheadProbability"],
            "ourSoc": chosen["ourSoc"],
            "defenderSoc": chosen["defenderSoc"],
        }

    finish_positions = payload.get("finishPositions") or {}
    actual_finish = finish_positions.get(selected)
    opponent_finish = finish_positions.get(opponent)
    actual_pair_ahead = (
        isinstance(actual_finish, (int, float)) and isinstance(opponent_finish, (int, float))
        and actual_finish < opponent_finish
    )
    modelled_pair_ahead = bool(path and path[-1]["aheadProbability"] >= 0.5)
    root_context = context(focus)
    rule_context = payload.get("ruleContext") if isinstance(payload.get("ruleContext"), dict) else {}
    rule_application = (
        "TRACK_DISTANCE_ALIGNMENT_REQUIRED"
        if rule_context.get("eventSpecificDataLoaded")
        else "DISCLOSURE_ONLY_UNTIL_EVENT_APPENDIX_LOADED"
    )
    return {
        "schemaVersion": "race-branch.v1",
        "treeVersion": "future-blind-two-car-rollout.v1",
        "mode": "FUTURE_BLIND_RACE_ROLLOUT",
        "tree": {
            "lap": start_lap,
            "ourSoc": round(soc_from_laps(energy_laps, selected, start_lap), 3),
            "defenderSoc": round(soc_from_laps(energy_laps, opponent, start_lap), 3),
            "bestAction": path[0]["action"] if path else None,
            "children": [],
        },
        "path": path,
        "horizon": len(path),
        "startLap": start_lap,
        "finishLap": total_laps,
        "selectedRoleAtJump": str(focus.get("selectedRole") or ("DEFENDING" if initial_ahead else "ATTACKING")),
        "actionChanges": action_changes,
        "modelledPairAheadAtFlag": modelled_pair_ahead,
        "modelledPairAheadProbabilityAtFlag": path[-1]["aheadProbability"] if path else 0.0,
        "actualPairAheadAtFlag": actual_pair_ahead,
        "actualFinishPosition": actual_finish,
        "opponentFinishPosition": opponent_finish,
        "fullGridFinishForecast": None,
        "finishComparison": {
            "pairOrderMatchesObserved": modelled_pair_ahead == actual_pair_ahead,
            "fullGridComparable": False,
            "reason": "Only the selected pair is modelled. No full-grid pace, pit, tyre, traffic, retirement, or race-control counterfactual is available.",
        },
        "stateProvenance": {
            "horizonLaps": len(path),
            "observedContextLaps": 1 if path else 0,
            "carriedContextLaps": max(0, len(path) - 1),
            "sourceCounts": {"START_STATE_CARRIED_FUTURE_BLIND": len(path)},
            "note": "The branch receives only the selected-lap public state. Every later step carries and evolves model state; later recorded race rows are excluded from policy input.",
        },
        "decisionContext": {
            "raceControl": "CLEAR" if root_context["track_clear"] else "GATED",
            "pitDistorted": root_context["pit_distorted"],
            "attackerTrackStatus": focus.get("attackerTrackStatus"),
            "defenderTrackStatus": focus.get("defenderTrackStatus"),
            "overtakeActionsEnabled": root_context["track_clear"] and not root_context["pit_distorted"],
            "ruleContext": {**rule_context, "application": rule_application},
        },
        "opponentPolicy": {
            "id": "CONSERVATIVE_BEST_RESPONSE_HEURISTIC_V1",
            "type": "DETERMINISTIC_CONSERVATIVE_BEST_RESPONSE",
            "description": "At each simulated lap the opposing car selects the response that maximises its modelled immediate objective using only carried state.",
        },
        "energyModelVersion": ENERGY_MODEL_VERSION,
        "regulationEra": regulation_era,
        "energyConfig": {
            "id": energy_config["id"],
            "capacityMj": energy_config["capacityMj"],
            "calibrationStatus": energy_config["calibrationStatus"],
            "source": energy_config["source"],
        },
        "assumptions": [
            "The rollout is future-blind: later recorded rows are not supplied to the policy.",
            "SoC is a public-data model surrogate, not private battery telemetry.",
            "Only the selected car and its starting attack target are simulated.",
            "A full-grid finish forecast is intentionally unavailable until pit, tyre, traffic, pace, retirement and race-control models exist.",
        ],
    }


def build_tree(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("mode") == "FUTURE_BLIND_RACE_ROLLOUT":
        return rollout_to_finish(payload)
    focus = payload.get("focus") or {}
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    selected = str(focus.get("driver") or "")
    initial_defender = str(focus.get("defender") or "")
    start_lap = int(number(focus.get("lap"), 1))
    total_laps = max(start_lap, int(number(payload.get("totalLaps"), start_lap)))
    horizon = max(0, min(MAX_HORIZON, int(number(payload.get("holdLaps"), MAX_HORIZON)),
                         total_laps - start_lap))
    energy_laps = payload.get("energyLaps", {})
    regulation_era = str(payload.get("regulationEra") or era_for_year(payload.get("year")))
    energy_config = get_era_config(regulation_era)
    initial_our_soc = soc_from_laps(energy_laps, selected, start_lap)
    initial_defender_soc = soc_from_laps(energy_laps, initial_defender, start_lap)
    def simulate(our_start: float, defender_start: float, deploy_scale: float = 1.0,
                 harvest_scale: float = 1.0) -> tuple[dict[str, Any], list[dict[str, Any]], int, int]:
        """Evaluate one deterministic tree from an explicit pair of SoC states."""
        node_count = 0
        leaf_count = 0

        def expand(state: dict[str, Any], depth: int) -> tuple[float, dict[str, Any]]:
            nonlocal node_count, leaf_count
            node_count += 1
            if depth >= horizon or state["lap"] >= total_laps:
                leaf_count += 1
                value = (state["leadLaps"] * 100.0
                         + (state["aheadProbability"] * 20.0)
                         + (state["ourSoc"] * RESERVE_WEIGHT))
                return value, {
                    "lap": state["lap"], "role": "DEFENDING" if state["ahead"] else "ATTACKING",
                    "leadLaps": round(state["leadLaps"], 3),
                    "aheadProbability": round(state["aheadProbability"], 4),
                    "ourSoc": round(state["ourSoc"], 3),
                    "defenderSoc": round(state["defenderSoc"], 3),
                    "children": [],
                }

            lap = state["lap"]
            forward = row_for(rows, lap, selected, initial_defender)
            reverse = row_for(rows, lap, initial_defender, selected) or {}
            role_aligned = reverse if state["ahead"] else forward
            if role_aligned:
                observed = role_aligned
                context_source = "OBSERVED_ROLE_ALIGNED"
            elif forward:
                # The observed pair still exists at this lap but the selected
                # car's counterfactual role differs from the real ordering.
                observed = forward
                context_source = "OBSERVED_MATCHUP_ROLE_CARRIED"
            else:
                # Do not fabricate an unobserved future battle row.  Carry
                # the decision-time state explicitly and disclose the limit.
                observed = focus
                context_source = "FOCUS_CONTEXT_CARRIED"
            children = []
            for action in ACTIONS:
                our_soc, deploy, harvest = transition_soc(
                    state["ourSoc"], action, observed, state["ahead"], era=regulation_era,
                    deploy_scale=deploy_scale, harvest_scale=harvest_scale)
                opponent_action, response_score = best_response(
                    reverse if state["ahead"] else forward,
                    state["defenderSoc"], state["ourSoc"], state["ahead"], regulation_era,
                    deploy_scale, harvest_scale,
                )
                opponent_soc, opponent_deploy, opponent_harvest = transition_soc(
                    state["defenderSoc"], opponent_action, observed, not state["ahead"],
                    era=regulation_era, deploy_scale=deploy_scale, harvest_scale=harvest_scale,
                )
                if state["ahead"]:
                    repass = attack_chance(reverse or forward, opponent_action,
                                           opponent_soc, our_soc, True)
                    survival = max(0.02, min(0.99, 1.0 - repass))
                    next_ahead = state["aheadProbability"] * survival
                    event_probability = survival
                else:
                    pass_probability = attack_chance(forward, action, our_soc,
                                                     opponent_soc, False)
                    next_ahead = state["aheadProbability"] + ((1.0 - state["aheadProbability"])
                                                                * pass_probability)
                    event_probability = pass_probability
                lead_laps = state["leadLaps"] + next_ahead
                next_state = {
                    "lap": lap + 1,
                    "ahead": next_ahead >= 0.5,
                    "aheadProbability": next_ahead,
                    "leadLaps": lead_laps,
                    "ourSoc": our_soc,
                    "defenderSoc": opponent_soc,
                }
                value, next_node = expand(next_state, depth + 1)
                children.append({
                    "action": action,
                    "probability": round(event_probability, 4),
                    "aheadProbability": round(next_ahead, 4),
                    "ourSoc": round(our_soc, 3),
                    "defenderSoc": round(opponent_soc, 3),
                    "deployMj": deploy,
                    "harvestMj": harvest,
                    "opponentAction": opponent_action,
                    "opponentDeployMj": opponent_deploy,
                    "opponentHarvestMj": opponent_harvest,
                    "opponentResponseScore": response_score,
                    "contextSource": context_source,
                    "pitPlan": "OBSERVED PIT WINDOW; NO BATTERY RESET" if bool(observed.get("pitDistorted")) else "STAY OUT",
                    "leadLaps": round(lead_laps, 3),
                    "value": value,
                    "next": next_node,
                })
            best = max(children, key=lambda child: child["value"])
            for child in children:
                child["best"] = child is best
            return best["value"], {
                "lap": lap, "role": "DEFENDING" if state["ahead"] else "ATTACKING",
                "leadLaps": round(state["leadLaps"], 3),
                "aheadProbability": round(state["aheadProbability"], 4),
                "ourSoc": round(state["ourSoc"], 3),
                "defenderSoc": round(state["defenderSoc"], 3),
                "bestAction": best["action"],
                "children": children,
            }

        _, scenario_tree = expand({
            "lap": start_lap,
            "ahead": False,
            "aheadProbability": 0.0,
            "leadLaps": 0.0,
            "ourSoc": clamp(our_start),
            "defenderSoc": clamp(defender_start),
        }, 0)
        scenario_path = []
        cursor = scenario_tree
        while cursor.get("children"):
            child = next((item for item in cursor["children"] if item.get("best")), cursor["children"][0])
            scenario_path.append({key: child[key] for key in (
                "action", "probability", "ourSoc", "defenderSoc", "opponentAction",
                "deployMj", "harvestMj", "opponentDeployMj", "opponentHarvestMj",
                    "pitPlan", "leadLaps", "aheadProbability")})
            scenario_path[-1]["contextSource"] = child.get("contextSource")
            scenario_path[-1]["lap"] = cursor["lap"]
            scenario_path[-1]["role"] = cursor["role"]
            cursor = child["next"]
        return scenario_tree, scenario_path, node_count, leaf_count

    tree, path, node_count, leaf_count = simulate(initial_our_soc, initial_defender_soc)
    sensitivity_summary = None
    if payload.get("includeSensitivity", True):
        sensitivity_cases = (
            ("ATTACKER_LOW", "ATTACKER −0.50 MJ", initial_our_soc - 0.5, initial_defender_soc),
            ("BASE", "BASE ESTIMATE", initial_our_soc, initial_defender_soc),
            ("DEFENDER_HIGH", "DEFENDER +0.50 MJ", initial_our_soc, initial_defender_soc + 0.5),
        )
        soc_sensitivity = []
        for scenario_id, label, our_start, defender_start in sensitivity_cases:
            scenario_tree, scenario_path, _, _ = simulate(our_start, defender_start)
            first_step = scenario_path[0] if scenario_path else {}
            soc_sensitivity.append({
                "id": scenario_id,
                "label": label,
                "attackerStartSocMj": round(clamp(our_start), 3),
                "defenderStartSocMj": round(clamp(defender_start), 3),
                "recommendedAction": first_step.get("action", scenario_tree.get("bestAction")),
                "expectedLeadLaps": round(number(scenario_path[-1].get("leadLaps") if scenario_path else 0.0), 3),
            })
        base_action = soc_sensitivity[1]["recommendedAction"]
        sensitivity_summary = {
            "perturbationMj": 0.5,
            "stableRecommendation": all(item["recommendedAction"] == base_action for item in soc_sensitivity),
            "baseAction": base_action,
            "cases": soc_sensitivity,
            "note": "Sensitivity scenarios perturb modelled SoC only; they are not measurements of either car's battery.",
        }

    # Starting SoC is only one uncertainty in the public-data surrogate.  Run
    # a second, deliberately small stress test against the energy transition
    # calibration itself.  The values are not regulatory limits and do not
    # imply that a real car deployed or harvested at these rates.
    calibration_sensitivity = None
    if payload.get("includeSensitivity", True):
        calibration_cases = (
            ("CONSERVATIVE", "HIGH DEPLOY / LOW HARVEST", 1.15, 0.85),
            ("BASE", "BASE CALIBRATION", 1.00, 1.00),
            ("FAVOURABLE", "LOW DEPLOY / HIGH HARVEST", 0.85, 1.15),
        )
        calibration_results = []
        for scenario_id, label, deploy_scale, harvest_scale in calibration_cases:
            scenario_tree, scenario_path, _, _ = simulate(
                initial_our_soc,
                initial_defender_soc,
                deploy_scale=deploy_scale,
                harvest_scale=harvest_scale,
            )
            first_step = scenario_path[0] if scenario_path else {}
            calibration_results.append({
                "id": scenario_id,
                "label": label,
                "deployScale": deploy_scale,
                "harvestScale": harvest_scale,
                "recommendedAction": first_step.get("action", scenario_tree.get("bestAction")),
                "expectedLeadLaps": round(number(scenario_path[-1].get("leadLaps") if scenario_path else 0.0), 3),
            })
        base_action = calibration_results[1]["recommendedAction"]
        calibration_sensitivity = {
            "variationPercent": 15,
            "stableRecommendation": all(item["recommendedAction"] == base_action for item in calibration_results),
            "baseAction": base_action,
            "cases": calibration_results,
            "note": (
                "This is a ±15% surrogate-calibration stress test for modelled deployment and harvest. "
                "It is not measured team telemetry, an FIA power limit, or an FIA recharge limit."
            ),
        }

    observed_lead_laps = int(number(focus.get("observedLeadLaps"), -1))
    if observed_lead_laps < 0:
        observed_lead_laps = int(number(focus.get("holdLaps"), 0)) if focus.get("held") else (1 if focus.get("passedNow") else 0)
    # The replay is intentionally a bounded tactical horizon.  Keep the raw
    # observed duration for auditability, but compare it with the estimate only
    # inside the same horizon; otherwise a pass held to the chequered flag would
    # be compared unfairly with a six-lap counterfactual.
    actual_lead_laps = min(observed_lead_laps, horizon)
    expected = path[-1]["leadLaps"] if path else 0.0
    root_context = context(focus)
    pit_tyre_context = observed_pit_tyre_context(
        rows, focus, selected, initial_defender, start_lap, horizon)
    context_source_counts: dict[str, int] = {}
    for step in path:
        source = str(step.get("contextSource") or "FOCUS_CONTEXT_CARRIED")
        context_source_counts[source] = context_source_counts.get(source, 0) + 1
    observed_context_laps = sum(
        count for source, count in context_source_counts.items()
        if source.startswith("OBSERVED_")
    )
    state_provenance = {
        "horizonLaps": horizon,
        "observedContextLaps": observed_context_laps,
        "carriedContextLaps": context_source_counts.get("FOCUS_CONTEXT_CARRIED", 0),
        "sourceCounts": context_source_counts,
        "note": (
            "Each tree step uses a same-lap observed matchup row when available. If a later "
            "counterfactual role has no matching public row, the tree carries the initial "
            "decision context forward and labels it as modelled rather than fabricating telemetry."
        ),
    }
    persistence_by_horizon = []
    for horizon_laps in (1, 2, 3, 5, 6):
        step = path[horizon_laps - 1] if len(path) >= horizon_laps else None
        persistence_by_horizon.append({
            "horizon": horizon_laps,
            "estimatedProbability": step["aheadProbability"] if step else None,
            "observed": actual_lead_laps >= horizon_laps,
        })
    finish_positions = payload.get("finishPositions") or {}
    raw_rule_context = payload.get("ruleContext")
    rule_context = raw_rule_context if isinstance(raw_rule_context, dict) else {}
    # A line position cannot affect the tactical tree until the decision rows
    # carry compatible track-distance coordinates.  Preserve the official
    # context for auditability without treating it as applied race physics.
    rule_application = (
        "TRACK_DISTANCE_ALIGNMENT_REQUIRED"
        if rule_context.get("eventSpecificDataLoaded")
        else "DISCLOSURE_ONLY_UNTIL_EVENT_APPENDIX_LOADED"
    )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "treeVersion": TREE_VERSION,
        "tree": tree,
        "path": path,
        "nodeCount": node_count,
        "leafCount": leaf_count,
        "horizon": horizon,
        "actualLeadLaps": actual_lead_laps,
        "observedLeadLaps": observed_lead_laps,
        "comparisonHorizonLaps": horizon,
        "expectedLeadLaps": expected,
        "actualFinishPosition": finish_positions.get(selected),
        "success": bool(path) and expected > actual_lead_laps,
        "persistenceByHorizon": persistence_by_horizon,
        "socSensitivity": sensitivity_summary,
        "energyCalibrationSensitivity": calibration_sensitivity,
        "opponentPolicy": {
            "id": "CONSERVATIVE_BEST_RESPONSE_HEURISTIC_V1",
            "type": "DETERMINISTIC_CONSERVATIVE_BEST_RESPONSE",
            "description": (
                "At every branch, the opponent selects ATTACK, SAVE, or DELAY that maximises its "
                "immediate pass chance when behind, or its survival chance when ahead, with a small "
                "remaining-SoC preference. This is a model assumption, not the real driver's radio command."
            ),
        },
        "stateProvenance": state_provenance,
        "decisionContext": {
            "raceControl": "CLEAR" if root_context["track_clear"] else "GATED",
            "pitDistorted": root_context["pit_distorted"],
            "attackerTrackStatus": focus.get("attackerTrackStatus"),
            "defenderTrackStatus": focus.get("defenderTrackStatus"),
            "overtakeActionsEnabled": root_context["track_clear"] and not root_context["pit_distorted"],
            "ruleContext": {**rule_context, "application": rule_application},
        },
        "pitTyreContext": pit_tyre_context,
        "selected": selected,
        "defender": initial_defender,
        "energyModelVersion": ENERGY_MODEL_VERSION,
        "regulationEra": regulation_era,
        "energyConfig": {
            "id": energy_config["id"],
            "capacityMj": energy_config["capacityMj"],
            "calibrationStatus": energy_config["calibrationStatus"],
            "source": energy_config["source"],
        },
        "assumptions": [
            "SoC is a modelled surrogate because team battery telemetry is not public.",
            "The selected driver branches into ATTACK, SAVE and DELAY at every lap.",
            "The opponent selects a best response from the same three actions using its remaining SoC.",
            "A pass reverses attacking and defending roles; pit stops change tyre/time context only and never recharge the battery.",
            "Observed pit cycles and tyre state are held fixed as context; this three-action tree does not simulate an alternative BOX strategy.",
            "The tactical horizon is capped at six laps to avoid an unbounded 3^N tree.",
            "Non-green race-control states and pit-distorted exchanges are gated as non-overtake windows.",
            "Success compares estimated persistence with observed consecutive laps ahead, not finish position alone.",
            "The SoC sensitivity panel varies modelled starting energy by 0.50 MJ; it is a robustness check, not private battery telemetry.",
            "The energy-calibration sensitivity panel varies surrogate deployment and harvest by 15%; it is not an FIA limit or measured team energy data.",
            "Later tree laps disclose whether their battle context is observed at that lap or carried from the initial observed decision state.",
        ],
    }


def main() -> None:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    print(json.dumps(build_tree(payload), allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
