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


TREE_VERSION = "tactical-tree.v3"
SCHEMA_VERSION = "replay-state.v3"
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
                  opponent_is_attacker: bool, era: str) -> tuple[str, float]:
    scores = {}
    for action in ACTIONS:
        next_soc, _, _ = transition_soc(
            opponent_soc, action, row, not opponent_is_attacker, era=era)
        chance = attack_chance(row, action, next_soc, our_soc, not opponent_is_attacker)
        # An attacker values a pass; a defender values survival.
        score = chance if opponent_is_attacker else (1.0 - chance)
        score += 0.05 * (next_soc / CAPACITY_MJ)
        scores[action] = score
    selected = max(ACTIONS, key=lambda action: scores[action])
    return selected, round(scores[selected], 4)


def build_tree(payload: dict[str, Any]) -> dict[str, Any]:
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
        forward = row_for(rows, lap, selected, initial_defender) or focus
        reverse = row_for(rows, lap, initial_defender, selected) or {}
        observed = reverse if state["ahead"] else forward
        children = []
        for action in ACTIONS:
            our_soc, deploy, harvest = transition_soc(
                state["ourSoc"], action, observed, state["ahead"], era=regulation_era)
            opponent_action, response_score = best_response(
                reverse if state["ahead"] else forward,
                state["defenderSoc"], state["ourSoc"], state["ahead"], regulation_era,
            )
            opponent_soc, opponent_deploy, opponent_harvest = transition_soc(
                state["defenderSoc"], opponent_action, observed, not state["ahead"],
                era=regulation_era,
            )
            if state["ahead"]:
                # The former attacker now tries to retake the place.  The
                # selected action changes the defence; opponent SoC matters.
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

    _, tree = expand({
        "lap": start_lap,
        "ahead": False,
        "aheadProbability": 0.0,
        "leadLaps": 0.0,
        "ourSoc": initial_our_soc,
        "defenderSoc": initial_defender_soc,
    }, 0)

    path = []
    cursor = tree
    while cursor.get("children"):
        child = next((item for item in cursor["children"] if item.get("best")), cursor["children"][0])
        path.append({key: child[key] for key in (
            "action", "probability", "ourSoc", "defenderSoc", "opponentAction",
            "deployMj", "harvestMj", "opponentDeployMj", "opponentHarvestMj",
            "pitPlan", "leadLaps", "aheadProbability")})
        path[-1]["lap"] = cursor["lap"]
        path[-1]["role"] = cursor["role"]
        cursor = child["next"]

    actual_lead_laps = int(number(focus.get("observedLeadLaps"), -1))
    if actual_lead_laps < 0:
        actual_lead_laps = int(number(focus.get("holdLaps"), 0)) if focus.get("held") else (1 if focus.get("passedNow") else 0)
    expected = path[-1]["leadLaps"] if path else 0.0
    root_context = context(focus)
    persistence_by_horizon = []
    for horizon_laps in (1, 2, 3, 5, 6):
        step = path[horizon_laps - 1] if len(path) >= horizon_laps else None
        persistence_by_horizon.append({
            "horizon": horizon_laps,
            "estimatedProbability": step["aheadProbability"] if step else None,
            "observed": actual_lead_laps >= horizon_laps,
        })
    finish_positions = payload.get("finishPositions") or {}
    return {
        "schemaVersion": SCHEMA_VERSION,
        "treeVersion": TREE_VERSION,
        "tree": tree,
        "path": path,
        "nodeCount": node_count,
        "leafCount": leaf_count,
        "horizon": horizon,
        "actualLeadLaps": actual_lead_laps,
        "expectedLeadLaps": expected,
        "actualFinishPosition": finish_positions.get(selected),
        "success": bool(path) and expected > actual_lead_laps,
        "persistenceByHorizon": persistence_by_horizon,
        "decisionContext": {
            "raceControl": "CLEAR" if root_context["track_clear"] else "GATED",
            "pitDistorted": root_context["pit_distorted"],
            "attackerTrackStatus": focus.get("attackerTrackStatus"),
            "defenderTrackStatus": focus.get("defenderTrackStatus"),
            "overtakeActionsEnabled": root_context["track_clear"] and not root_context["pit_distorted"],
        },
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
            "The tactical horizon is capped at six laps to avoid an unbounded 3^N tree.",
            "Non-green race-control states and pit-distorted exchanges are gated as non-overtake windows.",
            "Success compares estimated persistence with observed consecutive laps ahead, not finish position alone.",
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
