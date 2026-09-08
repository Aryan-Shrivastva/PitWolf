# PitWolf visual energy and overtake simulation

## Purpose

PitWolf will later add a dedicated **Simulation / Replay Lab**. It is not a
generic race replay and it is not a live driver command system. It is a
visible, interactive way to inspect how PitWolf's energy and overtake models
behave when they are given a real historical race state.

The user selects a completed race, a moment in that race, and a driver battle.
The screen starts from the observed state at that moment and then shows a
bounded, modelled branch for the selected battle pair. It makes model inputs,
actions, uncertainty, and the real observed outcome visible in one place.

## Visual reference

The interaction language is informed by Formula Telemetry's session track
visualization: <https://formula-telemetry.com/sessions>.

Observed reference features:

- race/year/session selectors and a selectable driver list;
- a 2D track visualization with moving car markers;
- lap, elapsed-time, and frame readouts;
- a right-side running-order and gap board;
- Play, Pause, Restart, speed, zoom, reset, and frame-scrubbing controls.

PitWolf must build its own interface and visual system. The reference is used
only for the interaction pattern: a clear moving circuit, a current-race
readout, a driver board, and replay controls.

## Core experience

1. Select season, race, session, and a recorded lap/time.
2. Select an attacker and defender from a detected close battle.
3. Load their actual observed state at that instant: track distance, gap,
   relative speed, tyres, lap context, modelled state of charge (SoC), and
   race-control context.
4. Show all cars on the track. Initially, non-selected cars remain recorded
   visual "ghost" cars so the simulator does not pretend to model the entire
   field.
5. Press Play to advance the branch. At each valid decision window, PitWolf
   evaluates ATTACK, SAVE, and DELAY for the selected attacker and computes a
   conservative defender response.
6. Animate changes to the selected pair's gap, energy state, and position.
7. Show whether a simulated pass occurs and whether the projected gained
   position is retained over the configured horizon.
8. Show the equivalent observed outcome from the real race beside the branch.

The first version is a **two-car, bounded-horizon simulation**. It does not
claim to simulate every team strategy, radio call, tyre choice, weather shift,
or the rest of the race grid.

## Required screen areas

### 1. Race and branch controls

- season, race, session;
- recorded lap/time and a scrubber;
- attacker and defender;
- branch mode: `OBSERVED REPLAY` or `PITWOLF BRANCH`;
- Play, Pause, Restart, speed, and reset controls;
- horizon selector, initially 1–6 laps.

### 2. Animated track view

- circuit map with all participants at recorded positions;
- highlighted attacker and defender;
- direction of travel and current lap/time;
- visible zone/detection/activation markers only where their source and
  track-distance alignment are known;
- optional camera focus on the battle pair;
- a visible distinction between recorded ghost cars and modelled pair markers.

### 3. Live battle board

For the selected pair, show:

- running position and gap;
- attacker and defender modelled SoC in MJ;
- tyre/lap context and race-control state;
- next eligible decision window;
- action probabilities for ATTACK, SAVE, and DELAY;
- current selected action and defender response;
- pass probability and durable-position probability;
- uncertainty and FIA command-gate status.

The normal running-order board should show the selected drivers and nearby
cars, including observed gaps where they are still recorded.

### 4. Decision event log

Every model step must append a readable event, for example:

```text
L18 · detection state assessed · gap 0.74 s
L18 · ATTACK 62% / DELAY 24% / SAVE 14%
L18 · energy state 2.10 MJ → 1.36 MJ
L19 · simulated pass transition: 62% probability
L22 · projected position durability: 54%
```

This is an explanation trace, not reconstructed team radio.

### 5. Observed-versus-model comparison

For the same horizon, compare:

- observed real race position and gap movement;
- PitWolf branch's projected position/gap/SoC;
- immediate-pass and durability estimates;
- baseline policies: Always SAVE, fixed deployment, and gap-only;
- whether modelled energy remained inside its 0–4 MJ operating window;
- whether the FIA live-command gate was open.

The page must clearly state that a different simulated branch cannot prove the
real race would have changed.

## How the simulation decides a pass

Choosing ATTACK must not automatically animate an overtake. A pass transition
may occur only when the modelled state reaches a valid decision window and the
separate, calibrated pass estimate supports it. The display must show the
probability and uncertainty rather than presenting the pass as a fact.

After a simulated pass, the durability component estimates whether that gained
position remains held at each selected horizon. The opponent's reaction is an
explicit conservative model assumption, never a claim about a real driver's
radio instruction or intended strategy.

## Data that already exists

PitWolf already has the foundations required for a first version:

- cached completed-race timing and position data;
- track-map and telemetry-derived position data for supported sessions;
- extracted close-battle decision points;
- action-model probabilities for ATTACK, SAVE, and DELAY;
- modelled per-lap energy/SoC traces;
- observed position summaries and durable-pass labels;
- bounded strategy replay logic with attacker/defender energy context;
- held-out validation and transparent baseline metrics;
- 2026 user-supplied visual zone evidence for 11 races, stored separately from
  official FIA event data.

## Data and gates still required

The page can be built progressively, but it must never elevate incomplete data
into a live command.

### Exact FIA event data

For a genuinely event-specific 2026 Overtake Mode evaluation, PitWolf needs
the FIA event-specific source for the relevant race's detection and activation
lines, zone definitions, and applicable race-control constraints.

### Track-distance alignment

Those official markers must be mapped to track distance before the simulator
can say a car is at an eligible zone. A turn description or user-supplied
illustration may be displayed as visual evidence but must not be converted into
invented coordinates or used to enable a live-command gate.

### Better outcome effects

The immediate-pass and durability components are currently diagnostic model
components. Their calibration, uncertainty, and causal limitations must remain
visible until stronger evaluation supports their use in the displayed branch.

### Separate BOX model

Pit/tyre alternatives remain a separate BOX model. The initial two-car
simulation holds pit planning fixed or displays observed pit context; it does
not silently choose pit stops. BOX can be integrated only after its own
validation is strong enough.

## FIA and provenance rules

- Every value is labelled as recorded, derived, modelled, or assumed.
- SoC is modelled from public telemetry-derived inputs, not private team
  battery telemetry.
- Historical replay data does not establish a 2026 Overtake Mode rule.
- Missing official event data results in `ANALYSIS ONLY`, not a live command.
- Zone markers appear only with an accompanying source status.
- Non-green or otherwise restricted race-control states block a command
  interpretation.
- No feature may use a future lap to choose an action at the starting state.

## Phased delivery

### Phase 1 — recorded visual replay

Build the track player from cached position data: selectors, driver board,
playback, speed controls, lap/time readout, and recorded ghost cars. No
counterfactual decision is shown yet.

### Phase 2 — selected battle state

Allow selection of a detected attacker/defender decision point. Display the
real state at that instant and visually identify the pair on the circuit.

### Phase 3 — bounded PitWolf branch

Run the existing attacker/defender action and energy logic through a short
horizon. Overlay the modelled pair's branch over the recorded replay and show
the action/reaction/event log.

### Phase 4 — outcome and baseline comparison

Add observed outcome, pass/durability probability, energy-window checks, and
the Always SAVE/fixed-deployment/gap-only comparisons.

### Phase 5 — FIA zone fidelity

When official event appendices and track-distance references are available,
replace generic model decision windows with exact event-aligned
detection/activation/zone checks. Keep any incomplete race in analysis-only
mode.

### Phase 6 — BOX integration and wider field modelling

Only after independent validation, introduce pit/tyre alternatives and consider
expanding beyond the selected two-car pair. This is not part of the first
simulation version.

## Definition of a credible first release

The first release is complete when a user can choose a cached race and a real
battle point, press Play, watch a clearly labelled two-car PitWolf branch,
inspect the model's state/action/reaction/SoC trace, and compare it against the
recorded outcome and baselines. It must preserve the FIA analysis-only gate
where official event-specific data is unavailable.
