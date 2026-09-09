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
2. Select one driver. Resolve the car immediately ahead as the attack target
   and the car immediately behind as the defence threat from the recorded
   classification at that instant.
3. Load their actual observed state at that instant: track distance, gap,
   relative speed, tyres, lap context, modelled state of charge (SoC), and
   race-control context.
4. Show all cars on the track. Initially, non-selected cars remain recorded
   visual "ghost" cars so the simulator does not pretend to model the entire
   field.
5. While the observed replay is running, choose a lap in `JUMP / BRANCH LAP`.
   Press `JUMP` to freeze the public state at that lap and begin a separate
   PitWolf branch from it.
6. The branch receives no later recorded decision rows. At each simulated lap,
   PitWolf evaluates ATTACK, SAVE, and DELAY for the selected car and applies a
   conservative response for its starting attack target.
7. Continue the branch to the chequered flag and compare its modelled
   selected-pair order with the observed final pair order from the real race.

The current version is a **future-blind, two-car race-to-flag rollout**. It
does not claim to simulate every team strategy, radio call, tyre choice,
weather shift, pit stop, retirement, race-control event, or the rest of the
race grid. Therefore it must not label a projected pair result as a full-grid
counterfactual finishing position.

## Required screen areas

### 1. Race and branch controls

- season, race, session;
- recorded lap/time selector and a `JUMP` action that starts the branch;
- one selected driver, with automatic attacking and defending relationships;
- branch mode: `OBSERVED REPLAY` or `PITWOLF BRANCH`;
- Play, Pause, Restart, speed, and reset controls;
- a branch status showing the selected start lap and flag lap.

### 2. Animated track view

- circuit map with all participants at recorded positions;
- highlighted selected driver, attack target, and defence threat;
- direction of travel and current lap/time;
- visible zone/detection/activation markers only where their source and
  track-distance alignment are known;
- optional camera focus on the battle pair;
- a visible distinction between recorded ghost cars and modelled pair markers.

### 3. Live battle board

For the selected pair, show:

- running position and gap;
- selected driver and attack-target modelled SoC in MJ;
- tyre/lap context and race-control state;
- next eligible decision window;
- action probabilities for ATTACK, SAVE, and DELAY;
- current selected action and conservative opponent response;
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

## Current implementation status

- **Implemented:** the former static Las Vegas screen is now a selectable
  recorded-race replay. It uses cached public FastF1 circuit/position streams,
  a shared session clock, all recorded cars as visual ghosts, replay controls,
  and an exact-lap selected battle panel.
- **Implemented:** for an extracted pair and exact lap, the screen scores the
  existing action model and runs the existing bounded two-car strategy tree.
  It shows action probabilities, modelled SoC, the conservative opponent
  response assumption, and a 1–6 lap path.
- **Implemented:** `OBSERVED` and `PITWOLF BRANCH` views. Select a lap and
  press `JUMP` to create a future-blind, two-car race-to-flag rollout. The
  branch receives the selected public state only; subsequent policy steps use
  carried model state rather than later recorded decision rows.
- **Implemented in part:** the flag comparison shows modelled selected-pair
  order and probability, observed selected-pair order, real selected-driver
  finish, modelled energy-window status, and the FIA command gate. It
  explicitly withholds a modelled full-grid finishing position.
- **Deliberately withheld:** Always SAVE, fixed-deployment, and gap-only
  baseline results remain unavailable until each baseline has been evaluated
  with the same held-out protocol. The screen says this explicitly rather than
  presenting unvalidated comparison data.
- **Still deliberately absent:** a modelled car is not moved through a
  fabricated pass trajectory. The model supplies probabilities, not
  counterfactual GPS. No marker is treated as an FIA eligibility line, and no
  result is shown as a live command. Until event-specific FIA lines are
  distance-aligned, the branch stays `ANALYSIS ONLY` and the field remains
  recorded replay data.

### Phase 1 — recorded visual replay

Build the track player from cached position data: selectors, driver board,
playback, speed controls, lap/time readout, and recorded ghost cars. No
counterfactual decision is shown yet.

### Phase 2 — selected battle state

Allow selection of a detected attacker/defender decision point. Display the
real state at that instant and visually identify the pair on the circuit.

### Phase 3 — future-blind PitWolf race branch

Freeze one real two-car state at `JUMP`, exclude all later recorded decision
rows, and roll the existing action/energy logic to the flag. Overlay the
modelled pair's state over recorded ghosts and show the action/reaction event
log. Do not invent counterfactual car movement; only animate a pass when a
calibrated trajectory model exists.

### Phase 4 — outcome and baseline comparison

The observed/modelled selected-pair flag comparison, energy-window checks, and
FIA command gate are now shown. Next, implement and evaluate the Always SAVE,
fixed-deployment, and gap-only baselines before displaying policy comparison.
Build a full-field pace, tyre, BOX, traffic, retirement, and race-control model
before making any better/worse/same full-race finishing-position claim.

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
