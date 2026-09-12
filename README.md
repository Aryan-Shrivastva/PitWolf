# PitWolf

### Energy and Overtake Intelligence for F1 Strategy

PitWolf is a race-strategy decision engine for Formula 1. It learns how overtakes develop from historical race data, models energy deployment under the 2026 regulations, and replays completed 2026 races to evaluate whether its recommendations could have produced a better result for a selected driver.

> See more. Decide faster. Win on strategy.

## The idea

An overtake is not only a question of speed. It is an energy-budget decision made under changing race conditions. PitWolf combines:

1. **Overtake intelligence** - learns where and how overtakes happen using gaps, speed deltas, tyre state, track zones, pace, and race context.
2. **Energy intelligence** - estimates deployment and recovery from telemetry, then projects it onto the 2026 power-unit rules and track limits.
3. **Decision intelligence** - recommends `ATTACK`, `SAVE`, `DELAY`, or `BOX` with a plain-language reason.

## Retrospective 2026 race replay

PitWolf does not alter real races. Completed 2026 races are treated as observed race histories. The system replays them from the perspective of a selected driver and evaluates counterfactual decisions at historical decision points.

The replay uses the actual grid, driver, positions, gaps, telemetry, tyres, pit stops, track zones, safety-car and flag events where available, and the applicable 2026 rules. At each point, the model recommends an action and tracks the resulting position, energy state, overtake outcome, and durability of any gain.

For example, if a driver finished fourth in reality, but the replay recommends a feasible overtake and reaches third while remaining ahead for two or three laps or until the finish, that is evidence of a successful strategy recommendation. The system must show the reason: gap, pace delta, tyre condition, energy budget, target zone, and race context.

These results are retrospective counterfactual estimates, not claims that the real driver definitely would have achieved the replayed result.

## Training and evaluation

### Historical training data: 2018-2025

The model is trained on completed races from 2018 through 2025. Decision examples include driver and opponent, grid and current position, gap, speed-trap delta, track zone, tyre compound and age, recent pace, energy context where available or derivable, safety-car/flag state, and the eventual overtake outcome.

Outcomes distinguish immediate success from durable success: a pass reversed on the next lap is not treated like one that survives two or three laps or lasts to the finish. Unsuccessful attempts and lost positions are also labeled.

### Held-out 2026 replay set

Completed 2026 races remain separate from training. PitWolf evaluates each recommendation using only information available at that moment, preventing future race information from leaking into earlier decisions.

It compares the real outcome with the replayed outcome, including action quality, position durability, energy used or saved, overtake feasibility, pit decisions, final finishing position, and strategic improvement.

## Decision model

- **`ATTACK`** - deploy energy when the gap, zone, pace delta, tyre state, and remaining budget support a likely overtake.
- **`SAVE`** - preserve energy when the current opportunity is weak or a stronger one is likely soon.
- **`DELAY`** - avoid a low-probability or rule-risky attempt and wait for recovery or a better zone.
- **`BOX`** - pit when tyre state, traffic, race position, and projected strategy favor stopping now.

Every action should include an explanation, such as: “SAVE - the current zone has low overtake probability; hold energy for the next braking zone.”

## Energy model

The energy layer is grounded in telemetry rather than an unexplained conversion. A simplified instantaneous power estimate is:

```text
P(t) = m * a(t) * v(t)
```

The system projects computed demand onto the 2026 power-unit split, applies the relevant deployment rules, accumulates energy across the lap, and checks it against the published per-track ceiling. Where public data does not expose battery state or complete power-unit measurements, assumptions must be documented clearly. Physical plausibility must not be presented as exact battery telemetry.

When the remaining battery parts are implemented, overtake and save scoring must follow **[BATTERY.md](BATTERY.md)**: selected-driver modelled SoC plus a `0…1` randomness term raise or lower overtake probability, then ATTACK / SAVE use that adjusted percentage.

## Validation

PitWolf should be evaluated on:

1. **Prediction quality** - precision, recall, F1 score, and ROC-AUC for overtake feasibility and outcome classes.
2. **Durability quality** - how often predicted gains survive 1, 2, or 3 laps.
3. **Strategy quality** - final-position improvement, successful attacks, avoided failed attacks, and pit-stop decisions.
4. **Physical and regulatory plausibility** - zone alignment, energy ceilings, cross-track behavior, and 2026 rule compliance.

Because complete public battery-state data is not available for every race, the project must distinguish numerical model accuracy from physical plausibility and disclose its assumptions.

## Data and methodology references

The intended sources are FastF1 session/lap/telemetry/position data, OpenF1 and compatible open race-data sources, FIA 2026 regulations, and established telemetry feature-engineering approaches.

```text
Training:   2018-2025
Evaluation: completed 2026 races
```

No 2026 race outcome should train a recommendation that is later evaluated on that same race.

## Product experience

The strategy cockpit should provide selected-driver controls, a track map with overtake/recovery/deployment zones, gap and pace deltas, tyre and energy state, driver-versus-opponent telemetry overlays, recommendation confidence, plain-language explanations, a replay timeline, and a comparison between real and counterfactual results.

## Repository structure

```text
PitWolf/
├── frontend/                    # React + Vite strategy cockpit interface
├── backend/                     # Node.js API and decision services
├── analytics/                   # Data preparation, modeling, and evaluation
├── supabase/                    # Database schema and backend configuration
├── BACKEND_HANDOFF.md           # Backend implementation notes
├── VALIDATION.md                # Validation requirements and checks
├── database_plan.md             # Data model planning
└── package.json                 # Root workspace commands
```

## Run locally

```bash
npm install
npm run dev
```

In a second terminal:

```bash
npm run server
```

Useful checks:

```bash
npm run build
node --check backend/server.mjs
```

The frontend runs on Vite and the backend listens on port `8787` by default.

## Implementation roadmap

1. Ingest and normalize 2018-2025 race history.
2. Build a leakage-safe overtake and durability dataset.
3. Add telemetry, track-zone, tyre, gap, and energy features.
4. Implement the regulation-aware 2026 energy projection model.
5. Train and validate overtake-feasibility and outcome models.
6. Build a deterministic retrospective 2026 replay engine.
7. Add driver selection, race-event controls, replay timelines, and explanations.
8. Score recommendations against real outcomes and durable position gains.

## Principles

- Use real race data wherever possible.
- Keep 2026 evaluation races held out from training.
- Never hide assumptions behind a black-box energy conversion.
- Treat rule compliance as a hard constraint.
- Score durable race outcomes, not only immediate overtakes.
- Explain every recommendation in driver- and engineer-friendly language.
- Clearly label counterfactual replay results as estimates.
