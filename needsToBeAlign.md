# What Still Needs to Be Aligned

This document records the agreed technical and methodological direction for
PitWolf. It is a planning and review document. It does not claim that the
items below are already implemented.

## Initial implementation status

The first alignment slice is now implemented in the working tree:

- a backend `tactical-tree.v1` replay evaluates ATTACK, SAVE, and DELAY at
  every lap in a bounded six-lap tree;
- both cars carry modelled SoC, the opponent chooses a best response, and a
  successful pass reverses the attacking and defending roles;
- the frontend Strategy view consumes the backend tree and requests the
  opponent's energy trace when it is available;
- race-speed and energy-surrogate features now use past-only information at a
  lap-start cutoff, and model reports include temporal-split metadata and an
  always-SAVE baseline;
- the decision-point extractor now emits `decision-point.v4` records with
  alignment, grid, tyre/stint, track-status, weather, missingness, and
  exclusion metadata; shared-clock lap synchronization excludes lapping /
  backmarker events from training and old cache versions are automatically
  rebuilt;
- pit context no longer adds battery energy to the race trace or surrogate.
- the action-level transition is now versioned in one shared backend module and
  is used by both the historical surrogate and tactical replay.
- stale v1 decision and energy caches are no longer accepted by the replay
  route; requested races are rebuilt under the current schema before scoring;
- replay pass probability now penalises depleted SoC, and observed consecutive
  laps ahead are compared against estimated persistence rather than using a
  zero-length label horizon.
- the shared energy transition now has deterministic capacity, clipping, and
  attacker/defender SoC sensitivity validation covering depleted, partial, and
  full-charge states.
- the shared transition now records explicit `2018_2025` and `2026` era
  configurations plus `energy-transition.v2` provenance in replay responses;
  the current action calibration remains explicitly provisional because public
  data does not expose team battery deployment.
- action deployment is clipped to charge actually available; ATTACK and DELAY
  are blocked below their minimum energy thresholds, while SAVE can rebuild
  reserve across the horizon. This applies globally across races and drivers.
- the Strategy tab can expand the winning path and inspect all three action
  branches per replay lap, including opponent response and both SoC values.
- replay results are keyed to the selected race/lap, so a pending request can
  no longer briefly display a previous lap's fallback tree.
- the v2 decision records now include public-data proxies for pack traffic,
  slipstream opportunity, dirty-air exposure, and tyre degradation; the replay
  consumes these proxies when adjusting pass and repass probability.
- decision-point.v5 now enforces the LAP_START cutoff for speed delta, lap
  time, tyre age, and degradation by using only the previous completed lap;
  stale v4 caches are rejected, and `validate_causal_features.py` verifies
  future-lap invariance for the energy and degradation features.
- the final holdout retrain now uses 2018–2025 for training and completed 2026
  races for evaluation: 30,431 training rows and 2,072 unseen test rows across
  12 races. The report now includes race-level bootstrap uncertainty,
  multiclass calibration diagnostics, and an explicit model-vs-always-SAVE
  comparison; the Overtake page displays these results and does not call the
  classifier successful when it loses to the baseline.

This is a foundation, not completion of the full specification. The real-race
replay state, exact on-track zone configuration, pit/tyre model, safety-car
state, full two-car telemetry alignment, and completed unseen-season evaluation
remain subsequent work items below.

## Executive verdict

The current project is a useful prototype, but it is not yet a complete
implementation of the intended research question.

The parts that already point in the right direction are the FastF1 data
pipeline, real-vs-modelled provenance labels, tyre and position context, the
ATTACK/SAVE/DELAY vocabulary, the dual-driver energy idea, and the first
recursive strategy-tree interface.

The main gap is that the current system is still closer to an overtake
classification demo than to a deterministic, leakage-safe, two-car
retrospective decision simulator. The final system must answer this narrower
question:

> Given the exact information available to a selected driver at a real
> decision point, which action—ATTACK, SAVE, or DELAY—would maximise the number
> of subsequent laps that the driver remains ahead, while accounting for the
> opponent's ability to respond, tyre state, pit opportunities, track zones,
> flags, and both cars' modelled energy states?

The result is a retrospective counterfactual estimate. It must never be
described as proof that the real race would definitely have changed.

## 1. What currently aligns with the goal

These are the foundations to preserve:

- FastF1 is the primary source for public timing, laps, telemetry, tyre,
  position, results, and circuit information.
- The project separates REAL, DERIVED, and MODELLED/SIMULATED values.
- Decision examples distinguish immediate passes from passes that survive a
  persistence horizon.
- The model uses ATTACK, SAVE, and DELAY as the tactical choices.
- The selected driver and the car being challenged are both represented.
- The recursive tree changes the selected driver's role after a pass: the
  attacker becomes the defender and the former defender can attempt a repass.
- The interface reports the real finish position beside the modelled result.
- Large FastF1 caches and generated models are kept out of Git and recreated
  locally.
- A temporal evaluation split is documented instead of relying only on random
  row splits.

These foundations are valuable, but they must be moved from partly hard-coded
frontend behaviour into one reproducible backend/data contract.

## 2. The problem must be split into three separate tasks

The project currently risks mixing three different questions. They need
separate datasets, outputs, and metrics.

### A. Descriptive event detection

Find real on-track situations from race history:

- a driver is close enough to the car ahead to create a realistic battle;
- a pass is attempted or a position is exchanged;
- the exchange is not caused only by a pit stop, lapping event, penalty, DNF,
  safety-car procedure, or missing timing data;
- the position is then tracked for several future laps.

This task produces auditable decision-point records. It does not yet decide
what the driver should have done.

### B. Outcome-optimal action scoring

For each detected point, use only information available at that point as model
features. Use later race history only to create the retrospective target:

- `ATTACK` if an immediate pass occurred and remained ahead for the selected
  persistence horizon;
- `DELAY` if an immediate pass did not occur but a durable pass became
  available within the horizon;
- `SAVE` if neither attack nor delay produced a durable gain in the observed
  window.

These are outcome-derived strategy labels, not records of what the driver
actually commanded. The UI and reports must call them `outcome-optimal`
labels, not actual driver decisions.

### C. Counterfactual tactical replay

At a real 2026 decision point, branch the state into the three actions and
roll the state forward. The replay asks whether an alternative action could
have kept the selected driver ahead for longer. This is different from asking
whether the model guessed the driver's real action.

The product must show both scores separately:

1. classification quality on held-out decision points; and
2. simulated persistence improvement against the actual race.

Neither score should be used as a substitute for the other.

## 3. Data that still needs to be aligned

### 3.1 Build a single canonical event schema

Every race and decision point should have stable identifiers:

```text
season, round, event, session, driver, opponent, lap, distance_m, zone_id
```

Each record should also contain:

- grid position, current classified position, and position of every nearby car;
- gap, closing rate, relative speed, sector times, lap times, and speed-trap
  values;
- throttle, brake, speed, gear, RPM, DRS/override availability, and track
  location where FastF1 supplies them;
- tyre compound, tyre age, stint number, pit-in/pit-out status, and pit loss;
- race-control state: safety car, virtual safety car, yellow flags, red flags,
  penalties, lapped status, and DNF state where available;
- weather and track conditions where available;
- circuit length, corners, braking zones, straight zones, and a stable
  track-coordinate system;
- real finish position and future position history for scoring;
- data source, extraction version, missingness flags, and cache-generation
  metadata.

The current extractor focuses on lap-level position and a maximum speed value.
That is a good starting point, but it is not yet the complete state needed by
the replay. A single row also cannot represent all cars that may affect a
selected driver's traffic, pit, or safety-car outcome.

### 3.2 Detect real on-track overtakes correctly

Position swaps must be classified before they become training labels. The
pipeline needs explicit exclusion reasons for:

- pit-cycle swaps;
- cars entering or leaving the timing classification;
- lapped-car interactions;
- penalties and steward changes;
- safety-car or red-flag order changes;
- missing or invalid telemetry;
- a swap that is only a timing artefact between lap crossings.

For every retained event, store the preceding state, the first lap in which
the pass is visible, the first lap in which it is held, and the identity of the
car that can attempt a repass. Do not silently drop ambiguous events.

### 3.3 Align time and distance across both cars

The attacker and defender must be compared at the same race moment, not only
by joining two independent lap summaries. The preferred hierarchy is:

1. same timestamp or track distance;
2. same sector with interpolation;
3. same lap with an explicit lower-confidence flag.

The alignment method and its tolerance must be recorded in the exported data.

### 3.4 Replace the synthetic race simulator

The current `frontend/src/components/RaceSimView.jsx` still uses generated lap
times, fixed Las Vegas zones, and synthetic gap dynamics. It is not yet a
replay of a real 2026 race and cannot defensibly produce an alternative final
position.

The FastF1 dashboard and this synthetic simulator are currently separate
systems. The final replay must initialise from the actual selected race, grid,
driver timeline, telemetry, pit events, flags, safety-car timing, and track
geometry. Any remaining synthetic value must be visibly labelled as modelled
and must not be mixed with the observed race timeline.

## 4. Training and validation must be leakage-safe

### 4.1 Recommended experiment split

The main product experiment should use:

```text
Historical training: 2018-2025
Final evaluation:    completed 2026 races only
```

No result, future position, future energy estimate, or post-race information
from a 2026 race may enter the features or model fitting when that same race is
being evaluated.

For a stricter research report, also run a secondary experiment:

```text
Training:       2018-2023
Development:    2024-2025
Final holdout:  completed 2026 races
```

This shows whether the conclusion survives a longer unseen period. The chosen
split must be printed in every model report.

### 4.2 Split by race, not by adjacent rows

Rows from the same race are strongly correlated. A random row split can put
neighbouring laps from one battle in both training and test sets and produce
an inflated score. Use season/race-grouped temporal splits and report the
number of races, decision points, and drivers in every partition.

### 4.3 Features versus labels

Features may use only the information available at the decision point. Future
position, future gap, whether the pass eventually succeeded, and future tyre
or energy state may be used only to construct the target and evaluation
metrics. This rule should be enforced in the feature-building code and tested
automatically.

The previous v4 cache format had concrete leakage risks. The extractor fix is
now implemented in decision-point.v5:

- the row-level `raceMeanSpeedKph` is now an expanding past-only mean; the
  complete-race summary is stored separately as
  `raceMeanSpeedKphFinalObserved` and is not a model feature;
- the energy reference uses only completed prior-lap values;
- speed delta, lap time, tyre age, and degradation use the previous completed
  lap rather than a complete current lap.

The v5 historical rebuild and the final 2026 holdout rebuild are complete. The
current report is reproducible under the strict split, but the model is not
yet a winning policy: its 2026 accuracy is 58.25% versus 69.79% for
always-SAVE. This is an evaluation finding, not a reason to hide or tune on
the holdout; the next modelling work should improve the component models and
replay fidelity using training/development data only.

### 4.4 Baselines and uncertainty

Every result must be compared with at least:

- always `SAVE`;
- a gap-only rule;
- the current fixed rule-based engine;
- a non-recursive model that does not use the tactical tree.

Report sample counts, class balance, precision, recall, F1, calibration, and
confidence intervals. A percentage from four or five events is a demonstration,
not evidence of generalisation.

The current report must not be presented as evidence of success without this
comparison. The review snapshot showed approximately 56.26% test accuracy on
2,538 rows, while an always-`SAVE` baseline was approximately 70.72% and macro
F1 was approximately 0.426. These numbers may change after a leakage-safe
rebuild, but the classifier must beat meaningful baselines before it is called
a successful decision engine.

## 5. Energy intelligence needs a more honest architecture

### 5.1 Keep the provenance boundary

Public FastF1/OpenF1 data does not expose the team's true battery state of
charge, deployment map, or complete power-unit controls. Therefore:

- measured telemetry remains REAL;
- calculations from measured telemetry remain DERIVED;
- battery state, deployment budget, harvest, and response capability remain
  MODELLED/SIMULATED;
- every report must state the assumptions and model version.

The system must not present a surrogate SoC value as measured battery data.

### 5.2 Replace isolated energy heuristics with a shared state transition

The current tree contains fixed action costs and a separate lap-time SoC
profile. These are useful prototypes, but the final replay needs one shared
energy transition function used by:

- the historical feature generator;
- the standalone predictor;
- the recursive tree;
- the real-race replay;
- the evaluation report.

At minimum, the transition must account for:

```text
SoC_next = SoC_now - deployment(action, zone, speed, throttle)
                   + harvest(braking, zone, speed, brake)
                   + direct_supply_or_recovery_terms
```

Then apply capacity, per-lap, per-zone, and regulation constraints. The
parameters must be versioned and calibrated without using the final held-out
race outcomes.

### 5.3 Model both cars, not only the selected driver

After an overtake, the former attacker becomes vulnerable and the former
defender retains whatever energy the model assigns to it. The opponent must be
able to:

- use remaining energy to defend before the pass;
- spend energy to attempt a repass after losing the position;
- harvest and recover over subsequent braking zones;
- be affected by tyres, traffic, pits, flags, and the same regulation limits.

The opponent's SoC must not be reset, held constant, or represented only by a
generic reserve multiplier. Both cars need state, transitions, and provenance.

### 5.4 Separate regulation facts from calibration assumptions

The 2018-2025 power-unit rules and the 2026 regulations are different model
eras. The code must not silently apply old constants to 2026. Maintain:

- a cited regulation configuration per season/era;
- a separate calibration configuration;
- a model version and configuration hash in every replay;
- unit tests for capacity, deployment, harvest, and clipping limits.

If a 2026 rule or public parameter is uncertain, mark it as an assumption and
run a sensitivity analysis rather than presenting it as fact.

### 5.5 Separate Straight Mode from Overtake Mode

The 2026 rules must be represented as separate event-specific concepts:

- Straight Mode is an active-aero configuration available in designated areas.
- Overtake Mode is an attack-specific energy advantage governed by the
  detection gap and detection line.
- Boost may be used offensively or defensively when energy and restrictions
  allow it.

Detection gaps, detection lines, activation lines, per-circuit limits, and
flag-related restrictions can vary by event. A permanent rule such as
`gap <= 1.0 s` is not sufficient for every circuit. These values must come from
the event configuration and be stored with the replay.

Most importantly, a pit stop must never be treated as an energy reset. A pit
stop changes tyres and costs time; it does not recharge the battery during a
race. The replay must model pit loss, tyre warm-up, traffic, and position
consequences without adding battery energy merely because a pit occurred.

## 6. The recursive decision tree must represent the real objective

### 6.1 Branching rule

At every eligible decision point, evaluate all three tactical branches:

```text
ATTACK
SAVE
DELAY
```

Every child becomes the parent of three new children at the next eligible lap
or tactical zone. The selected action is the branch with the best expected
outcome under the configured objective—not simply the branch with the highest
one-step pass probability.

### 6.2 Reverse the roles after every pass

The state transition must explicitly reverse the battle:

```text
before pass: selected driver = attacker, opponent = defender
after pass:  selected driver = defender, opponent = attacker
```

The opponent can then attack with its own remaining energy. The selected
driver's success is therefore measured by how long the lead survives, not only
by whether the first pass happened.

### 6.3 Objective function

The primary objective is:

```text
maximise expected laps ahead after the pass
```

Report these supporting metrics:

- probability of gaining the position;
- probability of remaining ahead for 1, 2, 3, 5, and 6 laps;
- expected laps ahead over the configured horizon;
- probability of remaining ahead to the chequered flag;
- energy spent, harvested, and remaining for both cars;
- pit action and tyre consequences;
- final-position delta as a secondary metric.

Finishing position or winning the race is not the primary success definition
for this feature. A driver who gains a place and holds it for six laps can be a
successful tactical result even if the final race position does not change.

### 6.4 Treat opponent behaviour explicitly

The first practical version may use a calibrated opponent-response model. It
must state which of these it uses:

- a fixed observed opponent policy;
- a probabilistic response model;
- a conservative best-response assumption;
- a minimax or robust policy.

The tree should show the assumption. A single hidden formula that makes every
opponent respond identically is not sufficient.

### 6.5 Handle pit stops without corrupting the three-action tree

The tactical tree should keep ATTACK/SAVE/DELAY as its three overtake choices.
Pit strategy can be handled as a separate constrained transition or a clearly
marked `BOX` decision at a pit window. If `BOX` becomes a fourth branch, the
report must say so instead of claiming that every node has only three choices.

Pit loss, tyre warm-up, traffic, undercut/overcut effect, and the selected
driver's position after the stop must be represented before a pit branch can be
called realistic.

## 7. Use separate predictive components before combining them in the tree

One action classifier should not be expected to learn every causal quantity
needed by the replay. The recommended architecture is:

1. **Pass-probability model** — probability of gaining the position under an
   attack at the current state.
2. **Hold/repass model** — probability of staying ahead for 1, 2, 3, 5, and 6
   laps, including the opponent's response.
3. **Lap-time impact model** — time cost of attacking, defending, saving, or
   delaying.
4. **Energy model** — both drivers' SoC, deployment, harvesting, clipping, and
   uncertainty.
5. **Pit/tyre model** — pit loss, compound effect, tyre degradation, warm-up,
   and traffic after rejoining.

The recursive planner should combine these outputs. Keeping them separate
makes each part testable and prevents a single label from hiding whether the
failure came from pass probability, energy, or lead durability.

## 8. The counterfactual replay needs clear boundaries

The replay may use the real race as an observed environment, but it cannot
claim that changing one action definitely changes every later event. A safer
design is:

1. initialise the replay with the real state at a decision point;
2. branch only the selected tactical action;
3. propagate both cars through a documented transition model;
4. keep unrelated observed events fixed where justified;
5. report uncertainty when later traffic, tyre, pit, or safety-car outcomes
   cannot be inferred reliably;
6. compare the modelled persistence outcome with the real observed outcome.

Use language such as:

```text
Under the stated replay assumptions, ATTACK is estimated to preserve the
position for 4.2 laps versus 2 real laps.
```

Do not use language such as:

```text
The driver definitely would have finished third.
```

## 9. Current code areas that need alignment

### `backend/scripts/extract_decision_points.py`

Keep it as the source of auditable decision-point records, but extend it to
export the full at-the-time state, robust event exclusions, both-driver
energy inputs, track-zone identity, race-control context, and future position
outcomes at multiple horizons.

### `backend/scripts/train_overtake_model.py`

Keep the temporal split, but enforce race-grouped splits, feature/label
separation, baseline metrics, calibration, and a report that distinguishes
outcome-label agreement from replay improvement.

### `backend/scripts/predict_overtake.py`

Keep the model endpoint, but make its feature schema versioned and shared with
the extractor. It must reject or clearly mark missing energy, zone, opponent,
and regulation inputs instead of silently filling important state with a
neutral value.

### `backend/scripts/energy_surrogate.py`

Keep the surrogate as an explicitly modelled fallback. Calibrate it against
historical lap-time/telemetry patterns, expose uncertainty, and make it produce
state for both cars. It must be used by the same transition code as the tree.

### `frontend/src/components/DecisionTabs.jsx`

The current recursive tree is an important visual prototype, but its hard-coded
action costs and frontend-only SoC evolution must not remain the authoritative
simulation. Move tree calculation to a deterministic backend service and let
the frontend render returned nodes, assumptions, probabilities, and provenance.

The tree must also not be described as a full-race tree. A naïve 50-lap
three-choice expansion has approximately `3^50` action sequences before
opponent responses and pit choices are added. The current six-lap tactical
horizon is reasonable for persistence analysis, but longer planning should use
receding-horizon control, memoisation/dynamic programming, beam search, or
Monte Carlo tree search.

### `frontend/src/components/RaceSimView.jsx`

Replace the generated Las Vegas lap-time/gap simulation with the backend replay
output. It must consume the same race state, track zones, events, energy
transition, and model version as the analysis tabs. There must be one canonical
replay, not a real-data dashboard beside an unrelated synthetic animation.

### `frontend/src/lib/decisionEngine.js` and `frontend/src/lib/energyModel.js`

There are currently two conceptual decision paths: a fixed rule engine and a
Random Forest/tree path. Define their roles explicitly:

- rule engine: transparent baseline and fallback;
- trained classifier: held-out action/outcome model;
- replay engine: sequential counterfactual evaluator.

They must not silently disagree while the UI labels both as the same model.
The energy module also needs an explicit era configuration before it is used to
describe 2026 regulation behaviour.

## 10. Recommended implementation order

### Phase 1 — Freeze the methodology

- Finalise the event schema and provenance vocabulary.
- Define the persistence horizons and success metrics.
- Freeze the temporal/race-grouped split.
- Write leakage tests and baseline reports.

### Phase 2 — Make the historical dataset trustworthy

- Extract complete 2018-2025 race decision points.
- Add robust overtake and pit-cycle detection.
- Add track zones, flags, weather, tyres, stints, and both-car context.
- Store exclusion reasons and missingness.

### Phase 3 — Calibrate the two-car energy layer

- Version the 2018-2025 and 2026 regulation configurations.
- Replace separate heuristics with one shared transition function.
- Generate attacker and defender state on every replay step.
- Run sensitivity tests over uncertain SoC and calibration parameters.

### Phase 4 — Implement the backend replay engine

- Start from the exact real state at each 2026 decision point.
- Evaluate ATTACK, SAVE, and DELAY every eligible lap/zone.
- Reverse attacker/defender roles after a pass.
- Include repass risk, pit transitions, tyre evolution, and safety-car state.
- Return the best path, alternative branches, assumptions, and uncertainty.

### Phase 5 — Validate on unseen 2026 races

- Keep all evaluated 2026 outcomes out of training.
- Score classification and replay persistence separately.
- Compare against the real race and all baselines.
- Report results per race, driver, track, and decision horizon.

### Phase 6 — Make the UI explain the evidence

- Display the real state at the decision point.
- Show all three branches and why the selected branch wins.
- Show both cars' modelled energy and repass capability.
- Show real, derived, and modelled values separately.
- Show actual persistence beside estimated persistence.
- Label every counterfactual result as an estimate.

## 11. Acceptance criteria for saying the project is aligned

The project should not be described as fully aligned until all of the
following are true:

- A held-out 2026 race is never used to train or tune its own evaluation.
- Every feature can be traced to real data, a documented derivation, or an
  explicit model assumption.
- On-track passes are separated from pit and classification swaps.
- The model receives both cars' current state, including modelled energy.
- The same energy transition is used in extraction, prediction, and replay.
- Every eligible decision point evaluates ATTACK, SAVE, and DELAY.
- The roles reverse after an overtake and the opponent can attempt a repass.
- The visible race simulator is driven by the same real-race replay state and
  is not based on fixed-track synthetic dynamics.
- The primary metric is sustained laps ahead, with position change reported
  separately.
- Pit, tyre, flag, and safety-car effects are either modelled or explicitly
  held fixed and disclosed.
- The backend replay is deterministic for a fixed input, model version, and
  random seed.
- Results include baselines, sample counts, uncertainty, and per-race scores.
- The UI never presents modelled SoC or counterfactual finishing position as
  measured fact.

## 12. Additional scope decisions

### 12.1 How the application pages fit the objective

The current pages fit the project when each has one clear responsibility:

| Page | Responsibility in the aligned system |
| --- | --- |
| Strategy | Main decision surface: recommendation, selected tree path, expected laps ahead, energy cost, and opponent response |
| Track | Real circuit geometry, braking zones, straight zones, detection/activation lines, and event-specific mode rules |
| Telemetry | Evidence view for speed, throttle, brake, gear, RPM, DRS/Overtake state, lap time, and driver comparison |
| Energy | Both cars' modelled SoC, deployment, harvesting, clipping, constraints, and remaining response capability |
| Overtake | Decision-point detection, pass probability, hold/repass probability, and ATTACK/SAVE/DELAY outputs |
| Legends | REAL, DERIVED, and MODELLED definitions, sources, assumptions, and uncertainty |

These pages are conceptually correct, but they must consume one canonical
backend replay state. They should not independently calculate energy, gaps,
track zones, or outcomes. `Legends` may be renamed to `Methodology &
Provenance` because it explains the trust boundary of every displayed value.

The application also needs a clearly visible replay function, either inside
`Strategy` or as a dedicated `Replay` page. It must replace the current
synthetic race animation rather than sit beside it as a separate simulation.

### 12.2 The product has two operating modes

The project goal contains both live decision support and historical research.
They must share the same inference and transition code while keeping their
inputs and claims separate.

#### Real-time inference

At time `t`, the engine receives only the current state:

- current gap and closing rate;
- selected driver and opponent;
- track zone and event-specific Straight/Overtake Mode state;
- both cars' modelled energy;
- tyre compound, age, and recent pace;
- weather, flags, safety-car status, and pit context;
- regulation and energy constraints.

It returns ATTACK, SAVE, or DELAY, with an optional BOX recommendation for a
pit decision, confidence, estimated energy cost, pass probability, expected
lead duration, opponent repass risk, and an explanation.

The live engine cannot know the future outcome. It can only estimate it from
the trained models and state-transition assumptions. FastF1 provides live and
historical timing access through its supported data sources, while OpenF1
documents detailed historical data from 2023 onward and requires a paid
subscription for real-time access. OpenF1 is therefore a useful supplement,
not a solution to public battery-state availability.

Reference: [OpenF1 documentation](https://openf1.org/docs/).

#### Retrospective 2026 replay

For each completed 2026 race, the evaluator:

1. starts from the exact real state at a decision point;
2. runs the same inference logic that would have been used live;
3. evaluates ATTACK, SAVE, and DELAY branches;
4. reverses the roles after an overtake;
5. lets the opponent respond using its remaining modelled energy;
6. measures how long the selected driver stays ahead; and
7. compares the estimate with the observed race outcome.

This is a retrospective counterfactual estimate. It does not change the real
race and does not prove that an alternative finish would definitely have
occurred.

### 12.3 2026 rules must be event-specific

Straight Mode and Overtake Mode are separate concepts. Straight Mode is an
active-aero configuration available in designated areas; Overtake Mode is an
attack-related energy mode controlled by detection and activation conditions.
The model must store the event's detection gap, detection line, activation
line, energy limits, and flag/safety-car restrictions rather than applying one
permanent gap threshold to every track.

Reference: [FIA 2026 Sporting Regulations](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_05_-_2026-02-27.pdf)
and [Formula 1 2026 regulation explanation](https://www.formula1.com/en/latest/article/the-beginners-guide-to-the-2026-regulations.6j0tS0hrHG2T01tpmK6XYz).

## 13. Claims that remain out of scope

Even after the implementation is aligned, the project should not claim that:

- public data reveals a team's actual battery state;
- the model knows the driver's real private tactical instruction;
- a counterfactual replay proves what would have happened;
- a better simulated position guarantees a better real finish;
- a score from one demonstration race proves generalisation;
- a 2018-2025 model automatically understands every 2026 regulation detail.

The defensible claim is narrower and stronger: PitWolf uses real historical
race conditions, a documented two-car energy/overtake model, and unseen-race
evaluation to estimate which tactical choice would have maximised position
durability under stated assumptions.
