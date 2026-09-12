# PitWolf model roadmap

## Ground rules

PitWolf must clearly separate **observed data** from **modelled output**.

- Observed public FastF1 data: lap time, timing position, tyre compound, speed, throttle, brake, RPM, GPS/track position, weather, timing gaps and session/race events.
- Not publicly observed: ERS deployment commands, battery state of charge, energy-management mode, exact fuel load, setup, tyre condition, and team strategy intent.
- Therefore, any battery/ERS output must remain labelled **MODELLED** unless a future licensed data source supplies those channels.
- Every predictive model needs a hold-out test set. A good-looking simulation alone is not validation.

## FIA compliance layer — implemented foundation

The energy engine now has a separate FIA compliance context rather than
silently treating all modelled traces as fully legal.

- Global 2026 limits applied: C5.2.7 350 kW ERS-K DC ceiling, C5.2.8
  speed-dependent propulsion ceiling, C5.2.9 4 MJ SoC swing, C5.2.10 default
  8.5 MJ/lap Recharge ceiling, C5.2.11 500 Nm MGU-K torque ceiling and C5.2.21
  0.97 Standard ECU conversion correction.
- C4 mass is session-aware: Qualifying/Sprint Qualifying uses 726 kg plus the
  nominal tyre mass; all other sessions use 724 kg plus the nominal tyre mass.
- All 13 completed 2026 project rounds (Australia through Italy) now load their
  official FIA Power Unit Information document: session-specific Recharge caps,
  qualifying/race curves, C5.2.8iii alternative-power sectors, B7.2 detection
  gap/line/activation line and the cited document date/URL. These traces are
  labelled `FIA_EVENT_LIMITS_LOADED`.
- A later or unsourced event remains `FIA_GLOBAL_LIMITS_ONLY`. Required event
  fields are a cited FIA source/publication date, session-specific Recharge
  limits, the applicable curve and—outside qualifying—the official
  alternative-power-sector list. Missing values are never inferred from
  telemetry or another Grand Prix.

## Current baseline — already usable

### Qualifying usable-SoC reference set

**Implemented foundation for energy intelligence.** Each completed 2026
qualifying event now has a driver-specific reference based on that driver's
fastest clean Soft lap. The model begins at 100% and calculates deployment and
braking recovery through the FIA 4 MJ on-track SoC window, targeting 0% at lap
end without ever forcing a recharge or an unphysical deployment.

- 282 references across 13 completed rounds were generated from cached public
  FastF1 telemetry.
- Four drivers have no recorded clean Soft qualifying lap and remain explicitly
  unavailable—no other tyre is substituted.
- A small number of traces cannot consume the full 4 MJ window under their
  recorded speed/power envelope; they remain visibly `NOT_CALIBRATED` rather
  than falsely being forced to 0%.
- This is **modelled usable SoC**, not a team-measured battery percentage or
  hidden deployment command.

### Qualifying ERS deployment inference

**Implemented foundation.** Qualifying traces now use the FIA Overtake-active
curve that applies lap-wide during an LTCS. For every reference lap, PitWolf
uses public throttle, longitudinal acceleration, tractive-power demand and
braking telemetry to prioritise likely ERS deployment locations, then applies
the FIA 350 kW, speed-curve, recharge and 4 MJ usable-SoC constraints.

- The resulting trace is `PUBLIC_TELEMETRY_CONSTRAINED_INFERENCE`.
- It can identify likely high-deployment / near-350 kW regions and their
  associated modelled battery draw.
- It cannot reconstruct a team's ERS button, private SoC, fuel-flow or actual
  deployment command. Those channels are not present in public FastF1 data.
- The user interface shows the instantaneous inferred ERS-K power and the
  telemetry-evidence score alongside the live modelled battery state.

### User Straight Mode deployment policy

**Implemented for qualifying.** The circuit Straight Mode turn ranges supplied
by the user now form a hard deployment-placement boundary: PitWolf may model
ERS-K discharge only within those mapped ranges. Blue map segments therefore
represent inferred battery deployment only there; green segments are ICE
propulsion without inferred ERS draw. Braking recovery remains modelled from
public braking telemetry anywhere on the circuit.

- This is an explicit PitWolf scenario policy based on user-supplied circuit
  maps. It is **not** FIA Overtake activation/detection-line data and is not
  presented as measured team deployment.
- FIA C5 power, SoC, torque and recharge limits continue to apply inside a
  permitted range.
- If no supplied range exists for an event, its qualifying trace deliberately
  stays ICE-only rather than inventing a battery deployment location. Current
  supplied references do not include round 4; Monaco is marked with no
  Straight Mode ranges. Monza now has a five-range, user-confirmed topology
  fallback and is explicitly marked as such in the model output.
- The policy is applied consistently to the recorded qualifying energy trace,
  the 2026 qualifying battery-reference dataset and the generic optimal-lap
  optimiser, so the latter cannot redistribute energy outside a supplied zone.

### FIA-constrained physics energy optimiser

**What it does**

Uses public telemetry with vehicle-physics assumptions and FIA 2026 power/energy constraints to produce a regulation-bounded deployment and harvesting plan.

**Current data**

- 13 completed 2026 qualifying sessions and 1,769 clean timing laps.
- 1,768 clean laps with locally cached, usable car telemetry across all 13 sessions.
- Every available 2026 qualifying session now has public speed, throttle, brake, RPM and position coverage.

**Status**

Implemented. It is an explainable, deterministic optimiser—not an ML model—and its ERS/SoC outputs are modelled.

**Current phase gate (September 2026)**

Phase 1 is **complete**. The first qualifying pace baseline was trained and
tested on 1,768 observed rows from all 13 available sessions. Random Forest
outperformed Ridge on a temporal hold-out of the last three rounds, but its
held-out MAE is 1.747s with a +0.844s bias. It is therefore
`PACE_BASELINE_EVALUATED`, not a team-optimal or generic-optimal serving model.
It predicts observed pace delta only; it cannot infer a private battery state
or deployment plan. Do not proceed to an energy-serving model until the
battery/energy target is defined and its own validation gate passes.

**Exit gate before expanding it**

- Keep all FIA regulation checks automated.
- Show modelled assumptions and uncertainty in the UI.
- Compare predicted pace gains against held-out qualifying sessions; do not claim exact team deployment.

---

## Phase 1 — Complete the qualifying feature dataset

**Purpose**

Build the common dataset needed for every qualifying pace ML model.

**What must be collected for every clean 2026 qualifying lap**

- Event, circuit, round, session segment and track length.
- Driver and team.
- Lap time, sector and mini-sector times.
- Tyre compound and lap age where available.
- Weather and track temperature.
- Speed trace, top speed, average speed and speed-trap metrics.
- Throttle, brake, coast and full-throttle duration.
- Track-position / distance-normalised features.
- Data-quality flags: deleted laps, pit laps, invalid telemetry, yellow/red flag effects.

**Required before starting**

- Cached FastF1 qualifying data for every completed 2026 round.
- A consistent definition of a “clean lap.”
- Circuit-aware normalisation: a 75-second Monaco lap cannot be treated as equivalent to an 85-second Monza lap.

**Output**

One reproducible versioned table, for example:

`data/models/qualifying_features_2026_v1.parquet`

Each row is one clean lap, with a documented schema and data-source version.

**Validation gate**

- No duplicate laps.
- No pit/out/in laps used as performance targets.
- Feature completeness report by event and driver.
- At least several completed rounds held back from model training.

**Then proceed to**

Phase 2: baseline pace models.

---

## Phase 2 — Explainable qualifying pace baselines

### 2A. Ridge / Elastic Net regression

**Why first**

It is simple, fast and explainable. It reveals whether the feature dataset contains useful signal before adding a more complex model.

**Target**

- Lap time, or preferably normalised mini-sector/lap-time delta to the event’s P1 reference.

**Inputs**

- Features produced in Phase 1.

**Required before starting**

- Phase 1 dataset and round-based train/test split.

**Validation**

- Mean absolute error (MAE) in seconds.
- Error by circuit type, team and tyre compound.
- Coefficient review to catch impossible relationships.

**Exit gate**

- Baseline outperforms a simple “event median” prediction on held-out rounds.
- No feature leakage: a result from the held-out race cannot enter training.

### 2B. Random Forest regressor

**Why**

Provides a nonlinear, interpretable baseline and feature importance without relying on a single linear relationship.

**Required before starting**

- Same Phase 1 dataset and evaluation split.

**Validation / exit gate**

- Compare held-out MAE against Ridge.
- Inspect feature importance for sensible drivers: tyre, speed/brake profile, weather and circuit—not future-result fields.

**Then proceed to**

Phase 3: boosted pace model if it materially improves the baseline.

---

## Phase 3 — XGBoost or LightGBM qualifying pace model

**Purpose**

Predict realistic achievable lap-time or mini-sector pace from telemetry and session context. This model does **not** predict true team ERS deployment.

**Why use XGBoost / LightGBM**

- Excellent for structured tabular data.
- Captures nonlinear interactions: circuit × tyre × temperature × braking profile.
- Can provide feature importance and be validated against held-out events.

**Required before starting**

- Completed Phase 1 feature table.
- Ridge/Random Forest baseline scores from Phase 2.
- At least 10–13 completed events; more events improve confidence.
- Strict leave-one-round-out or chronological validation split.

**Training target**

- Primary: clean-lap delta to the P1 reference, or mini-sector delta.
- Optional secondary: probability that a lap is within a defined percentage of the event’s pole pace.

**Validation**

- Held-out MAE / RMSE by round.
- Calibration plot: predicted vs actual pace.
- Feature leakage audit.
- Compare against Ridge and Random Forest; do not adopt XGBoost merely because it is more complex.

**Exit gate**

- XGBoost/LightGBM improves held-out error meaningfully and consistently.
- Errors are reported in the product and model card.
- Model artefact, feature schema, training rounds and evaluation result are versioned.

**Then proceed to**

Phase 4: combine predicted pace with the FIA-constrained optimiser.

---

## Phase 4 — Hybrid optimal-lap model

**Purpose**

Create a credible simulated qualifying optimum:

`ML pace potential + FIA-constrained physics/energy optimiser`

**How it works**

1. The ML model estimates achievable pace by mini-sector from observed telemetry/context.
2. The physics optimiser selects a legally bounded energy allocation and recovery plan.
3. The optimiser may redistribute modelled energy; it must not create energy or exceed FIA limits.
4. The result reports a predicted lap time with an uncertainty range.

**Required before starting**

- A validated Phase 3 pace model.
- Existing regulation checks remain active.
- A clearly documented tyre, weather and fuel/context assumption.

**Validation**

- Back-test on held-out qualifying rounds.
- Confirm every output respects power, recovery and battery-window limits.
- Compare predicted optimum against P1 without claiming that it represents a real team’s unseen deployment plan.

**Product language**

Use: `PitWolf modelled qualifying optimum`.

Do not use: `actual full deployment`, `real battery state`, or `what the team used`.

**Exit gate**

- Every output includes model version, reference lap, expected gain and uncertainty.
- Physics constraints and held-out evaluation pass.

**Then proceed to**

Phase 5: uncertainty and telemetry-state learning.

---

## Phase 5 — Bayesian uncertainty and HMM driving-state model

### 5A. Bayesian uncertainty layer

**Purpose**

Replace false precision such as “gain = 0.112s” with a range such as:

`Expected modelled gain: 0.11s (credible range: 0.04s–0.18s)`

**Required before starting**

- Phase 3/4 predictions across multiple held-out rounds.
- Residual errors stored by circuit, tyre and weather context.

**Validation / exit gate**

- Prediction intervals contain the actual held-out result at approximately their declared rate.
- UI presents the range, not just a point estimate.

### 5B. Hidden Markov Model (HMM) driving-state classifier

**Purpose**

Learn recurring observed driving phases from telemetry: braking, coast, traction, full throttle and high-speed acceleration.

**Required before starting**

- Phase 1 normalised sequential telemetry data.
- A state-count experiment and human review of output zones.

**Important limitation**

It can classify observable driving states. It cannot identify actual hidden ERS modes because those are not public labels.

**Validation / exit gate**

- State transitions align with observed brake/throttle/speed changes.
- F1-aware review shows that states map sensibly around corners and straights.

---

## Phase 6 — Race interaction models

### 6A. Overtake probability model

**Recommended model family**

Start with logistic regression or XGBoost classification.

**Target**

Whether an overtake occurs during a defined attack window.

**Required data**

- Labelled attack windows and pass/no-pass outcomes.
- Gap to car ahead, relative speed, DRS eligibility, tyre age/compound, lap number and track location.
- Safety car, VSC, red flag and wet/dry condition flags.
- Several seasons of consistent race data; 2026 alone will be too small initially.

**Validation / exit gate**

- Precision/recall and probability calibration on held-out races.
- Separate results by circuit, since Monaco and Monza behave very differently.

### 6B. Tyre degradation / pit-stop timing model

**Recommended model family**

Survival/hazard model first; compare with XGBoost regression/classification later.

**Required data**

- Full stint history, tyre compound/age, lap-time degradation, pit stops, weather, traffic and safety-car timing.
- Clear targets: pit within N laps, degradation slope, or probability of losing position.

**Validation / exit gate**

- Back-test recommendations against held-out races.
- Include uncertainty and avoid treating pit strategy as a guaranteed result.

---

## Phase 7 — Sequence models

### Temporal CNN, LSTM or Transformer

**Purpose**

Learn full-lap telemetry sequences rather than manually aggregated features.

**Required before starting**

- A much larger, quality-controlled cross-season telemetry dataset: thousands of clean laps per relevant rules era.
- A precise target: mini-sector pace, tyre degradation trajectory, driving-state sequence, or pass probability.
- Baseline XGBoost results to beat.
- Proper compute, experiment tracking and train/validation/test split by event—not random telemetry points.

**Why not start here**

These models can overfit badly because adjacent telemetry samples are highly correlated. They should be used only after the simpler models are validated.

**Exit gate**

- Consistent improvement over Phase 3 on unseen rounds.
- No leakage between laps from the same event across training and test sets.

---

## Phase 8 — Full-race optimisation

### Model-predictive control (MPC) / reinforcement learning (RL)

**Purpose**

Optimise a full race: tyre plan, energy use, attack/defence choice, pit timing and response to changing race state.

**Required before starting**

- Validated tyre, pit, traffic and overtake models from Phase 6.
- FIA-compliant energy model from the current baseline.
- A robust race simulator with safety car/VSC/red-flag handling.
- Historical back-tests and a defined reward function that does not reward illegal or unsafe behaviour.

**Why MPC before RL where possible**

MPC is easier to constrain, explain and validate. RL is valuable for long-horizon choices, but needs a trustworthy simulator or it learns simulator mistakes.

**Exit gate**

- Counterfactual recommendations outperform simple baselines in historical simulations.
- Every recommendation reports assumptions and uncertainty.
- This remains a decision-support tool, not a claim of certain race outcomes.

---

## Recommended implementation order

1. Finish **Phase 1**: all clean 2026 qualifying laps into a feature table.
2. Implement and evaluate **Ridge** and **Random Forest** baselines.
3. Implement **XGBoost/LightGBM** only if it beats those baselines on held-out rounds.
4. Integrate it into the existing FIA energy optimiser as the **hybrid optimal-lap model**.
5. Add **Bayesian uncertainty** and, separately, the **HMM driving-state classifier**.
6. Build historical race interaction data before creating overtake, tyre and pit models.
7. Consider sequence models and then full-race MPC/RL only after the simpler models are proven.

## What PitWolf can accurately claim at each point

| Completed through | Accurate description |
|---|---|
| Current baseline | FIA-constrained, telemetry-driven physics energy optimiser and race replay engine. |
| Phase 2 | Explainable ML qualifying pace baselines, evaluated on held-out 2026 rounds. |
| Phase 3 | Validated gradient-boosted qualifying pace prediction using 2026 telemetry features. |
| Phase 4 | Hybrid ML + FIA-constrained modelled qualifying optimiser with uncertainty. |
| Phase 6 | Validated race-interaction, overtake and tyre/strategy prediction models. |
| Phase 8 | Constraint-aware full-race decision-support simulation, back-tested on historical data. |

## Non-negotiable documentation for every model

- Model name and version.
- Training data scope and date range.
- Input feature schema.
- Target definition.
- Train/validation/test split method.
- Held-out metrics and uncertainty.
- Known limitations and unavailable signals.
- Clear `OBSERVED` versus `MODELLED` labels in the UI.
