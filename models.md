# PitWolf Model Plan

## Purpose

PitWolf is being developed as a real-time energy and overtake decision engine with retrospective race replay and position-durability evaluation.

At each eligible decision point, the system should recommend one of three actions:

- `ATTACK` — spend available energy to attempt the overtake now.
- `SAVE` — protect energy and position for a later opportunity.
- `DELAY` — remain in the battle without committing the full attack yet.

The system must also consider the opponent's remaining modelled energy. After a pass, the selected driver becomes the defender, so the tree must evaluate the risk of an immediate re-pass and how many laps the position is likely to be retained.

## Current architecture

PitWolf contains several different kinds of models. They should not be described as if they are all machine-learning models.

### 1. Random Forest Classifier — current machine-learning model

**Status:** Implemented and trained.

The Random Forest predicts `ATTACK`, `DELAY`, or `SAVE` from the features available at the decision-point cutoff:

- gap and closing rate;
- speed difference;
- tyre age, compound, and degradation proxies;
- lap fraction and track position;
- DRS/overtake eligibility;
- traffic, pack density, slipstream, and dirty-air proxies;
- car mass and fuel-load proxies;
- attacker and defender modelled state of charge.

The current implementation uses a class-balanced Random Forest with 400 trees, bounded depth, and a minimum leaf size. Its probabilities are passed to the recursive strategy tree, where they are combined with energy and race-context rules.

Candidate choice uses expanding historical validation rather than the final
2026 holdout: fit through 2021 and validate 2022, then repeat through 2025
with each next season held out. On the current `decision-point.v6` cache,
Random Forest leads weighted macro F1 across those folds (43.61%, three of
four fold wins), ahead of Gradient Boosting (43.00%). Its action thresholds
are then selected only from historical out-of-fold predictions. The resulting
held-out-2026 action policy scores 70.85% accuracy / 48.20% macro F1; it still
trails the always-SAVE baseline on raw accuracy (77.36%), so it remains a
research benchmark—not a validated live race-command policy.

The model is trained using a strict temporal split. It must never train on the race used for evaluation. The current generated report trains on 2018–2025 and holds out the completed 2026 races.

### Weather feature ablation

**Status:** Implemented as an experiment; excluded from the active production
feature set.

Decision-time FastF1 weather snapshots (air and track temperature, humidity,
wind, rainfall, and a missing-data flag) are available without future-race
leakage. The earlier v5 experiment regressed the Random Forest, so weather is
excluded from the active feature policy. It remains available behind
`--include-weather`, but must be re-run on the current v6 labels before any
new performance claim is made.

### 2. Logistic Regression — interpretable benchmark

**Status:** Implemented as the next comparison benchmark; Random Forest remains
the current research candidate until a model wins on the frozen holdout and the
downstream persistence metrics.

Logistic Regression will provide a simple linear benchmark. It will show whether the Random Forest is learning useful nonlinear relationships or merely benefiting from straightforward relationships such as smaller gap leading to more attack labels.

It should use the same features, temporal split, and labels as the Random Forest. It should also use class weighting and probability calibration where appropriate.

### 3. Gradient-boosted trees — likely primary tabular challenger

**Status:** Implemented as a portable benchmark with scikit-learn
`GradientBoostingClassifier`; Random Forest remains the current research candidate
until a candidate wins on the frozen holdout and downstream persistence.

The preferred candidates are XGBoost, if the dependency is available, or scikit-learn HistGradientBoosting as a portable alternative.

This family is well suited to structured telemetry rows because it can model interactions such as:

- a small gap being useful only when the attacker has enough energy;
- tyre advantage helping on one track but not another;
- traffic and dirty air changing the value of an attack;
- the defender's energy reserve changing re-pass risk.

The best candidate will be selected only if it improves balanced metrics and remains temporally honest. Higher raw accuracy alone is not sufficient.

### 4. Probability calibration layer

**Status:** Implemented as a benchmark; not selected for production.

The recursive tree uses action probabilities, so a prediction of 70% should behave approximately like a 70% event over many comparable cases. We will evaluate calibration using reliability checks and, if needed, use Platt scaling or isotonic calibration fitted only inside the training period.

Calibration is important for risk-reward decisions, position durability, and opponent response. It is separate from choosing the most accurate class.

The current sigmoid experiment fits its Random Forest on 2018–2024, learns
the calibration map only from 2025, and evaluates only on 2026. Under the
current v6 cache it lowers ECE from 20.99% to 8.14%, but reduces macro F1 from
48.20% to 37.75%. It is therefore retained as an honest benchmark only.

### 5. Immediate-pass component — descriptive benchmark

**Status:** Implemented for held-out evaluation only; not used by the action
classifier or tactical tree.

This binary model estimates whether a real immediate on-track pass was
observed from the causal battle context. It is useful for separating the
question *is a pass occurring at this point?* from the later question *will
the gained position survive?*

It is deliberately not a causal action-effect model. A positive row reflects
the real driver's energy deployment, defensive response, and race environment;
it cannot tell us what would have happened had that driver instead saved or
delayed. It must improve probability calibration and gain a defensible causal
action design before it can affect the tree.

The current calibrated re-run fits the underlying component through 2024,
fits only a sigmoid probability map on 2025, and then evaluates the untouched
2026 holdout. Its immediate-pass Brier error improves from 0.154 to 0.096 and
its AUC is 0.783. This is useful probability calibration, but it remains an
observational benchmark rather than a causal action-effect model.

### 6. Pass-durability component — separate conditional benchmark

**Status:** Implemented for held-out evaluation only; not used by the action
classifier or tactical tree.

This is deliberately a different model from the three-action classifier. It
is evaluated only after a real immediate on-track pass has occurred and asks:

> Given the causal pre-pass context, what is the probability that this gained
> position survives for at least 2, 3, 5, or 6 laps?

It uses the same public, causal feature set and strict 2018–2025 training /
2026 holdout boundary. This component is useful because it separates two
questions that must not be conflated: *can a driver gain the position now?*
and *if they do, will they stay ahead?*

The current 2026 result is not strong enough to deploy. A one-lap target has
no negative class because each retained immediate pass is already observed to
survive its first lap. For 2–6 laps, the model has modest discrimination (AUC
0.73–0.78) but does not yet beat a simple historical hold-rate reference on
Brier probability error. It is therefore shown as a transparent diagnostic
benchmark, and it must not yet change a tree branch, an overtake
recommendation, or a counterfactual claim.

The same pre-2026 sigmoid calibration protocol improves held-out Brier error
for the 2, 3, 5, and 6-lap targets (respectively 0.076→0.063,
0.082→0.063, 0.105→0.088, and 0.106→0.088). These values are displayed in
Validation as `CAL. BRIER`; they are not replay inputs.

To make it eligible for integration, it needs a better event definition,
calibrated probability improvement versus the reference, and a causal method
for scoring the no-pass/SAVE/DELAY alternatives without using future outcomes
as inputs.

### 7. Neural Network — optional sequence model

**Status:** Later-stage research.

A neural network should only be introduced after the row-based models are stable and the data is represented as lap sequences or short telemetry windows. Its purpose would be to learn temporal patterns such as:

- energy deployment over several consecutive laps;
- repeated attacks and failed attempts;
- tyre degradation trends;
- the effect of a previous pass on the next few laps.

It should not be added merely because it is more complex. It requires careful sequence construction, more compute, stronger leakage controls, and enough consistent telemetry across seasons.

### 8. Support Vector Machine — low priority

**Status:** Not currently planned for production.

An SVM could be used as an academic comparison, but it is a weak fit for the current use case. The dataset is large, the feature set is tabular, and the strategy tree needs reliable class probabilities. Training and probability calibration would also be less convenient than with tree-based models.

It should only be tested if the simpler benchmarks produce an unexpected result that needs investigation.

### 9. Reinforcement Learning — future decision policy

**Status:** Deferred until the simulator is trustworthy.

Reinforcement Learning is not the same as the current recursive tree. The tree evaluates a finite set of actions over a short horizon. Reinforcement Learning would learn a policy from repeated interactions with a simulator.

It becomes appropriate only after the project has:

- a validated lap-by-lap state transition model;
- realistic two-car energy response;
- tyre, pit, traffic, safety-car, and race-control transitions;
- a reward function focused on position durability, not simply finishing position;
- strict held-out race evaluation.

The reward should value staying ahead for additional laps while penalising energy waste, failed attacks, re-passes, unsafe/rule-inconsistent actions, and unnecessary pit loss. RL must not be trained directly on the real race outcome as if it were an alternate historical truth.

## Non-ML models already used

### Energy state-of-charge surrogate

The project does not have private battery telemetry. The energy layer therefore estimates state of charge from public timing/telemetry and configured 2026 assumptions. It models deployment, harvesting, capacity limits, and continuity for both cars. Every such value must remain labelled `MODELLED`, not measured team data.

### Recursive strategy tree

At each decision point the tree evaluates all three actions. It then models the opponent's response, changes the attacker/defender roles after a pass, and projects persistence across the configured horizon. This is a deterministic search/planning layer, not a separately trained neural network or reinforcement-learning agent.

### Race and rule gates

Safety-car, yellow-flag, red-flag, pit-cycle, lapping, and other non-normal states gate or alter overtake opportunities. These are domain rules and data-quality controls, not learned predictions.

## Baselines used for honest comparison

Every candidate model should be compared with the same held-out races and at least these baselines:

- **Always-SAVE:** always predicts `SAVE`; this exposes the class-imbalance problem.
- **Gap-only:** `ATTACK` at gap ≤ 0.70 seconds, `DELAY` at gap ≤ 1.20 seconds, otherwise `SAVE`; this tests whether the model adds value beyond gap size alone.
- **Majority-class baseline:** predicts whichever label is most common in the training period.

The current UI exposes the Random Forest action policy, Logistic Regression,
Gradient Boosted Trees, always-SAVE, and gap-only comparisons in the held-out
validation view. On the current v6 2026 holdout, the Random Forest action
policy scores 70.85% accuracy / 48.20% macro F1, Logistic Regression scores
56.68% / 41.17%, and Gradient Boosted Trees scores 60.26% / 44.81%.
Always-SAVE reaches 77.36% accuracy but only 29.08% macro F1 because it never
identifies ATTACK or DELAY. The Random Forest is the current research
candidate; no result is presented as a validated production race command.

## Evaluation order

1. Freeze the decision-point labels and feature cutoff so no future information enters the input.

   The current `decision-point.v6` audit additionally requires a clean future
   outcome window: an immediate ATTACK label must hold for six laps, and a
   DELAY label must become a durable pass within five laps. Rows contaminated
   by pit cycles, flags/invalid timing, lapping interactions, or an incomplete
   horizon are excluded rather than guessed. A model must only be retrained
   after every participating cache file has this schema.

   The first clean v6 retrain used 10,558 rows from 2018–2025 and 614 unseen
   2026 rows. Random Forest reached 62.38% accuracy and 44.33% macro F1;
   always-SAVE reached 77.36% accuracy because SAVE is overwhelmingly common.
   This means the v6 result is a trustworthy benchmark, not a policy approval.
   Its ATTACK F1 is 37% and DELAY F1 is 20%, so the next modelling work should
   improve those minority actions rather than optimise headline accuracy.

   A subsequent decision-policy layer selected ATTACK and DELAY evidence
   weights using 5,909 expanding-fold historical validation rows only. It
   selected `ATTACK × 0.65`, `DELAY × 0.80`, and `SAVE × 1.00`: non-SAVE
   actions therefore require stronger model evidence before recommendation.
   On the untouched 2026 holdout this improved macro F1 from 44.33% to 48.20%
   and accuracy from 62.38% to 70.85%. It remains below the always-SAVE
   accuracy baseline, so it is a balanced-decision improvement, not policy
   deployment approval.
2. Train Logistic Regression, Random Forest, and gradient-boosted trees on the same historical rows.
3. Evaluate only on later, unseen races using a race-grouped temporal split.
4. Compare accuracy, macro F1, per-class precision/recall, confusion matrices, and probability calibration.
5. Check whether recommendations improve on the baselines, especially for the minority `ATTACK` and `DELAY` classes.
6. Pass the selected calibrated model probabilities into the two-car recursive tree.
7. Evaluate persistence: how many laps the selected driver remains ahead after a model-recommended pass, compared with the real race.
8. Consider a sequence neural model only if row-based models cannot represent multi-lap energy behaviour.
9. Consider reinforcement learning only after the replay environment and reward function pass validation tests.

## Selection rule

The eventual deployment candidate will not be chosen by raw accuracy alone. A candidate must:

- beat or meaningfully explain the baselines;
- perform acceptably on macro F1 and minority-class recall;
- produce usable calibrated probabilities;
- avoid temporal leakage;
- improve the downstream persistence and counterfactual analysis;
- remain explainable enough for a race-strategy decision.

If a complex model does not improve these measures, the simpler model remains preferable.

The current persistence evaluator uses the completed 2026 immediate-pass rows
as its evaluation units and a six-lap replay horizon. It reports both the raw
observed hold duration and a horizon-capped observed value; the latter is the
fair comparison with the replay estimate. Run it after training with:

```text
python backend/scripts/evaluate_replay_persistence.py
```

Its output is stored in `overtake_report.json` under `replayPersistence` and
is rendered in the Overtake validation page. A positive delta is an estimate
under the documented modelled-SoC/opponent-response assumptions, not proof of
an alternate real-race result.

## Important limitations

- Public FastF1 data does not expose the teams' true battery state, deployment commands, or complete strategic intent.
- Energy and some opponent-response values are modelled surrogates.
- A retrospective replay evaluates a counterfactual under the configured model; it does not claim that the real race was altered.
- More model types do not automatically make the result more accurate. Data quality, labels, leakage control, and simulator validity are more important than model count.

## Zone replay and BOX model foundations

The future simulation view has two deliberately separate evidence paths:

- **Zone replay:** `extract_zone_opportunities.py` takes only a cited FIA
  Detection Line / Activation Line record and interpolates public FastF1
  car-data timing at the same track distance for every car. It emits actual
  detection-line gaps and speeds. `build_zone_replays.py` may associate the
  existing lap-start outcome label and modelled energy state with that record,
  but labels this `OBSERVATIONAL_ASSOCIATION_ONLY`; it does not claim that a
  zone action caused the pass or subsequent durability.
- **BOX candidate model:** `extract_box_candidates.py` builds an independent
  per-driver state immediately after a completed lap and labels only whether
  that car entered the pits on the following lap. `train_box_model.py` uses a
  2018–2025 / 2026 temporal split and a class-weighted logistic regression as
  a transparent baseline. Its target is an observed pit-window pattern, not
  an optimal tyre strategy, radio instruction, or live BOX command.

The first BOX baseline contains 154,874 training rows and 11,387 unseen 2026
rows. Its held-out AUC is 0.6941, but the event is rare (3.26% in the 2026
holdout) and precision is only 6.12% at the neutral 0.50 threshold. That is
not a usable automatic pit-call rate. The baseline is retained to establish a
leakage-safe starting point and will require race-strategy features, calibrated
thresholds, and counterfactual evaluation before it can influence a simulation.

Neither path can satisfy the real-time gate on its own. A cited event-specific
FIA appendix, exact zone identity, complete decision-state inputs, and a
defensible causal evaluation remain required before any live-command claim.
