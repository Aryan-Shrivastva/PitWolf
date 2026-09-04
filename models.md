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

The model is trained using a strict temporal split. It must never train on the race used for evaluation. The current generated report trains on 2018–2024 and holds out 2025–2026, subject to the races available in the local cache.

### 2. Logistic Regression — interpretable benchmark

**Status:** Planned next comparison.

Logistic Regression will provide a simple linear benchmark. It will show whether the Random Forest is learning useful nonlinear relationships or merely benefiting from straightforward relationships such as smaller gap leading to more attack labels.

It should use the same features, temporal split, and labels as the Random Forest. It should also use class weighting and probability calibration where appropriate.

### 3. Gradient-boosted trees — likely primary tabular challenger

**Status:** Planned after the Logistic Regression benchmark.

The preferred candidates are XGBoost, if the dependency is available, or scikit-learn HistGradientBoosting as a portable alternative.

This family is well suited to structured telemetry rows because it can model interactions such as:

- a small gap being useful only when the attacker has enough energy;
- tyre advantage helping on one track but not another;
- traffic and dirty air changing the value of an attack;
- the defender's energy reserve changing re-pass risk.

The best candidate will be selected only if it improves balanced metrics and remains temporally honest. Higher raw accuracy alone is not sufficient.

### 4. Probability calibration layer

**Status:** Planned after model comparison.

The recursive tree uses action probabilities, so a prediction of 70% should behave approximately like a 70% event over many comparable cases. We will evaluate calibration using reliability checks and, if needed, use Platt scaling or isotonic calibration fitted only inside the training period.

Calibration is important for risk-reward decisions, position durability, and opponent response. It is separate from choosing the most accurate class.

### 5. Neural Network — optional sequence model

**Status:** Later-stage research.

A neural network should only be introduced after the row-based models are stable and the data is represented as lap sequences or short telemetry windows. Its purpose would be to learn temporal patterns such as:

- energy deployment over several consecutive laps;
- repeated attacks and failed attempts;
- tyre degradation trends;
- the effect of a previous pass on the next few laps.

It should not be added merely because it is more complex. It requires careful sequence construction, more compute, stronger leakage controls, and enough consistent telemetry across seasons.

### 6. Support Vector Machine — low priority

**Status:** Not currently planned for production.

An SVM could be used as an academic comparison, but it is a weak fit for the current use case. The dataset is large, the feature set is tabular, and the strategy tree needs reliable class probabilities. Training and probability calibration would also be less convenient than with tree-based models.

It should only be tested if the simpler benchmarks produce an unexpected result that needs investigation.

### 7. Reinforcement Learning — future decision policy

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

The current UI exposes the Random Forest, always-SAVE, and gap-only comparisons in the held-out validation table.

## Evaluation order

1. Freeze the decision-point labels and feature cutoff so no future information enters the input.
2. Train Logistic Regression, Random Forest, and gradient-boosted trees on the same historical rows.
3. Evaluate only on later, unseen races using a race-grouped temporal split.
4. Compare accuracy, macro F1, per-class precision/recall, confusion matrices, and probability calibration.
5. Check whether recommendations improve on the baselines, especially for the minority `ATTACK` and `DELAY` classes.
6. Pass the selected calibrated model probabilities into the two-car recursive tree.
7. Evaluate persistence: how many laps the selected driver remains ahead after a model-recommended pass, compared with the real race.
8. Consider a sequence neural model only if row-based models cannot represent multi-lap energy behaviour.
9. Consider reinforcement learning only after the replay environment and reward function pass validation tests.

## Selection rule

The production model will not be chosen by raw accuracy alone. A candidate must:

- beat or meaningfully explain the baselines;
- perform acceptably on macro F1 and minority-class recall;
- produce usable calibrated probabilities;
- avoid temporal leakage;
- improve the downstream persistence and counterfactual analysis;
- remain explainable enough for a race-strategy decision.

If a complex model does not improve these measures, the simpler model remains preferable.

## Important limitations

- Public FastF1 data does not expose the teams' true battery state, deployment commands, or complete strategic intent.
- Energy and some opponent-response values are modelled surrogates.
- A retrospective replay evaluates a counterfactual under the configured model; it does not claim that the real race was altered.
- More model types do not automatically make the result more accurate. Data quality, labels, leakage control, and simulator validity are more important than model count.
