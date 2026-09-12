# PitWolf — New Laptop Setup and Data-Rebuild Handoff

Use this document when moving PitWolf to another computer. It is written for
the agent setting up that laptop: complete the steps in order, validate each
checkpoint, and do not replace modelled values with invented data when a
dataset or external source is unavailable.

## 1. Target outcome

The new laptop must be able to:

- run the React/Vite frontend at `http://127.0.0.1:5173`;
- run the Node.js API at `http://127.0.0.1:8787`;
- fetch and cache public FastF1 session, telemetry, energy, track-map, and
  recorded replay data on demand;
- build the historical 2018–2025 decision-point dataset and train the local
  overtake model;
- keep completed 2026 races held out from model training and use them only for
  retrospective replay/evaluation;
- run the Simulation page, including future-blind two-car attack and defence
  branches, using real cached public race state.

This is a reproducible local setup. `backend/data/f1-cache/` is deliberately
ignored by Git because it can be large and machine-generated. It must be
rebuilt locally or transferred securely as an optional cache acceleration.

## 2. Source code

Clone the repository and use the branch requested by the project owner. If no
branch is specified, use `main`.

```powershell
git clone https://github.com/Aryan-Shrivastva/PitWolf.git
cd PitWolf
git switch main
git pull --ff-only origin main
```

Before installing anything, check that these files exist:

```text
package.json
package-lock.json
frontend/package.json
backend/package.json
backend/requirements-data.txt
backend/server.mjs
backend/scripts/fetch_f1_replay_window.py
backend/scripts/replay_strategy.py
frontend/src/components/SimulationReplayView.jsx
frontend/src/simulationreplay.css
```

If a feature branch is supplied instead, run `git fetch origin`, then switch
to that exact branch and use `git pull --ff-only origin <branch-name>`.

## 3. Software prerequisites

Install these before starting:

- Git.
- Node.js LTS with npm (Node 20+ is recommended).
- Python 3.13.x. The active development setup uses Python 3.13.
- Windows PowerShell.
- Internet access for first-time FastF1 downloads and `npm ci`.
- At least 30 GB free disk space for FastF1 data, generated datasets, and
  model artifacts. More is safer for a full multi-season rebuild.

Check versions from the repository root:

```powershell
git --version
node --version
npm --version
py -3.13 --version
```

If `py -3.13` does not work, install Python 3.13 and ensure the Python launcher
is enabled. Do not silently use an arbitrary system Python version.

## 4. Install JavaScript dependencies

The repository includes a lock file. Prefer a deterministic install:

```powershell
npm ci
```

Use `npm install` only if the lock file is intentionally being updated.

Validate the frontend bundle before any data work:

```powershell
npm run build
node --check backend/server.mjs
```

Expected result: Vite completes successfully and Node reports no syntax error.

## 5. Create the Python data/model environment

From the repository root:

```powershell
py -3.13 -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements-data.txt
```

`backend/requirements-data.txt` currently provides the data pipeline stack:

```text
duckdb==1.5.5
fastf1>=3.8,<4
pandas>=2.2,<3
numpy>=1.26,<3
scikit-learn>=1.5,<2
```

Validate that the backend will launch scripts from this environment:

```powershell
Get-Command python
python -c "import fastf1, pandas, numpy, sklearn, duckdb; print('Python data stack OK')"
python backend\scripts\batch_extract_decision_points.py --help
python backend\scripts\train_overtake_model.py --help
```

`Get-Command python` must resolve to
`backend\.venv\Scripts\python.exe`. The Node backend invokes `python`, so
start it from a terminal with this virtual environment activated.

## 6. Environment files and secrets

Create local environment files only when an optional provider is needed:

```powershell
Copy-Item .env.example .env
Copy-Item backend\.env.example backend\.env
Copy-Item frontend\.env.example frontend\.env
```

The core strategy, energy, overtake, and simulation workflow does **not** need
Supabase or Hugging Face credentials. Leave these blank unless those optional
features are being used:

```text
SUPABASE_URL=
SUPABASE_ANON_KEY=
HF_API_TOKEN=
HUGGINGFACEHUB_API_TOKEN=
VITE_SUPABASE_URL=
VITE_SUPABASE_ANON_KEY=
```

For local development the frontend API URL is:

```text
VITE_API_URL=http://localhost:8787
```

Never copy secrets into Git, screenshots, documentation, or prompts. The
templates are safe; populated `.env` files are machine-specific.

## 7. Data that is tracked vs rebuilt

### Tracked in Git

- application source under `frontend/src/` and `backend/`;
- JavaScript package manifests and lock file;
- Python dependency manifest;
- small curated/static data under `backend/data/` such as radio examples,
  overtake rule context, and straight-mode visual evidence;
- documentation, model methodology, and schema/validation code.

### Rebuilt or cached locally

```text
backend/data/f1-cache/
```

Typical local cache namespaces include:

```text
sessions/              # FastF1 session/lap/result data
telemetry/             # per-driver lap telemetry
trackmap/              # projected circuit geometry and corner labels
replay-window/         # selected-lap replay windows
race-replay/           # full recorded race playback windows
energy/                # per-lap energy/SoC surrogate outputs
energyrace/            # race-level energy outputs
decision-points/       # close-battle model rows
zone-opportunities/    # zone-aligned observational evidence
zone-replays/          # zone replay records
models/                # locally trained .joblib model and reports
```

`backend/data/fia-docs/` is also local evidence storage. It is not a source of
truth unless files have provenance and are cited. Do not fabricate FIA zone,
detection, or activation coordinates from an image.

## 8. Minimum runnable data setup

For a first launch, do **not** wait for the full 2018–2025 extraction. Start
the services and open a completed race. The backend fetches public data on
demand and writes it to `backend/data/f1-cache/`.

In terminal 1:

```powershell
cd C:\path\to\PitWolf
backend\.venv\Scripts\Activate.ps1
npm run server
```

In terminal 2:

```powershell
cd C:\path\to\PitWolf
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173/` and select a completed race. The first request
may take several minutes because FastF1 needs to download/cache public data.
Future requests use the local cache.

Verify both services:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8787/api/health
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5173/
```

Both must return HTTP `200`.

## 9. Full reproducible overtake-model dataset

The local simulation UI can replay recorded public timing before the full model
is trained, but model-backed action probabilities require the historical cache
and trained artifact below.

### 9.1 Build historical training rows

With the venv activated and internet available:

```powershell
backend\.venv\Scripts\python.exe backend\scripts\batch_extract_decision_points.py `
  --start-year 2018 `
  --end-year 2025 `
  --session R `
  --hold-laps 6
```

This is long-running but resumable. It generates the historical decision-point
cache under `backend/data/f1-cache/decision-points/`.

Important data rule:

```text
Training rows: 2018–2025 only
Final held-out evaluation/replay: completed 2026 races
```

Never add a 2026 race to the training data if that same race is used to judge
the system. The Simulator must receive only the selected historical state at
JUMP; it must not read later laps as policy input.

### 9.2 Train the overtake model

After extraction has completed:

```powershell
backend\.venv\Scripts\python.exe backend\scripts\train_overtake_model.py `
  --test-from 2026 `
  --n-jobs 1
```

Expected local artifacts:

```text
backend/data/f1-cache/models/overtake_rf.joblib
backend/data/f1-cache/models/overtake_report.json
```

The current trained policy is a research Random Forest classifier. It produces
`ATTACK`, `DELAY`, and `SAVE` signals. It is not a validated live race-command
system. Modelled state of charge is a public-data surrogate, never private team
battery telemetry.

### 9.3 Optional secondary datasets

Do these only when the relevant product work requires them:

- Zone evidence/replays: `extract_zone_opportunities.py` and
  `build_zone_replays.py`, after importing cited FIA event-appendix evidence.
- BOX baseline: `batch_extract_box_candidates.py` followed by
  `train_box_model.py`.
- Radio data: the `ingest-*.mjs` scripts, only if the radio/transcription
  experience is being worked on.

Do not block core simulation startup on these optional pipelines.

## 10. Simulation-specific acceptance checks

After data is available, verify these behaviours in the Simulation page:

1. Select a completed 2026 race and a driver; the recorded replay and
   classification load.
2. Driver positions, gap board, tyre data, and track markers derive from
   recorded public timing/telemetry rather than placeholder values.
3. Select a lap from **JUMP / BRANCH LAP**. Selection alone must not start a
   counterfactual.
4. Press **JUMP**. The page starts a future-blind two-car branch from that
   selected public state through the finish.
5. For a selected driver behind a close opponent, the role is `ATTACKING`.
   For a driver with a close car behind, the role is `DEFENDING`. Both are
   valid branches when an extracted battle exists.
6. The branch displays action probabilities, modelled energy, opponent
   response, pair-order estimate, and observed comparison.
7. A lap without an extracted close battle must say no branch is available;
   it must not fabricate a battle, pass, FIA zone, or result.
8. Upcoming/uncompleted 2026 races must show the no-race-data message instead
   of pretending to simulate them.

The current branch is intentionally pair-only. It does not claim a
full-grid alternate final classification because it does not yet model all
cars, pit alternatives, traffic, tyre evolution, retirements, or future race
control events.

## 11. Validation commands

Run before handing the new laptop back:

```powershell
npm run build
node --check backend/server.mjs
python -m py_compile backend\scripts\replay_strategy.py
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8787/api/health
```

For full model work, additionally confirm:

```powershell
Test-Path backend\data\f1-cache\models\overtake_rf.joblib
Test-Path backend\data\f1-cache\models\overtake_report.json
```

If the `Test-Path` checks are `False`, model-backed recommendations are not
ready yet. Do not describe that state as a working trained model.

## 12. Common problems and fixes

| Symptom | Cause | Safe fix |
| --- | --- | --- |
| `spawn python ENOENT` | Backend cannot find the Python venv. | Activate `backend\.venv` before `npm run server`; verify `Get-Command python`. |
| `ModuleNotFoundError: fastf1` | Packages installed into the wrong Python. | Activate the venv and rerun `python -m pip install -r backend\requirements-data.txt`. |
| `model not trained yet` | Historical extraction/training artifacts are missing. | Run sections 9.1 and 9.2; do not fake an artifact. |
| First race page is slow | FastF1 is downloading/cache-building data. | Keep the backend running, wait for the request, then reuse the cache. |
| `ECONNREFUSED` on `/api` | Backend is not running on port 8787. | Activate venv, run `npm run server`, then test `/api/health`. |
| Frontend appears stale | Vite process is old or on a different port. | Stop the Vite process, restart with `npm run dev -- --host 127.0.0.1`, and use the URL it prints. |
| No data for a future 2026 event | The race has not occurred or public data is unavailable. | Display the no-race-data state; do not produce a fictional replay. |
| FIA alignment warning | Event-specific cited detection/activation evidence is missing. | Keep the result as disclosure/analysis only; import cited official evidence before exact zone claims. |

## 13. Optional cache transfer

Copying `backend/data/f1-cache/` from a trusted existing setup can avoid a long
rebuild, but it is optional and should be treated as a generated artifact:

1. Stop both applications on both machines.
2. Copy the directory through a trusted private channel.
3. Preserve the repository version/commit alongside the cache.
4. On the new laptop, still run the validation commands above.
5. If cache schemas or source code versions differ, delete only the affected
   cache namespace and let the backend regenerate it.

Never commit this cache, virtual environments, populated `.env` files, or
runtime logs to Git.

## 14. Completion checklist for the receiving agent

- [ ] Correct branch checked out and clean source verified.
- [ ] `npm ci` completed.
- [ ] Python 3.13 venv created and data dependencies installed.
- [ ] Optional `.env` files created without committing secrets.
- [ ] Frontend and backend both return HTTP 200.
- [ ] A completed race loads and creates local cache entries.
- [ ] Historical 2018–2025 decision-point extraction completes, if model
      training is required on that laptop.
- [ ] `overtake_rf.joblib` and `overtake_report.json` exist, if model-backed
      action branches are required.
- [ ] `npm run build`, Node syntax check, and Python compile check pass.
- [ ] Simulation is described honestly as future-blind, public-data based,
      pair-only analysis—not a live command or a rewritten historical race.
