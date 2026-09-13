# H3C DRL Training

Reproducible training and evaluation code for the five deep reinforcement-learning baselines used
in *Causality-Constrained Hierarchical LLM Agents for Online Rule Adaptation in Building HVAC
Control*. The repository trains centralized PPO and multi-agent PPO (MAPPO) policies against an
externally managed BOPTEST service.

The release includes processed paper results, de-identified evaluation time series, and a
scientifically compatible snapshot of the historical training source. It excludes the 15
evaluated-policy checkpoints, raw BOPTEST trajectories, and BOPTEST server images. The historical
snapshot retains the compatibility modules and five legacy policies required by its offline
preflight; these are not the paper's 15 evaluated policies or standalone companion implementations.

## Research artifact family

| Artifact | Scope | Repository |
|---|---|---|
| H3C | Hierarchical Agent framework, rule admission, execution, and Agent results | [ideas-lab-nus/H3C](https://github.com/ideas-lab-nus/H3C) |
| DRL training | PPO/MAPPO training, evaluation, and DRL reference results | This repository |
| MPC training | ARX identification, hierarchical MPC validation, and frozen MPC models | [wlxin-nus/building-mpc-training](https://github.com/wlxin-nus/building-mpc-training) |

## Scope

The release contains:

- five frozen DRL task definitions across three BOPTEST cases;
- a common observation, normalization, reward, occupancy, and action contract;
- four-environment PPO and MAPPO training;
- deterministic training-window model selection and preregistered early stopping;
- atomic checkpoints, exact epoch-boundary resume, and transient HTTP recovery;
- serial or two-task parallel scheduling;
- W&B, TensorBoard, CSV, JSON, and JSONL logging;
- deterministic held-out evaluation and multi-seed aggregation;
- processed run metrics, training histories, and de-identified held-out time series;
- the historical source/configuration snapshot whose fingerprint and all 15 task--seed hashes
  match the retained training manifests;
- offline unit tests and a read-only monitoring notebook.

It does **not** install or start BOPTEST. Provision BOPTEST separately before running a live smoke test or full training.

## Registered tasks

| Task key | Case | Algorithm | Global observation | Local observation | Action | Episode | Maximum |
|---|---|---|---:|---:|---:|---:|---:|
| `sz_air_ppo` | BESTEST single-zone four-pipe FCU | PPO | 36 | — | 1 | 672 steps (7 days) | 700 epochs |
| `mz_air_ppo` | Multizone office air | PPO | 96 | — | 5 | 672 steps (7 days) | 700 epochs |
| `mz_air_mappo` | Multizone office air | MAPPO | 96 | 36 per actor | 5 | 672 steps (7 days) | 700 epochs |
| `mz_hydro_ppo` | Multizone office hydronic | PPO | 51 | — | 2 | 480 steps (5 days) | 700 epochs |
| `mz_hydro_mappo` | Multizone office hydronic | MAPPO | 51 | 36 per actor | 2 | 480 steps (5 days) | 700 epochs |

Each task uses four independent training environments. A fifth TestID is created only for deterministic training-window validation. PPO uses a four-process `SubprocVecEnv`; MAPPO owns four independent environments and submits their BOPTEST calls concurrently.

The allowed study seeds are `42`, `1337`, and `2026`. Each seed is a cold start and never inherits another seed's model or optimizer state.

## External requirements

- 64-bit Python 3.10;
- Git;
- an accessible BOPTEST service with these provisioned test cases:
  - `bestest_air`
  - `multizone_office_simple_air`
  - `multizone_office_simple_hydronic`
- BOPTEST API version `0.8.0-dev` (reference) or `1.0.0-dev` (supported with a warning);
- 12 BOPTEST workers for two concurrent training tasks;
- optional NVIDIA GPU for MAPPO. PPO normally runs on CPU.

The local capacity manager reserves five worker slots per task. Two simultaneous tasks therefore peak at ten TestIDs, leaving two workers for service overhead and recovery. Capacity leases coordinate processes on one checkout; they do not coordinate multiple computers sharing one BOPTEST server.

## Quick start on Windows

### 1. Install Python and Git

If Python 3.10 is not already available:

```powershell
winget install --id Python.Python.3.10 --exact --source winget
winget install --id Git.Git --exact --source winget
```

Open a new PowerShell and verify:

```powershell
py -3.10 --version
git --version
```

Conda is not required.

### 2. Clone and create a virtual environment

```powershell
git clone https://github.com/wlxin-nus/h3c-drl-training.git
Set-Location h3c-drl-training
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Choose the PyTorch build before installing the remaining packages. For CPU training,
the standard PyPI build is sufficient:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For NVIDIA execution, instead install the official PyTorch 2.9.0 wheel selected for the
host CUDA runtime from <https://pytorch.org/get-started/locally/> first, then run the two
commands above. `requirements.txt` pins the direct reference dependencies. Archive
`python -m pip freeze` with every run, and do not mix different Torch builds within one
reported multi-seed aggregate.

If PowerShell blocks local scripts, enable them only for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 3. Point the client at BOPTEST

The default URL is port 8000:

```powershell
$env:BOPTEST_URL = "http://127.0.0.1:8000"
Invoke-RestMethod "$env:BOPTEST_URL/version" | ConvertTo-Json -Depth 5
```

Use the actual address if BOPTEST runs on another host or port. If BOPTEST is already
running with 12 workers, no Docker command is needed in this repository. Otherwise,
change to the separate BOPTEST checkout (whose Compose service names may differ), start
the workers there, and then return to this repository. For example:

```powershell
Push-Location C:\path\to\project1-boptest
docker compose up -d --scale worker=12
docker compose ps
$workerCount = (docker compose ps -q worker | Measure-Object).Count
if ($workerCount -ne 12) { throw "Expected 12 running BOPTEST workers; found $workerCount" }
Pop-Location
```

Replace `worker` with the Compose service name used by the external BOPTEST checkout.
The expected count is 12 before launching a two-task suite.

### 4. Optional W&B login

```powershell
wandb login
$env:WANDB_PROJECT = "h3c-drl-paper-reproduction"
```

No API key belongs in this repository. Use `-WandbMode offline` or `disabled` when online tracking is unavailable. Local logs and checkpoints are always written.

### 5. Validate the checkout

```powershell
python scripts\make_checksums.py --verify
python scripts\self_check.py
python -m drl_multiseed.cli preflight --online --seed 1337 --endpoint $env:BOPTEST_URL
```

The first two commands are offline. The third checks the live BOPTEST version and writes `runs/preflight.json`.

### 6. Run a smoke test

Always smoke-test the exact machine and endpoint before a full run:

```powershell
.\scripts\run_seed_suite.ps1 `
  -Seed 1337 -Mode smoke -MaxParallel 2 -GpuSlots 0 `
  -ThreadsPerTask 4 -Resume -WandbMode online `
  -Endpoint $env:BOPTEST_URL
```

Smoke and full modes use separate directories. Full training never inherits smoke weights.
The command above is CPU-safe. Set `-GpuSlots 1` only when PyTorch detects a usable
CUDA device; otherwise keep `-GpuSlots 0` and MAPPO will run on CPU.

### 7. Run full training

Two-task scheduler on a strong machine:

```powershell
.\scripts\run_seed_suite.ps1 `
  -Seed 1337 -Mode full -MaxParallel 2 -GpuSlots 1 `
  -ThreadsPerTask 4 -Resume -WandbMode online `
  -Endpoint $env:BOPTEST_URL
```

Serial scheduler on a weaker machine:

```powershell
.\scripts\run_seed_suite.ps1 `
  -Seed 1337 -Mode full -MaxParallel 1 -GpuSlots 0 `
  -ThreadsPerTask 4 -Resume -WandbMode online `
  -Endpoint $env:BOPTEST_URL
```

Select a subset with `-Tasks`. For example, MZ-Air PPO and MAPPO are scheduled sequentially because they use the same testcase:

```powershell
.\scripts\run_seed_suite.ps1 `
  -Seed 1337 -Mode full -MaxParallel 2 -GpuSlots 1 -Resume `
  -Tasks mz_air_ppo,mz_air_mappo `
  -Endpoint $env:BOPTEST_URL
```

To run exactly one task:

```powershell
.\scripts\run_task.ps1 `
  -Task mz_hydro_ppo -Seed 1337 -Mode full -Resume `
  -Device cpu -WandbMode online -Endpoint $env:BOPTEST_URL
```

## Resume and automatic recovery

Keep `-Resume` on training commands. Each completed epoch is one transaction: a complete rollout and PPO update are checkpointed before the epoch counter and logs advance.

After `Ctrl+C`, a reboot, power loss, or process crash, restore the same checkout and run directory, activate the same environment, verify BOPTEST, and rerun the identical command. Training resumes from the latest committed epoch. At most one incomplete epoch is replayed.

Transient BOPTEST/HTTP failures—including connection reset, timeout, Windows socket errors `10048`/`10055`, and worker pipe error `109`—trigger bounded automatic recovery. The runner:

1. retains the last valid atomic checkpoint;
2. stops only TestIDs owned by the failed run;
3. waits for the BOPTEST health endpoint;
4. resumes with the same optimizer, LR schedule, RNG state, early-stop state, global step, and W&B run ID.

Checkpoint files are the only authority for progress. CSV or W&B history never advances model state.

To move a complete run directory to another machine, copy the entire task directory and pass `-AllowHostMigration`. Configuration and source fingerprints must still match.

See [Recovery and checkpoint semantics](docs/RECOVERY.md).

## Early stopping and model selection

Every 25 epochs, an independent TestID runs one deterministic rollout on the training window. The held-out evaluation window is not accessed during training or checkpoint selection.

- return is maximized (`-500` is better than `-600`);
- epochs through 100 are warm-up and do not consume patience;
- a plateau anchor resets only after at least 1% relative improvement;
- three post-warm-up validations without such improvement stop training;
- any new numerical maximum updates `best`, even when the gain is below 1%;
- final evaluation loads `best`, not `latest`.

The first possible early stop is epoch 175. The rule is intentionally aggressive and may miss slow late-stage improvement. The preregistered cap is 700 epochs. Passing `-ContinueUntilConverged` is an explicit protocol extension: training then proceeds in 25-epoch blocks at the fixed 10% LR floor until the same plateau condition is met. Record this deviation and use it consistently across every seed in a reported comparison.

## Deterministic held-out evaluation

After training, evaluate one task:

```powershell
.\scripts\evaluate_task.ps1 `
  -Task mz_hydro_ppo -Seed 1337 -Mode full `
  -Device cpu -Endpoint $env:BOPTEST_URL
```

Registered evaluation windows are:

- SZ-Air: day 203, 672 steps;
- MZ-Air: day 199, 672 steps;
- MZ-Hydronic: day 220, 480 steps.

To evaluate all five tasks for all registered seeds after their training runs finish:

```powershell
$tasks = @("sz_air_ppo", "mz_air_ppo", "mz_air_mappo", "mz_hydro_ppo", "mz_hydro_mappo")
foreach ($seed in 42, 1337, 2026) {
  foreach ($task in $tasks) {
    .\scripts\evaluate_task.ps1 `
      -Task $task -Seed $seed -Mode full -Device cpu `
      -Endpoint $env:BOPTEST_URL
  }
}
```

Run formal evaluation serially unless the external simulator operator has explicitly
validated a higher evaluation load.

Evaluation commits a trajectory only if every timestamp and step is complete. Aggregate completed seeds with:

```powershell
python -m drl_multiseed.cli aggregate --mode full
```

The evaluator reports cost, energy, occupied PMV-violation zone-hours, PMV-hours, PMV
violation rates, return, action saturation, best epoch, and actual training steps.
Aggregation verifies every seed-specific configuration hash and refuses to pool runs
whose protocol, source fingerprint, or BOPTEST version differs.

KPI definitions use 15-minute zone-steps:

- `cost` is the sum of `energy_step_kwh * PriceElectricPowerDynamic`; its currency unit
  is inherited from the selected BOPTEST tariff profile;
- `energy_kwh` is total electric energy over the complete evaluation trajectory;
- `occupied_zone_hours` is retained as a machine-readable compatibility key but means
  occupied **PMV-violation** zone-hours, namely `0.25 * count(occupied and |PMV| > 0.5)`;
- `pmv_hours` is `0.25 * sum(occupied * max(0, |PMV| - 0.5))` across all zones;
- occupied PMV and deep-PMV rates divide the respective zone-step counts by all
  occupied zone-steps, with the deep threshold set to `|PMV| > 0.6`;
- occupied action saturation divides occupied zone-steps with `|action| > 0.95` by all
  occupied zone-steps.

The aggregate also writes `drl_information_contract.csv`. It documents only the five
DRL baselines; cross-method information fairness with H3C must be established using the
companion H3C method archive.

## Verify the paper results

Processed paper results are released under
[`reference_results/paper_2026`](reference_results/paper_2026/README.md). The package contains 15 run-level
rows, five task-by-seed groups, training curves, training-window evaluations, and de-identified
held-out time series.

Run the commands below from a complete Git checkout. The research data and historical source
snapshot are repository assets and are not installed by the runtime wheel.

Recompute occupied comfort and setpoint-dynamics metrics from the released time series:

```powershell
python -m scripts.recompute_paper_metrics
```

Verify hashes, row counts, task/seed coverage, multi-seed summary statistics, and the independently
recomputed metrics:

```powershell
python -m scripts.verify_reference_results
```

The released time series determine discomfort zone-hours, PMV deviation-hours, occupied peak
absolute PMV, setpoint total variation, reversals, and occupied comfort-band crossings. They do
not contain the power, electricity-price, step-cost, or step-reward signals needed to reconstruct
reward, cost, or energy; the audited terminal values for those metrics remain available in the
run and summary tables.

## Outputs

Generated files are ignored by Git and written under:

```text
runs/{smoke|full}/seed{seed}/{task}/
```

Important files include:

```text
run_identity.json
preflight.json
run_manifest.json
training_metrics.csv
updates.csv
train_window_eval.csv
checkpoints/latest.json
checkpoints/best.json
tensorboard/
wandb/
lifecycle/*.jsonl
formal_evaluation/trajectory.csv
formal_evaluation/metrics.json
artifacts/model_manifest.json
```

The code retains the two latest checkpoints, the deterministic best, and the final checkpoint references. TestID lifecycle records are local JSONL files and are never sent as dynamic W&B tables during training.

## Scientific configuration

The machine-readable sources of truth are:

- `src/drl_multiseed/config.py` for algorithm and schedule parameters;
- `src/drl_multiseed/data/cases/*.json` for BOPTEST points, occupancy, comfort, reward scales, and physical controls;
- `src/drl_multiseed/data/observation_contract.json` for temporal features, dimensions, and normalization.

A checkpoint configuration hash covers all three. A separate source fingerprint covers behaviorally relevant training, environment, comfort, occupancy, observation, and protocol modules. Resume is rejected if either identity changes after an epoch has been committed.

See [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) and [Observation and action spaces](docs/OBSERVATION_AND_ACTION_SPACES.md) for the exact equations and case-specific exceptions.
Source and simulator lineage are documented in [Provenance](docs/PROVENANCE.md).

## Linux and macOS

PowerShell scripts are thin wrappers. After creating and activating a Python 3.10 virtual environment, use the CLI directly:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/self_check.py

python -m drl_multiseed.cli train \
  --task mz_hydro_ppo --seed 1337 --mode full --resume \
  --endpoint http://127.0.0.1:8000 --device cpu --wandb-mode online
```

For a CPU-only two-task suite:

```bash
python -m drl_multiseed.cli suite \
  --seed 1337 --mode full --max-parallel 2 --gpu-slots 0 \
  --threads-per-task 4 --resume --wandb-mode online \
  --endpoint http://127.0.0.1:8000
```

## Repository layout

```text
docs/                    Protocol, recovery, parallelism, and reproducibility notes
notebooks/               Read-only training monitor
scripts/                 PowerShell launchers and integrity checks
src/drl_multiseed/       Training code and packaged case/observation contracts
tests/                   Offline regression tests
reference_results/       Processed paper results and integrity metadata
```

## Reproducibility boundaries

- BOPTEST is external and must be cited and versioned independently.
- The reference experiments used BOPTEST `0.8.0-dev`; version `1.0.0-dev` is accepted but must not be pooled without a compatibility check.
- GPU kernels and parallel HTTP scheduling can prevent bit-for-bit equality across hardware. The protocol targets configuration and statistical reproducibility, not identical floating-point trajectories across machines.
- The registered 5-day and 7-day held-out windows are short-period evaluations and should not be described as long-term deployment tests.
- The canonical MZ-Air MAPPO task preserves the registered done-based GAE and minibatch advantage normalization. Unreported screening candidates are intentionally excluded from this release.
- The processed results were produced by historical runs whose source and seed-specific
  configuration identities differ from the current publication source tree. The compatible
  historical source/configuration snapshot is included under
  `historical_training_source_287b452`, and the repository independently recomputes six trajectory
  metrics. Direct replay of the evaluated policies still requires the 15 matching best
  checkpoints, which are not included.
- The published PMV time series use zone air temperature as both dry-bulb and mean-radiant
  temperature in all three cases, including `MZ_Hydro`.
- Historical training used `pythermalcomfort 3.8.0`, while the canonical held-out evaluation used
  `3.9.8`. The PMV kernel and constants were checked as numerically equivalent for this project's
  exact `pmv_ppd_iso` call path; this does not imply bitwise equality across platforms.

See [Reproducibility](docs/REPRODUCIBILITY.md), [Security](SECURITY.md), and [Known limitations](docs/KNOWN_LIMITATIONS.md).

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). Add the article DOI to a tagged archival release when it becomes available.

The code is distributed under the revised BSD terms in [LICENSE](LICENSE). Third-party software and service notices are summarized in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
