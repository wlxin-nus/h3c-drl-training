# Processed DRL paper results

This directory contains publication-facing data for the PPO and MAPPO baselines in
*Causality-Constrained Hierarchical LLM Agents for Online Rule Adaptation in Building HVAC
Control*.

## Files

| File | Contents |
|---|---|
| `drl_run_metrics.csv` | Nine physical metrics for the 15 evaluated models: five case--algorithm tasks and three seeds. |
| `drl_metric_summary.csv` | Mean, sample standard deviation, and seed-level values for each task and metric. |
| `drl_evaluation_timeseries.csv` | Zone-level setpoint, temperature, PMV, and effective occupancy for every held-out evaluation step. |
| `training_epochs.csv` | Epoch-level training reward for all 15 runs. |
| `training_window_evaluations.csv` | Deterministic training-window evaluations used for checkpoint selection and early stopping. |
| `manifest.json` | SHA-256, byte count, row count, source-file identity, and evaluation-source provenance. |

The registered task groups are `SZ_Air/PPO`, `MZ_Hydro/PPO`, `MZ_Hydro/MAPPO`,
`MZ_Air/PPO`, and `MZ_Air/MAPPO`. Every group contains seeds `42`, `1337`, and `2026`.
The 15 held-out evaluations used the best checkpoint retained during each training run.

## Independent metric check

The released time series independently determine six reported metrics:

- occupied discomfort zone-hours;
- occupied PMV deviation-hours;
- occupied peak absolute PMV;
- setpoint total variation;
- setpoint direction reversals; and
- occupied comfort-band crossings.

Recompute these metrics and compare them with `drl_run_metrics.csv`:

```console
python -m scripts.recompute_paper_metrics
```

Verify file integrity, the five task-by-seed groups, the run-to-summary statistics, and all six
recomputed metrics:

```console
python -m scripts.verify_reference_results
```

The time-series table does not contain power, electricity price, step cost, or step reward.
Consequently, it cannot independently reconstruct reward, cost, or energy use; their audited
terminal values are included in the run and summary tables. Publishing those additional raw
signals would require a separate de-identified trajectory release.

This package also omits BOPTEST test identifiers, local paths, run UUIDs, raw simulator files,
and the 15 evaluated-policy checkpoints. Their historical source and seed-specific configuration
identities differ from the current publication source tree. A compatible snapshot whose
fingerprint and all 15 task--seed hashes match the retained manifests is included under
`historical_training_source_287b452`. Direct replay still requires the missing best checkpoints.

The released PMV values use zone air temperature as both dry-bulb and mean-radiant temperature in
all cases, including `MZ_Hydro`. See the repository
[reproducibility guide](../../docs/REPRODUCIBILITY.md) and
[known limitations](../../docs/KNOWN_LIMITATIONS.md) for the full boundary.
