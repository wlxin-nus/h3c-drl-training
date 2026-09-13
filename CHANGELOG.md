# Changelog

All notable changes to the publication training package are recorded here.

## Unreleased

- Added processed paper metrics, training histories, and de-identified held-out time series.
- Added independent recomputation of comfort and setpoint-dynamics metrics.
- Added integrity, task/seed, and multi-seed statistical verification for the public result package.
- Linked the H3C, DRL, and MPC research artifacts and documented the public data boundary.
- Documented the distinct training and held-out-evaluation `pythermalcomfort` versions, their
  verified compatibility scope, and the historical checkpoint/source identity boundary.
- Added a scientifically compatible historical training-source snapshot, the 15 verified
  task--seed configuration identities, and the five legacy policies required by its offline
  preflight. The 15 evaluated best checkpoints remain outside the release.

## 1.0.0 - 2026-09-13

- Created a standalone, English-only source release for five registered PPO/MAPPO tasks.
- Packaged the scientific case and observation contracts with the Python distribution.
- Added atomic epoch checkpoints, deterministic model selection, early stopping,
  transient BOPTEST recovery, parallel capacity control, evaluation, and aggregation.
- Added publication metadata, integrity checks, offline regression tests, and a clean
  monitoring notebook.
- Excluded pretrained models, generated runs, historical experiments, stability
  screens, RBC/MPC/H3C application code, and the BOPTEST server distribution.
