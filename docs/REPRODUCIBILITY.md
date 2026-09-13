# Reproducibility

## Identities recorded per run

- task and seed;
- complete algorithm configuration;
- case profile, reward, comfort, occupancy, and static-control configuration;
- observation and normalization contract;
- SHA-256 scientific configuration identity;
- behaviorally relevant source-code fingerprint;
- Git commit when available;
- hostname and run UUID;
- Python, OS, dependency, and BOPTEST versions from preflight;
- W&B run ID;
- actual epoch, global step, LR, early-stop reason, and selected best checkpoint.

## Recommended reproduction procedure

1. Archive the repository commit and `CHECKSUMS.sha256`.
2. Record the exact BOPTEST repository commit, Docker image digest, testcase FMU digests, Docker version, CPU/GPU, CUDA runtime, and driver.
3. Use Python 3.10 and the exact direct pins in `requirements.txt`; archive `pip freeze`
   with every reported run to capture the complete platform-specific dependency graph.
4. Run the offline self-check and live preflight.
5. Run smoke mode on the target machine.
6. Train seeds 42, 1337, and 2026 from cold starts.
7. Evaluate each deterministic training-window best on the registered held-out window.
8. Aggregate only runs with matching protocol, source, case, and BOPTEST identities.

## Verify the released paper results

The processed package under `reference_results/paper_2026` provides an offline route for checking
the published DRL values without launching BOPTEST. From the repository root, run:

```console
python -m scripts.verify_reference_results
```

The verifier checks byte identities and row counts, confirms all five task groups and three seeds,
recomputes the mean and sample standard deviation in the summary table, and independently derives
six comfort and setpoint metrics from the released time series. Run
`python -m scripts.recompute_paper_metrics` for the metric comparison alone.

These checks require a complete repository checkout. The processed data and historical snapshot
are not part of the installable runtime wheel.

The released time series do not include power, tariff, or per-step reward. Reward, cost, and energy
can therefore be checked against the audited terminal tables but cannot be independently
reconstructed from the public time series.

Formal evaluation recomputes the installed package fingerprint and requires an exact
match with the training identity before loading a checkpoint. It also requires a
terminal training manifest and an exclusive run lock.

The evaluated checkpoints were created by historical source and configuration identities that
differ from the current publication package. A scientifically compatible historical snapshot is
included under `historical_training_source_287b452`: its source fingerprint and all 15
seed-specific configuration hashes match the retained training manifests. It also contains the
five legacy policies required by the historical offline preflight. The snapshot can be used for
fresh retraining under the historical contract. Direct replay of the published policies is not
supported because the 15 evaluated best checkpoints, the matching held-out-evaluation source
snapshots, and exact BOPTEST image/FMU identities are not all included. Checkpoints supplied
separately must be paired with their training configuration, evaluation source, and simulator
identities.

The aggregation command enforces this boundary: it validates each seed-specific hash
against the packaged task and rejects mixed protocol hashes, source fingerprints, or
BOPTEST versions instead of silently computing a pooled statistic.

## Numerical reproducibility

Seeds and RNG states are controlled, but exact bitwise equality across operating systems, CPU/GPU types, CUDA libraries, and parallel HTTP schedules is not guaranteed. Compare complete configuration identities and multi-seed distributions rather than assuming identical trajectories on different hardware.

## BOPTEST versions

The reference experiments use `0.8.0-dev`. Version `1.0.0-dev` is accepted for practical portability but recorded as non-reference. Do not combine versions in one mean-and-standard-deviation result without first evaluating the same frozen controller on both versions and documenting any drift.

## Release integrity

Run `python scripts/make_checksums.py --verify` after download. Text files are hashed after
canonical LF line-ending normalization so that the same release verifies on Windows and POSIX
systems; binary files are hashed byte for byte. The checksum file excludes generated runs, local
environments, caches, credentials, and W&B data.
