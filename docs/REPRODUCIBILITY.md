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

Formal evaluation recomputes the installed package fingerprint and requires an exact
match with the training identity before loading a checkpoint. It also requires a
terminal training manifest and an exclusive run lock.

The aggregation command enforces this boundary: it validates each seed-specific hash
against the packaged task and rejects mixed protocol hashes, source fingerprints, or
BOPTEST versions instead of silently computing a pooled statistic.

## Numerical reproducibility

Seeds and RNG states are controlled, but exact bitwise equality across operating systems, CPU/GPU types, CUDA libraries, and parallel HTTP schedules is not guaranteed. Compare complete configuration identities and multi-seed distributions rather than assuming identical trajectories on different hardware.

## BOPTEST versions

The reference experiments use `0.8.0-dev`. Version `1.0.0-dev` is accepted for practical portability but recorded as non-reference. Do not combine versions in one mean-and-standard-deviation result without first evaluating the same frozen controller on both versions and documenting any drift.

## Release integrity

Run `python scripts/make_checksums.py --verify` after download. The checksum file excludes generated runs, local environments, caches, credentials, and W&B data.
