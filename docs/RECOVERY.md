# Recovery and Checkpoint Semantics

## Transaction boundary

The recoverable unit is a complete epoch, not a 15-minute simulator step. BOPTEST cannot reliably recreate an arbitrary partially advanced FMU state, so an interrupted partial epoch is discarded and replayed from the last committed checkpoint.

PPO checkpoints contain the SB3 policy/value model, optimizer, LR schedule, Python/NumPy/PyTorch RNG states, global step, committed epoch, best state, deterministic validation history, early-stop state, W&B ID, scientific hash, and source fingerprint.

MAPPO checkpoints additionally contain every actor, the centralized critic, all optimizers and LR schedulers, the rollout RNG, GAE mode, and advantage-normalization mode.

## Atomic commit

For each epoch, files are written beside their final destination under unique temporary names. The manager flushes them, reloads the model or bundle, verifies the epoch and checksums, renames files, and atomically updates `latest.json`. Five exponentially backed-off attempts are allowed. Failure leaves the previous latest pointer unchanged.

The latest two complete checkpoints and the checkpoint referenced by `best.json` are retained.

## Manual resume

Rerun the identical command with `-Resume` or `--resume`. Do not edit CSV files or pointer JSON. The checkpoint is authoritative and restores the log projection if necessary.

Only use `-AllowHostMigration` after copying the complete task run directory. The new host must have the same scientific configuration, source fingerprint, Python environment, and compatible BOPTEST service.

## HTTP recovery

Action calls are never retried in place because the FMU may have advanced even if the response was lost. A recognized transport error exits the current trainer attempt, cleans only TestIDs owned by that run, waits for `/version`, and launches epoch-boundary resume.

Recovery events appear in `http_auto_resume.jsonl`; the latest unresolved failure appears in `last_failure.json`.

## Ownership safety

Every TestID is persisted immediately after selection. Each lifecycle file is namespaced by run and environment rank. Cleanup operates only inside the requested run directory. Cross-process capacity and run locks prevent a second process from resuming the same run and prevent a third five-worker task from exceeding a 12-worker local capacity declaration.
