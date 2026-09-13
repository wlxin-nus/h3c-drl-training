# Recovery and concurrency semantics

## Atomic epoch transaction

The committed checkpoint—not CSV, TensorBoard or W&B—is the sole progress authority.
Each epoch bundle is written beside its destination, flushed, checksummed and loaded
back before `latest.json` is replaced atomically. Up to five exponential-backoff
attempts are made. Best plus the latest two complete epochs are retained.

If an interruption occurs before pointer replacement, the prior epoch remains latest.
If pointer replacement succeeds before logging, the next launch regenerates local log
projections from the state embedded in the checkpoint.

## Automatic BOPTEST transport recovery

The runner recognizes bounded transient BOPTEST HTTP/socket failures, including
Windows pipe error 109, socket errors 10048/10055 and a PPO worker EOF caused by
a failed HTTP request. It never retries an individual action request: an acknowledgement may
have been lost after the FMU advanced, so doing so could advance the testcase
twice. Instead it closes the failed trainer, stops only TestIDs owned by that run,
waits for the read-only `/version` endpoint, and launches the trainer again with
resume enabled. The last atomically committed epoch is authoritative and the
incomplete epoch is replayed.

The default is 12 consecutive recovery attempts with 15/30/60-second bounded
backoff and a 600-second health timeout. A successfully committed epoch resets the
consecutive counter. `http_auto_resume.jsonl` is the audit log. Non-transport
failures, checkpoint corruption, NaN failures and user interrupts fail normally.

## Ownership

Each run has an exclusive filesystem lock and a persistent UUID. Worker leases contain
hostname, PID, UUID, task, seed and heartbeat. A process only removes another lease or
TestID after the owner is proven dead (or a foreign-host heartbeat is stale). Exiting
one parallel task therefore cannot stop another task's TestIDs.

## Recovery commands

Same host:

```powershell
.\scripts\run_task.ps1 -Task mz_air_mappo -Seed 1337 -Mode full -Resume
```

After copying the complete run directory to another host:

```powershell
.\scripts\run_task.ps1 -Task mz_air_mappo -Seed 1337 -Mode full -Resume -AllowHostMigration
```

Formal evaluation is also transactional. An interrupted 480/672-step trajectory is
not published; the next invocation reruns that controller from step 1.
