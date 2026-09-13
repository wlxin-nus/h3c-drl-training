# Hardware and parallelism

Each training task always uses four independent BOPTEST environments. PPO owns a
four-process `SubprocVecEnv`; MAPPO submits the four HTTP transitions concurrently.
Changing `MaxParallel` changes the number of **training tasks**, not the number of
environments within a task.

- `MaxParallel 1`: one task, four persistent training TestIDs, temporary fifth
  validation TestID.
- `MaxParallel 2`: two different building cases, eight persistent training TestIDs,
  at most ten while both validate.
- Capacity is fixed at twelve; a third task cannot acquire a lease.
- The scheduler will not co-run PPO and MAPPO for the same testcase. A local smoke
  measurement showed severe FMU contention for that pairing.

PPO defaults to CPU. MAPPO uses CUDA when a GPU slot is available and falls back to
CPU. On RTX 5090/5080 hosts, run the post-install CUDA check from the README before
starting a full run. If it prints `False`, install the PyTorch 2.9.0 CUDA wheel
appropriate for that host and repeat preflight; do not silently run a planned GPU
task on CPU unless the slower execution is intentional.

One seed should normally remain on one machine. If migration is unavoidable, copy
the complete task run directory and use `-AllowHostMigration`; configuration, code,
BOPTEST and checkpoint identities are rechecked before training resumes.
