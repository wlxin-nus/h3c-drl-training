# Parallel Execution

## Within one task

- PPO creates four Windows-safe spawned subprocess environments through `SubprocVecEnv`.
- MAPPO owns four environment objects and performs their reset and step HTTP calls concurrently.
- Each environment receives an independent TestID.
- Deterministic training-window validation temporarily uses a fifth TestID.

Four workers reduce rollout wall time but do not reduce the number of samples per epoch. Changing to one worker changes batch composition, optimizer updates per epoch, and the meaning of the registered epoch count; it is therefore not equivalent to this protocol.

## Across tasks

The suite scheduler supports one or two concurrent tasks. Each task reserves five slots. With a local capacity of 12, two tasks use at most ten slots.

Tasks sharing the same BOPTEST testcase are not launched together because two policies competing for the same FMU worker pool can sharply reduce throughput. A typical pair is one CPU PPO task for one building and one GPU MAPPO task for another building.

`GpuSlots=1` permits one MAPPO process on the GPU. `GpuSlots=0` forces suite MAPPO tasks to CPU. `GpuSlots=2` permits two MAPPO tasks to share one GPU and should be used only after checking memory and utilization.

Formal evaluation acquires the same run lock as training and a one-worker capacity
lease. It refuses to start until the selected training manifest is terminal, so an
accidental evaluation command cannot stop or reuse an active trainer's TestIDs.

## Multiple computers

Use an independent BOPTEST deployment per computer. File-based leases coordinate only processes that see the same `runs/_runtime` directory; they cannot protect a shared remote 12-worker service from clients on multiple computers.

Do not train two machines into a shared network directory. Give each machine its own clone and run directory, then merge complete `runs/full/seed*/` directories after training.
