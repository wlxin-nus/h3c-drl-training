# MZ_Hydro centralized PPO

- Checkpoint: `epoch_0650_steps_001248000.model.zip`
- Training identity: epoch 650, 1,248,000 environment steps
- Inference: one centralized Stable-Baselines3 PPO policy, CPU, deterministic action
- Tensor contract: 45 observations → 2 residual actions, zone order `NZ, SZ`
- History contract: temperature, action and power past slots exclude the latest sample; missing
  history uses the training-time 25 °C / zero defaults
- Action-history observation scale: 15–35 °C, matching the archived training owner; this is
  separate from the physical 20–30 °C action limit
- Action contract: occupied 25 °C / unoccupied 30 °C base plus residual [-5, 5] °C
- Selection: user-designated extended-training checkpoint
- Limitation: the original extended-training report marked convergence false (criterion C1 failed)

The checkpoint is frozen for honest evaluation, not claimed to be fully converged.
