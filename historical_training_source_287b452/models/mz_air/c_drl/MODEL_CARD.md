# MZ_Air centralized PPO

- Checkpoint: `best_model_ppo.zip`
- Epoch: 281
- Embedded training counter: 755,328 environment steps
- Inference: one centralized Stable-Baselines3 PPO policy, CPU, deterministic action
- Tensor contract: 81 observations → 5 residual actions
- Policy zone order: `cor, nor, sou, eas, wes` (explicitly different from profile order)
- Action contract: 25 °C base plus residual [-5, 5] °C, clamped to [20, 30] °C
- Observation contract: temperature/action/power histories include the latest sample; occupancy
  is the raw forecast converted to a binary mask without the later official-HVAC-window filter
- Selection: maximum archived training-log mean reward (`-682.13884274`) in the retained final
  training directory; no new evaluation result was used to select or modify it

## Provenance limitation

The checkpoint is the retained `best_model_ppo.zip` from the complete final training archive and is
the strongest available identity for the paper-era C-DRL. The legacy
`drl_validation_air_5zone.csv` nevertheless records neither a checkpoint hash nor its pre-action
observation, and its first action is not reproduced under the saved `FinalMZAIR` tensor contract.
New results must therefore be described as a reconstruction with the best-supported archived
checkpoint, not as a cryptographically exact replay of that CSV.

The archived model does not prove its random seed; the card therefore makes no seed claim.
