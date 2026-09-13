# MZ_Air hierarchical MAPPO

- Checkpoint: `mappo_best.pt`
- Training identity: epoch 298, 801,024 environment steps
- Inference: five independent actor means on CPU; central critic is not used
- Tensor contract: global 81 observations, local 33 observations per actor
- Policy zone order: `cor, nor, sou, eas, wes`
- Local actor order: time, shared power/weather/price, own temperature/PMV/action, own raw-binary
  occupancy; histories include the latest sample
- Time-feature bounds: `[0, 1]` before symmetric min-max normalization, as executed by the
  training-machine evaluator
- Action contract: 25 °C base plus residual [-5, 5] °C, clamped to [20, 30] °C
- Selection: highest training-log mean return; no formal evaluation result was used

## Provenance recovery

The training-computer copy binds the epoch-298 checkpoint, the `Visfinal` evaluator and
`mappo_validation_air_5zone.csv` by path and file chronology. More importantly, restoring the
evaluator's `[0, 1]` fallback bounds for the two sin/cos columns makes this checkpoint reproduce
the archived first five raw actions within approximately `2e-6`. The earlier failure to reproduce
that action was caused by the migrated adapter's global `[-1, 1]` time bound, not by missing actor
bytes. The source-accounted action oracle is enforced in the test suite.

The immutable byte count and SHA-256 are owned by `models/registry.json`.
