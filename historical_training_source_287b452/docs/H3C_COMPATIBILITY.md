# Refined H3C observation and model compatibility contract

Authoritative upstream source commit:
`23186b2c499e02c042018f71f22ee61b5510b910`.

Read-only legacy registry SHA256:
`d82623ffc58528b92874af0336bef2b93f5f35046e5f6df3da462669c6c0a205`.

Authoritative refined contract: `configs/refined_observation_contracts.json`.
Its canonical SHA256 is checked by preflight.

## Inference path

Both legacy and refined paths construct policy tensors with
`h3c_baselines.policies.observation_contracts.PolicyObservationBuilder` and dispatches
the output through `CentralizedPpoPolicy` or `HierarchicalMappoPolicy`. Refined training
environments pass the new task entry to that same builder directly; they do not
reimplement column order, normalization, history offsets, missing-temperature behavior
or MAPPO local slicing.

The refined model contract is:

| Model | Zone order | Global/local input | Output order | Residual base |
|---|---|---|---|---|
| SZ-Air PPO | `zone1` | 36 / – | `zone1` | fixed 25 °C |
| MZ-Air PPO | `cor,nor,sou,eas,wes` | 96 / – | same order | fixed 25 °C |
| MZ-Air MAPPO | `cor,nor,sou,eas,wes` | 96 / 36 | same order | fixed 25 °C |
| Hydro PPO | `NZ,SZ` | 51 / – | same order | occupied 25 °C, unoccupied 30 °C |
| Hydro MAPPO | `NZ,SZ` | 51 / 36 | same order | occupied 25 °C, unoccupied 30 °C |

Every zone uses current temperature plus four true temperature lags and four previous
physical setpoints. Every case uses four previous whole-system power samples and five
forecast samples at 0, 15, 30, 45 and 60 minutes. Missing temperature lags repeat the
earliest available value. Both MAPPO cases use `zone_then_shared` local layout. Effective
occupancy counts are retained and normalized with testcase-specific capacity bounds.

All outputs are bounded residual actions in `[-1,1]`, multiplied by 5 K and clipped
to 20–30 °C after adding the registered base. Static control points come from the H3C
case profile; the final payload is produced by H3C `control_input()`.

PPO remains an SB3 `MlpPolicy` with separate `[256,256]` Tanh actor and critic paths.
MAPPO actors remain `Linear → LayerNorm → ReLU → Linear → LayerNorm → ReLU → Linear`
with one learned `log_std` per actor. Exported MAPPO bundles retain the `actors` mapping
expected by `HierarchicalMappoPolicy`; PPO exports remain native SB3 `.zip` files.
The exported `model_manifest.json` contains the complete refined registry entry. Old
81/45/33-dimensional checkpoints remain valid only with the legacy registry and are
never accepted as refined resume checkpoints.

## Fail-closed checks

`python -m drl_multiseed.cli preflight --online` verifies:

1. All five archived artifact hashes and legacy dimensions.
2. Legacy H3C model registry identity and golden inference behavior.
3. Refined observation-contract canonical hash.
4. True lag indices, four actions per zone, common missing-temperature rule and forecast offsets.
5. New PPO/MAPPO global/local dimensions and zone/output order.
6. Four training environments, a full episode per environment and batch divisibility.
7. BOPTEST version `0.8.0-dev` or `1.0.0-dev`, with the actual version recorded.

Any failure blocks training. Legacy policy arithmetic remains protected by golden
checks, while the post-warm-up refined models are isolated under `outputs_refine_v2/`, use W&B project
`h3c-drl-multiseed-refine-v2`, and carry the refined and early-stop contracts in their scientific hash.
