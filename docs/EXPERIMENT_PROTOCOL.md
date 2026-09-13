# Experiment Protocol

## Common settings

All five tasks use a 15-minute control interval, four independent BOPTEST training environments, ten PPO optimization epochs per rollout, discount factor 0.99, GAE parameter 0.95, clipping range 0.2, entropy coefficient 0.02, value-loss coefficient 0.5, gradient-norm limit 0.5, and two hidden layers of 256 units.

The occupied comfort penalty uses a PMV deadband of 0.5. Deterministic training-window validation occurs every 25 epochs. Seeds are restricted to 42, 1337, and 2026.

Every reset initializes BOPTEST at the registered training or evaluation start time with
a seven-day internal simulator warm-up. The warm-up trajectory is not returned to the
policy and does not count toward the 480/672 controlled steps. The first observation
therefore represents the requested start time after BOPTEST has propagated its physical
state through the preceding seven days.

## Task settings

| Task | Train day | Test day | Steps/env | Batch | Actor/PPO LR | Critic LR | LR decay horizon |
|---|---:|---:|---:|---:|---:|---:|---:|
| SZ-Air PPO | 196 | 203 | 672 | 168 | 5e-4 | — | 300 epochs |
| MZ-Air PPO | 192 | 199 | 672 | 168 | 3e-4 | — | 300 epochs |
| MZ-Air MAPPO | 192 | 199 | 672 | 168 | 3e-4 | 5e-4 | 300 epochs |
| MZ-Hydronic PPO | 213 | 220 | 480 | 120 | 3e-4 | — | 500 epochs |
| MZ-Hydronic MAPPO | 213 | 220 | 480 | 120 | 3e-4 | 5e-4 | 500 epochs |

Learning rates decrease linearly to 10% of their initial values over the registered decay horizon and remain at that floor afterwards. The 700-epoch cap does not stretch or restart the original schedule.

Hydronic PPO initializes the Gaussian policy standard deviation with `log_std_init=-1`. Air PPO retains the SB3 default `log_std_init=0`. MAPPO actors use `log_std_init=-1`.

## Network and optimizer details

PPO uses Stable-Baselines3 `MlpPolicy` with separate two-layer 256-unit policy and
value branches, Tanh activations, the library's default orthogonal initialization,
and Adam with the Stable-Baselines3 optimizer defaults. No target-KL stop or value
clipping is added.

MAPPO uses independent per-zone actors and one centralized critic. Every hidden
layer is `Linear -> LayerNorm -> ReLU`; linear weights use orthogonal initialization.
Actor output weights use gain 0.01, the critic output uses gain 1.0, and Adam uses
`eps=1e-5`. MAPPO does not use target-KL stopping or value clipping.

The canonical MZ-Air MAPPO run uses done-based GAE and minibatch advantage normalization. MZ-Hydronic MAPPO uses time-limit-safe GAE and rollout-level advantage normalization. This asymmetry is explicit because it matches the registered reported configurations; packaging does not silently promote later screening variants.

## Reward

For `Z` zones, the step reward is

```text
r = -(w_e s_e cost / Z
      + w_c s_c sum_z I[occupied_z] max(0, |PMV_z| - 0.5)^2 / Z
      + w_s s_s sum_z |u_z - u_z,previous| / Z)
```

All cases use weights `w_e=1`, `w_c=20`, and `w_s=0.1`.

| Case | Energy scale | Comfort scale | Smoothness scale |
|---|---:|---:|---:|
| SZ-Air | 120.351459 | 10.0 | 0.179875 |
| MZ-Air | 18.532822 | 10.0 | 0.386932 |
| MZ-Hydronic | 3.0 | 10.0 | 0.1755995 |

Energy cost uses total electric power, the 0.25-hour interval, and the dynamic electricity price. Smoothness is calculated from physical setpoint changes.

## Comfort

PMV is calculated with `pythermalcomfort` using metabolic rate 1.1 met, relative humidity 50%, and air velocity 0.1 m/s. Mean radiant temperature equals zone air temperature in this implementation. `limit_inputs=False` is used, non-finite results are rejected, and the finite PMV value is clipped to `[-3, 3]` before it enters the observation or reward calculation.

SZ-Air and MZ-Air update clothing once per day from the daily mean outdoor temperature: 1.0 clo at or below 10 degrees C, 0.5 clo at or above 26 degrees C, and linear interpolation between those temperatures. MZ-Hydronic uses fixed 0.5 clo.

## Early stopping and selection

The deterministic training-window return is the only model-selection signal. The held-out week remains inaccessible until training ends.

Let `A` be the plateau anchor. A validation is a significant improvement only when

```text
return > A + 0.01 * max(|A|, 1)
```

The anchor and miss counter reset after a significant improvement. Epochs through 100 do not accumulate misses. Three later misses stop training, so epoch 175 is the earliest possible stop. The `best` pointer is updated for every strict numerical maximum, even when the gain is less than 1%.

The preregistered cap is 700 epochs. Optional `--continue-until-converged` execution
beyond 700 is a protocol extension and proceeds in 25-epoch blocks at the LR floor
until the same plateau rule is met.

## Held-out evaluation metrics

All metrics are computed only after a complete deterministic held-out trajectory is
committed. With a 0.25-hour control interval:

```text
energy_kwh = sum(power_total_W * 0.25 / 1000)
cost = sum(energy_step_kwh * PriceElectricPowerDynamic)
occupied_zone_hours = 0.25 * count(occupied and |PMV| > 0.5)
pmv_hours = 0.25 * sum(occupied * max(0, |PMV| - 0.5))
```

The historical key `occupied_zone_hours` therefore denotes occupied PMV-violation
zone-hours, not total occupancy. PMV violation, deep violation (`|PMV| > 0.6`), and
action saturation (`|action| > 0.95`) rates use all occupied zone-steps as their
denominator. Cost inherits the currency unit of the active BOPTEST price profile.

## Checkpoints

One epoch is one transaction: full multi-environment rollout, PPO/MAPPO updates, model and optimizer serialization, integrity verification, latest-pointer update, then log projection. Interrupted partial epochs are discarded and replayed.
