# Observation and Action Spaces

## Temporal convention

One index is 15 minutes. Every task uses:

- zone temperature at `t, t-1, t-2, t-3, t-4`;
- physical setpoint history at `t-1, t-2, t-3, t-4`;
- normalized site power at `t-1, t-2, t-3, t-4`;
- outdoor temperature, solar irradiance, dynamic electricity price, and per-zone occupancy forecasts at `t, t+1, t+2, t+3, t+4` (0 to 60 minutes);
- current per-zone PMV;
- sine and cosine of time of day.

At episode start, missing temperature history repeats the earliest physical temperature, missing actions use 25 degrees C, and missing power uses zero. The PMV feature is initialized to zero before the first physical transition and is then updated from each observed zone temperature.

The global dimension is `21 + 15 * number_of_zones`: 36 for SZ-Air, 96 for MZ-Air, and 51 for MZ-Hydronic. Each MAPPO actor receives a 36-dimensional `zone_then_shared` local vector; the centralized critic receives the complete global vector.

## Normalization bounds

| Feature | Lower | Upper |
|---|---:|---:|
| Time sine/cosine | -1 | 1 |
| Zone temperature | 288.15 K | 308.15 K |
| Setpoint history | 293.15 K | 303.15 K |
| Normalized power | 0 | 1 |
| Outdoor temperature | 263.15 K | 313.15 K |
| Solar irradiance | 0 W/m2 | 1200 W/m2 |
| Electricity price | 0 | 0.2 |
| PMV | -3 | 3 |

Occupancy upper bounds are 2 for SZ-Air, 50 for MZ-Air, and 200 for MZ-Hydronic. Symmetric min-max normalization maps configured bounds to `[-1,1]`. Values are not clipped after normalization.

The exception is upstream of normalization: the finite value returned by the comfort
model is clipped to the declared physical PMV range `[-3,3]`. This same clipped PMV is
used by the reward and the observation builder. Other features are not clipped after
normalization.

MZ-Air occupancy is the forecast person count only when the count is positive and the official HVAC occupancy window `[06:00, 19:00)` is active. SZ-Air and MZ-Hydronic use positive forecast counts directly.

## Actions

The environment accepts one scalar per zone and clips every sample to `[-1,1]` before mapping it to a physical setpoint. A stochastic Gaussian policy can draw a pre-clipped sample outside this interval; deterministic MAPPO actor means are tanh-bounded.

Air cases use:

```text
setpoint_C = clip(25 + 5 * action, 20, 30)
```

MZ-Hydronic uses an occupancy-conditioned residual base:

```text
base_C = 25 if occupied else 30
setpoint_C = clip(base_C + 5 * action, 20, 30)
```

Thus the occupied Hydronic range is 20 to 30 degrees C, while the unoccupied effective range is 25 to 30 degrees C. Static control points and cooling-setpoint actuators are defined in `src/drl_multiseed/data/cases/*.json`. No H3C auxiliary overwrite points are sent.

## Forecast-horizon terminology

Case JSON uses `forecast_steps=4` to mean four future steps beyond the current sample. The algorithm contract uses five forecast samples, including the current sample. Both represent the same 0-to-60-minute input horizon.
