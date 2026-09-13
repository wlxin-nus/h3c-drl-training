# Known Limitations

- The held-out evaluation windows are five days for MZ-Hydronic and seven days for the air cases. They are not long-term deployment evaluations.
- Only three preregistered seeds are supported by the publication protocol. This is sufficient for reporting variability but not for strong distributional claims.
- Parallel simulator scheduling and GPU kernels can prevent bitwise repeatability across machines.
- BOPTEST `1.0.0-dev` is supported operationally but differs from the `0.8.0-dev` reference environment; mixed-version statistics require a separate drift check.
- The canonical MZ-Air MAPPO configuration preserves the registered done-based GAE and minibatch advantage normalization. A later time-limit-safe GAE screening candidate is not part of this release because it has not been promoted through the full held-out selection protocol.
- For continuity with the registered runs, the time-limit transition exposes the final physical state/history with the final in-episode forecast/time index. This affects only critic bootstrapping at an episode truncation boundary. A corrected terminal-index protocol would constitute a new scientific version and must not be mixed with results from this release.
- The `--continue-until-converged` option has no arbitrary secondary epoch cap. It should be used consistently across seeds and monitored for genuinely continuing improvements.
- All three packaged environments pass zone air temperature as both dry-bulb and
  mean-radiant temperature to the PMV implementation. Any manuscript statement
  describing a different Hydronic radiant-temperature approximation must be reconciled
  with this released implementation.
- Historical training used `pythermalcomfort 3.8.0`, whereas canonical held-out evaluation used
  `3.9.8`. Their PMV kernel and constants were checked as numerically equivalent for the exact
  project call path (`pmv_ppd_iso`, ISO/SI, `limit_inputs=False`, default rounding), not for every
  API path, platform, or future release.
- The Hydronic missing-occupancy fallback uses the calendar origin recorded in its case
  profile. This fallback is invoked only when BOPTEST returns missing occupancy values;
  normal runs use the supplied forecast directly.
- The processed release includes zone-level held-out time series, but not raw BOPTEST trajectories,
  service identifiers, power, tariff, or per-step reward. Reward, cost, and energy therefore cannot
  be reconstructed independently from the released time series.
- The 15 evaluated best checkpoints are not committed to Git. Their historical training source
  fingerprint is `aca13c42236511fc7e428ce0c9f2969c39d53779cc2d5c9880dcaf8e2cf46272`,
  which differs from the publication source fingerprint. The seed-specific scientific
  configuration identities also differ. Consequently, the checkpoints alone would not pass the
  release evaluator's identity checks. Direct replay requires the checkpoints together with their
  matching historical source and configuration package. The compatible historical source and all
  15 task--seed configuration identities are bundled, but the evaluated checkpoints are not.
