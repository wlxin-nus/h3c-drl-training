# Confirmed graph configurations

This directory stores canonical human-confirmed graphs and declarative
single-difference variants. Every confirmed graph records reviewer, date, source,
schema identity, and stable edge identifiers.

`mutations.json` is the single owner of the registered missing-edge and
qualitative timing-tag differences consumed by both the CLI and suite planner.
The suite contract selects the applicable timing direction per profile:
SZ_Air and MZ_Air change Immediate to Delayed, while MZ_Hydro changes its
canonical Delayed tag to Immediate.
