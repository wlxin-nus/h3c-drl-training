# Historical paper-training source

This directory is a scientifically compatible snapshot of the historical source and
configuration used for the 15 PPO/MAPPO training runs reported in the paper. The snapshot is
pinned to source commit `287b452c2874a36cf2777bd31696192b3622d706` and source fingerprint
`aca13c42236511fc7e428ce0c9f2969c39d53779cc2d5c9880dcaf8e2cf46272`.

The archive contains the original DRL package, its H3C compatibility modules, case and observation
contracts, training scripts, tests, environment specifications, and the exact registry and five
legacy policy files required by the historical offline preflight. `ORIGINAL_README.md` preserves
the documentation from the historical source tree. The main repository README is the current
publication guide.

The 15 seed-specific scientific configuration hashes are recorded in `contract.json`. Run the
repository-level verifier to recompute the source fingerprint and all 15 hashes from this archive:

```console
python -m scripts.verify_reference_results
```

The source fingerprint and all 15 seed-specific configuration hashes match the retained training
manifests. This compatibility is verified by the repository-level checker; the snapshot is not
presented as the exact Git object recorded by every run manifest.

The 15 best checkpoints evaluated in the paper are not included. The archive supports the
historical offline preflight and fresh retraining against an external BOPTEST service, but direct
replay of the evaluated policies additionally requires those matching checkpoints.

The archive intentionally excludes generated runs, raw simulator data, large legacy notebooks,
and large legacy notebooks. The five bundled legacy policy files serve only the historical
preflight and are not the 15 evaluated paper checkpoints. No command in the repository CI
contacts BOPTEST or another external service.
