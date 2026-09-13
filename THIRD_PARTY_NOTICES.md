# Third-Party Notices

This repository depends on third-party software but does not vendor those packages.
Their licenses apply independently.

| Component | Purpose | Project |
|---|---|---|
| BOPTEST | Building emulation service and testcase interfaces | <https://github.com/ibpsa/project1-boptest> |
| Stable-Baselines3 | PPO implementation | <https://github.com/DLR-RM/stable-baselines3> |
| PyTorch | Neural-network and optimization runtime | <https://pytorch.org/> |
| Gymnasium | Environment interface | <https://gymnasium.farama.org/> |
| pythermalcomfort | Fanger PMV calculation | <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort> |
| Weights & Biases | Optional experiment tracking | <https://wandb.ai/> |

The BOPTEST service, Docker images, FMUs, and testcase source are intentionally not
included. This package contains a client for the public BOPTEST HTTP API plus case-point
mappings for externally provisioned testcases; it does not vendor BOPTEST source code.
The training client was assembled from the project lineage identified in
[Provenance](docs/PROVENANCE.md), and BOPTEST attribution is retained in
[LICENSE](LICENSE).

The complete dependency versions resolved on a training host should be archived with
`python -m pip freeze`. See each upstream distribution for its authoritative license
text.
