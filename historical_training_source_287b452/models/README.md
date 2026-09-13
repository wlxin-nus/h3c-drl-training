# Historical preflight policies

These five immutable policy files satisfy the offline model-registry checks of the bundled
historical training snapshot. Every load is preceded by byte-count and SHA-256 verification
against `registry.json`.

They are legacy compatibility assets, not the 15 best checkpoints evaluated in the paper. They
must not be used to reproduce or reinterpret the reported PPO/MAPPO results.
