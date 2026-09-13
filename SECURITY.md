# Security Policy

## Supported version

Security fixes are applied to the latest tagged release.

## Reporting a vulnerability

Do not include credentials, private W&B links, BOPTEST TestIDs, or proprietary
building data in a public issue. Use the repository's
[private security-advisory form](https://github.com/wlxin-nus/h3c-drl-training/security/advisories/new)
and include the affected release, operating system, Python version, and a minimal
reproduction that contains no secrets. If GitHub private reporting is unavailable,
contact the corresponding author through the contact channel published with the H3C
article rather than opening a public issue.

## Checkpoint trust boundary

PPO and MAPPO checkpoints use PyTorch/Stable-Baselines3 serialization. Loading an
untrusted checkpoint can execute malicious pickle payloads. Only load checkpoints
created by this repository or obtained from a trusted archival source, and verify
their SHA-256 manifests before evaluation or resume.

## Credentials

W&B credentials and service tokens must be provided through the user's environment
or the official client login flow. They must never be committed to this repository.
`.env`, run outputs, W&B state, and virtual environments are excluded from Git and
release checksums.
