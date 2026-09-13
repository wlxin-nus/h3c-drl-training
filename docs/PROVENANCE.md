# Provenance

## Training implementation

This standalone package was assembled from the frozen DRL training implementation at
source commit `5220556d47fdfe2b61163ccf71330faac81b7cb5` in
<https://github.com/wlxin-nus/h3c-drl-multiseed-training>. That source repository may
remain private; the commit is therefore an auditable project-lineage identifier rather
than a guarantee of anonymous public access. Historical checkpoints,
generated results, stability-screen candidates, and the broader H3C application stack
were deliberately excluded. Version 1.0.0 starts a new, publication-focused Git history.

The scientific contract is represented by `TaskSpec`, the packaged case profiles, and
the packaged observation contract. A run records both their SHA-256 configuration hash
and a fingerprint of every Python/JSON runtime file in `drl_multiseed`.

## Simulator

BOPTEST is an external service and is not redistributed here. The reference protocol
uses API version `0.8.0-dev`; `1.0.0-dev` is accepted with a compatibility warning. The
exact BOPTEST repository commit, container image digest, testcase FMU digest, and worker
configuration must be recorded alongside archived experimental results because those
identifiers are deployment-specific and are not recoverable from this source package.

## Scientific versioning

Changes to observations, normalization, reward, comfort, occupancy, action mapping,
network structure, optimizer behavior, rollout semantics, training/test windows,
selection, or stopping require a new protocol version. Existing checkpoints must never
be resumed across such a change.
