# Frozen DRL policies

This directory contains inference-only checkpoints used by the independent H3C baselines.
Every load is preceded by byte-count and SHA-256 verification against `registry.json`.
Training code, optimizer state management, notebooks, and experiment trackers are intentionally
outside the release package. Evaluation results never participate in checkpoint selection.

The Hydro checkpoints are the user-designated endpoints of extended training. They are eligible
for evaluation, but their original convergence criteria were not fully satisfied; the per-model
cards preserve that limitation.
