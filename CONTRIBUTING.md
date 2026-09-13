# Contributing

This repository represents a frozen scientific protocol. Bug fixes, documentation
improvements, and portability changes are welcome, but changes that alter observations,
normalization, rewards, occupancy, action mapping, network structure, optimizer
settings, training windows, model selection, or early stopping must introduce a new
protocol version and must not resume an existing run.

Before submitting a change:

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python scripts\self_check.py
python -m pytest
python scripts\make_checksums.py
python scripts\make_checksums.py --verify
```

Release maintainers must regenerate `CHECKSUMS.sha256` only after the candidate tree is
frozen, review the checksum diff, rerun the full validation sequence, and commit the
checksum update in the same release commit.

Never commit checkpoints, run outputs, credentials, local service URLs, or TestIDs.
