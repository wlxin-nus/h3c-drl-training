from __future__ import annotations

import importlib.util


def test_required_upstream_runtime_packages_are_packaged() -> None:
    required = ("h3c.runtime.clients", "h3c.runtime.protocol")
    for module in required:
        assert importlib.util.find_spec(module) is not None
