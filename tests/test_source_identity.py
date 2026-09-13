from __future__ import annotations

from drl_multiseed import source_identity


def test_installed_package_does_not_borrow_current_directory_git_identity(monkeypatch) -> None:
    monkeypatch.setattr(source_identity, "source_checkout_root", lambda: None)
    assert source_identity.code_commit() == "installed-package-no-vcs"
