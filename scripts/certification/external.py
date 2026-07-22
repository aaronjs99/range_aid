"""Shared validation for pinned external assurance repositories."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess


def _git_value(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).rstrip()


def _normalize_git_url(value: str) -> str:
    value = value.rstrip("/")
    return value[:-4] if value.endswith(".git") else value


def verify_pinned_repository(
    path: Path, expected_url: str, expected_commit: str
) -> None:
    """Reject a dependency unless origin, commit, and worktree match the pin."""
    if _git_value(path, "rev-parse", "HEAD") != expected_commit:
        raise ValueError("repository commit does not match the required pin")
    remote = _git_value(path, "remote", "get-url", "origin")
    if _normalize_git_url(remote) != _normalize_git_url(expected_url):
        raise ValueError("repository origin does not match the official source")
    if _git_value(path, "status", "--porcelain"):
        raise ValueError("repository or its initialized submodules are dirty")
    submodule_status = _git_value(path, "submodule", "status", "--recursive")
    invalid_submodules = [
        line for line in submodule_status.splitlines() if line and line[0] != " "
    ]
    if invalid_submodules:
        raise ValueError(
            "repository has uninitialized, conflicting, or mismatched submodules"
        )


def validate_finite_tree(value, path: str = "result") -> None:
    """Reject non-finite floating-point values before artifact ingestion."""
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite_tree(item, "{}.{}".format(path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite_tree(item, "{}[{}]".format(path, index))
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("{} is non-finite".format(path))
