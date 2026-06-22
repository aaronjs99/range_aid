from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_unknown_backend_lists_only_real_backends() -> None:
    root = Path(__file__).resolve().parents[1]
    previous_scripts_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "scripts" or name.startswith("scripts.")
    }
    for name in previous_scripts_modules:
        del sys.modules[name]
    sys.path.insert(0, str(root))
    try:
        from scripts.optimization import simulate_and_estimate

        cfg = SimpleNamespace(optimizer_backend="riemann")

        with pytest.raises(ValueError, match="expected 'full' or 'cora'"):
            simulate_and_estimate(cfg)  # type: ignore[arg-type]
    finally:
        sys.path = [entry for entry in sys.path if entry != str(root)]
        for name in list(sys.modules):
            if name == "scripts" or name.startswith("scripts."):
                del sys.modules[name]
        sys.modules.update(previous_scripts_modules)
