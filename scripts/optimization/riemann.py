"""Riemannian backend placeholder.

This module intentionally mirrors the public behavior of the full nonlinear
backend for now. It exists as the future home for a low-rank/Riemannian
implementation while keeping backend selection and reporting stable.
"""

from __future__ import annotations

from scripts.configuration.config import SimConfig
from scripts.optimization import full


def simulate_and_estimate(cfg: SimConfig) -> dict:
    """Run the full backend until the Riemannian solver is implemented."""
    data = full.simulate_and_estimate(cfg)
    data["optimizer_backend_note"] = (
        "riemann placeholder backend; currently delegates to full SciPy "
        "least-squares optimization"
    )
    return data
