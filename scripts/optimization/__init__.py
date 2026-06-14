"""Optimizer backend selection."""

from __future__ import annotations

from scripts.configuration.config import SimConfig
from scripts.optimization import cora, full, riemann


def simulate_and_estimate(cfg: SimConfig) -> dict:
    """Run the configured optimizer backend."""
    if cfg.optimizer_backend == "full":
        return full.simulate_and_estimate(cfg)
    if cfg.optimizer_backend == "cora":
        return cora.simulate_and_estimate(cfg)
    if cfg.optimizer_backend == "riemann":
        return riemann.simulate_and_estimate(cfg)
    raise ValueError(
        f"unknown optimizer backend {cfg.optimizer_backend!r}; "
        "expected 'full', 'cora', or 'riemann'"
    )
