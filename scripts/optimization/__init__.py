"""Optimizer backend selection."""

from __future__ import annotations

from scripts.configuration.config import SimConfig


def simulate_and_estimate(cfg: SimConfig) -> dict:
    """Run the configured optimizer backend."""
    if cfg.optimizer_backend == "full":
        from scripts.optimization import full

        return full.simulate_and_estimate(cfg)
    if cfg.optimizer_backend == "snapshot_sdp_diagnostic":
        from scripts.optimization import snapshot_sdp_diagnostic

        return snapshot_sdp_diagnostic.simulate_and_estimate(cfg)
    raise ValueError(
        f"unknown optimizer backend {cfg.optimizer_backend!r}; "
        "expected 'full' or 'snapshot_sdp_diagnostic'"
    )
