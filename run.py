#!/usr/bin/env python3
"""Run the range-aided pose estimation demo."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from scripts.config import load_config
from scripts.pipeline import run_demo


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config" / "default.yaml",
        help="YAML config file to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory from the config.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Generate text and PNG outputs only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_cfg, output_cfg = load_config(args.config)
    if args.output_dir is not None:
        output_cfg = replace(output_cfg, output_dir=args.output_dir)
    run_demo(sim_cfg, output_cfg, render_video_output=not args.no_video)


if __name__ == "__main__":
    main()
