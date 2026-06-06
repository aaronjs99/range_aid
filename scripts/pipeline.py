"""End-to-end demo pipeline."""

from __future__ import annotations

from scripts.config import OutputConfig, SimConfig
from scripts.estimation import simulate_and_estimate
from scripts.figures import plot_position_timeseries, plot_summary
from scripts.reporting import write_summary
from scripts.video import render_video


def run_demo(
    sim_cfg: SimConfig, output_cfg: OutputConfig, render_video_output: bool = True
) -> None:
    output_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data = simulate_and_estimate(sim_cfg)
    summary_path = output_cfg.output_dir / output_cfg.summary_file
    summary_plot_path = output_cfg.output_dir / output_cfg.summary_plot_file
    position_plot_path = output_cfg.output_dir / output_cfg.position_plot_file
    video_path = output_cfg.output_dir / output_cfg.video_file

    write_summary(summary_path, data)
    plot_summary(summary_plot_path, data)
    plot_position_timeseries(position_plot_path, data)
    if render_video_output:
        render_video(video_path, data)

    print(summary_path.read_text().strip())
    print(f"wrote {summary_plot_path}")
    print(f"wrote {position_plot_path}")
    if render_video_output:
        print(f"wrote {video_path}")
