"""Text result reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.configuration.config import SimConfig
from scripts.math.usbl import predict_usbl_observations, wrap_angle
from scripts.viz.plot_utils import sigma_radius_from_cov


def write_summary(path: Path, data) -> None:
    cfg: SimConfig = data["cfg"]
    a_true = data.get("a_true_pose", data["a_pose"])
    a_est = data["a_pose"]
    b_true = data["b_true_pose"][:, :3]
    b_init = data["b_init_xyz"]
    b_est = data["b_est_pose"][:, :3]
    b_available = data.get("b_est_available_mask", np.all(np.isfinite(b_est), axis=1))
    b_available = b_available & np.all(np.isfinite(b_est), axis=1)
    cora_feed_mask = data.get("cora_feed_mask", np.ones(len(b_true), dtype=bool))
    stationary_ping_mask = data.get(
        "stationary_ping_mask", np.ones(len(b_true), dtype=bool)
    )
    clean_ranges = data["clean_ranges"]
    init_rmse = float(np.sqrt(np.mean(np.sum((b_init - b_true) ** 2, axis=1))))
    b_error = np.linalg.norm(b_est - b_true, axis=1)
    if np.any(b_available):
        first_available_idx = int(np.flatnonzero(b_available)[0])
        est_rmse = float(
            np.sqrt(
                np.nanmean(
                    np.sum((b_est[b_available] - b_true[b_available]) ** 2, axis=1)
                )
            )
        )
        b_error_mean = float(np.nanmean(b_error[b_available]))
        b_error_std = float(np.nanstd(b_error[b_available]))
        b_error_max = float(np.nanmax(b_error[b_available]))
    else:
        first_available_idx = -1
        est_rmse = b_error_mean = b_error_std = b_error_max = float("nan")
    active_error_mask = b_available & stationary_ping_mask
    if np.any(active_error_mask):
        b_error_active_mean = float(np.nanmean(b_error[active_error_mask]))
        b_error_active_std = float(np.nanstd(b_error[active_error_mask]))
        b_error_active_max = float(np.nanmax(b_error[active_error_mask]))
    else:
        b_error_active_mean = b_error_active_std = b_error_active_max = float("nan")
    a_position_rmse = float(
        np.sqrt(np.mean(np.sum((a_est[:, :3] - a_true[:, :3]) ** 2, axis=1)))
    )
    a_orientation_rmse = float(
        np.sqrt(
            np.mean(np.sum(wrap_angle(a_est[:, 3:6] - a_true[:, 3:6]) ** 2, axis=1))
        )
    )
    est_ranges, est_azimuths, est_elevations = predict_usbl_observations(
        data["a_pose"], b_est, cfg
    )
    consistency_mask = b_available & cora_feed_mask
    if not np.any(consistency_mask):
        consistency_mask = b_available
    range_rmse = float(
        np.sqrt(
            np.nanmean(
                (est_ranges[consistency_mask] - clean_ranges[consistency_mask]) ** 2
            )
        )
        if np.any(consistency_mask)
        else float("nan")
    )
    azimuth_rmse = float(
        np.sqrt(
            np.nanmean(
                wrap_angle(
                    est_azimuths[consistency_mask]
                    - data["true_azimuths"][consistency_mask]
                )
                ** 2
            )
        )
        if np.any(consistency_mask)
        else float("nan")
    )
    elevation_rmse = float(
        np.sqrt(
            np.nanmean(
                (
                    est_elevations[consistency_mask]
                    - data["true_elevations"][consistency_mask]
                )
                ** 2
            )
        )
        if np.any(consistency_mask)
        else float("nan")
    )
    depth_rmse = (
        float(
            np.sqrt(
                np.nanmean(
                    (b_est[consistency_mask, 2] - data["true_depths"][consistency_mask])
                    ** 2
                )
            )
        )
        if np.any(consistency_mask)
        else float("nan")
    )
    if cfg.optimizer_backend == "snapshot_sdp_diagnostic":
        smoothness_rms = float("nan")
    else:
        smoothness_rms = float(
            np.sqrt(
                np.nanmean(
                    np.sum(
                        ((b_est[2:] - 2.0 * b_est[1:-1] + b_est[:-2]) / (cfg.dt**2))
                        ** 2,
                        axis=1,
                    )
                )
            )
        )
    mean_2sigma = float(
        np.nanmean([sigma_radius_from_cov(c) for c in data["b_cov_xyz"]])
    )
    first_ping_start = cfg.stationary_ping_windows[0][0]
    last_ping_end = cfg.stationary_ping_windows[-1][1]
    a_error = np.linalg.norm(a_est[:, :3] - a_true[:, :3], axis=1)
    before_ping_mask = np.arange(len(a_error)) < first_ping_start
    after_ping_mask = np.arange(len(a_error)) >= last_ping_end
    a_error_before = (
        float(np.mean(a_error[before_ping_mask]))
        if np.any(before_ping_mask)
        else float("nan")
    )
    a_error_during = (
        float(np.mean(a_error[stationary_ping_mask]))
        if np.any(stationary_ping_mask)
        else float("nan")
    )
    a_error_after = (
        float(np.mean(a_error[after_ping_mask]))
        if np.any(after_ping_mask)
        else float("nan")
    )
    cora_rank_ratios = np.asarray(data.get("cora_rank_ratios", []), dtype=float)
    cora_slack_sums = np.asarray(data.get("cora_slack_sums", []), dtype=float)
    cora_sdp_valid = np.asarray(data.get("cora_sdp_valid", []), dtype=bool)
    cora_rank_tight = np.asarray(data.get("cora_rank_tight", []), dtype=bool)
    cora_certified_tight = np.asarray(data.get("cora_certified_tight", []), dtype=bool)
    cora_sdp_objectives = np.asarray(
        data.get("cora_sdp_objectives", data.get("cora_objectives", [])), dtype=float
    )
    cora_sdp_primal_objectives = np.asarray(
        data.get("cora_sdp_primal_objectives", data.get("cora_primal_objectives", [])),
        dtype=float,
    )
    cora_sdp_gaps = np.asarray(
        data.get("cora_sdp_gaps", data.get("cora_certificate_gaps", [])), dtype=float
    )
    cora_sdp_relative_gaps = np.asarray(
        data.get(
            "cora_sdp_relative_gaps",
            data.get("cora_relative_certificate_gaps", []),
        ),
        dtype=float,
    )
    cora_refined_local_objectives = np.asarray(
        data.get("cora_refined_local_objectives", []), dtype=float
    )
    cora_window_count = int(data.get("cora_window_count", 0))
    cora_sdp_valid_count = int(
        data.get("cora_sdp_valid_count", int(np.sum(cora_sdp_valid)))
    )
    cora_rank_tight_count = int(
        data.get("cora_rank_tight_count", int(np.sum(cora_rank_tight)))
    )
    cora_certified_tight_count = int(
        data.get("cora_certified_tight_count", int(np.sum(cora_certified_tight)))
    )
    cora_sdp_all_valid = bool(data.get("cora_sdp_all_valid", False))
    cora_sdp_diagnostic_success = bool(
        data.get("cora_sdp_diagnostic_success", cora_sdp_all_valid)
    )
    cora_local_refinement_success = bool(
        data.get("cora_local_refinement_success", False)
    )
    cora_pipeline_success = bool(data.get("cora_pipeline_success", False))
    cora_published_estimate_source = str(
        data.get("cora_published_estimate_source", "none")
    )
    cora_published_cost = float(data.get("cora_published_cost", float("nan")))
    cora_sdp_objective_sum = float(data.get("cora_sdp_objective_sum", float("nan")))
    cora_refined_local_objective_sum = float(
        data.get("cora_refined_local_objective_sum", float("nan"))
    )
    cora_invalid_reasons_by_event = data.get("cora_invalid_reasons_by_event", {})
    cora_statuses = data.get("cora_statuses", [])
    event_landmarks = data.get("event_landmark_estimates", [])
    num_events = int(data.get("num_events", len(event_landmarks)))
    recovery_methods = [
        result.recovery_method for result in data.get("cora_window_results", [])
    ]
    event_lines = []
    for event in event_landmarks:
        l_sdp = event.get("l_hat_sdp", event["l_hat"])
        l_refined = event.get("l_hat_refined", np.full(3, np.nan))
        l_final = event.get("l_hat_final", event["l_hat"])
        refinement_used = bool(
            event.get("refinement_used", event.get("refined", False))
        )
        final_source = (
            "refined USBL range+bearing" if refinement_used else "SDP recovery"
        )
        refined_objective = float(event.get("refined_local_objective", float("nan")))
        invalid_reasons = tuple(event.get("invalid_reasons", ()))
        invalid_reason_text = ", ".join(invalid_reasons) if invalid_reasons else "none"
        event_truth = b_true[int(event["start"])]
        l_sdp_error = (
            float(np.linalg.norm(l_sdp - event_truth))
            if np.all(np.isfinite(l_sdp))
            else float("nan")
        )
        l_final_error = (
            float(np.linalg.norm(l_final - event_truth))
            if np.all(np.isfinite(l_final))
            else float("nan")
        )
        event_lines.append(
            "- event {event_index}: [{start}, {end})\n"
            "  SDP diagnostic: L_sdp=({sx:.4f}, {sy:.4f}, {sz:.4f}) "
            "sdp_status={status} sdp_valid={sdp_valid} "
            "rank_tight={rank_tight} certified_tight={certified_tight} "
            "rank_ratio={rank_ratio:.6g} L_sdp_error={l_sdp_error:.4f}m "
            "gap={gap:.6g} rel_gap={rel_gap:.6g} recovery={recovery}\n"
            "  Feasibility: z00_error={z00_error:.6g} min_eig_Z={min_eig_z:.6g} "
            "psd_violation={psd_violation:.6g} "
            "max_range_constraint_residual={max_range_residual:.6g} "
            "min_slack_plus={min_slack_plus:.6g} "
            "min_slack_minus={min_slack_minus:.6g} min_slack={min_slack:.6g} "
            "objective_nonnegative={objective_nonnegative} "
            "invalid_reasons={invalid_reasons}\n"
            "  Final published event estimate: L_final=({fx:.4f}, {fy:.4f}, {fz:.4f}) "
            "source={final_source} L_final_error={l_final_error:.4f}m "
            "refined_objective={refined_objective:.6g} "
            "L_refined=({rx:.4f}, {ry:.4f}, {rz:.4f})".format(
                event_index=event["event_index"],
                start=event["start"],
                end=event["end"],
                sx=float(l_sdp[0]),
                sy=float(l_sdp[1]),
                sz=float(l_sdp[2]),
                fx=float(l_final[0]),
                fy=float(l_final[1]),
                fz=float(l_final[2]),
                rx=float(l_refined[0]),
                ry=float(l_refined[1]),
                rz=float(l_refined[2]),
                status=event["status"],
                sdp_valid=event.get("sdp_valid", False),
                rank_tight=event.get("rank_tight", False),
                certified_tight=event.get("certified_tight", False),
                rank_ratio=float(event["rank_ratio"]),
                l_sdp_error=l_sdp_error,
                gap=float(
                    event.get("sdp_gap", event.get("certificate_gap", float("nan")))
                ),
                rel_gap=float(
                    event.get(
                        "sdp_relative_gap",
                        event.get("relative_certificate_gap", float("nan")),
                    )
                ),
                recovery=event["recovery_method"],
                z00_error=float(event.get("z00_error", float("nan"))),
                min_eig_z=float(event.get("min_eig_z", float("nan"))),
                psd_violation=float(event.get("psd_violation", float("nan"))),
                max_range_residual=float(
                    event.get("max_range_constraint_residual", float("nan"))
                ),
                min_slack_plus=float(event.get("min_slack_plus", float("nan"))),
                min_slack_minus=float(event.get("min_slack_minus", float("nan"))),
                min_slack=float(event.get("min_slack", float("nan"))),
                objective_nonnegative=event.get("sdp_objective_nonnegative", False),
                invalid_reasons=invalid_reason_text,
                final_source=final_source,
                l_final_error=l_final_error,
                refined_objective=refined_objective,
            )
        )
    cora_lines = []
    if cfg.optimizer_backend == "snapshot_sdp_diagnostic":
        setup_lines = [
            "- Active model: event-triggered CORA-style stationary-ping SDP.",
            "- The scalar range factor is used in the SDP relaxation.",
            "- USBL angle/depth factors are not part of the SDP relaxation.",
            "- Optional CORA local refinement can use enabled nonlinear factors.",
            "- CORA uses no artificial target smoothness.",
        ]
    else:
        setup_lines = [
            "- Active model: moving target with smoothness and boat-mounted USBL factors.",
            "- The scalar range factor is always present.",
            "- USBL azimuth/elevation factors are enabled when configured.",
        ]
    target_line = (
        "- Object B moves before/after the ping and is assumed stationary only inside CORA ping windows."
        if cfg.optimizer_backend == "snapshot_sdp_diagnostic"
        else "- Object B is a moving underwater target carrying an acoustic transponder."
    )
    estimate_line = (
        "- CORA estimates one stationary event landmark per solved ping and archives it after the event."
        if cfg.optimizer_backend == "snapshot_sdp_diagnostic"
        else "- The optimizer estimates B_i position per timestamp with the active factor set."
    )
    if cfg.optimizer_backend == "snapshot_sdp_diagnostic":
        finite_rank_ratios = cora_rank_ratios[np.isfinite(cora_rank_ratios)]
        finite_slacks = cora_slack_sums[np.isfinite(cora_slack_sums)]
        finite_objectives = cora_sdp_objectives[np.isfinite(cora_sdp_objectives)]
        finite_primal_objectives = cora_sdp_primal_objectives[
            np.isfinite(cora_sdp_primal_objectives)
        ]
        finite_gaps = cora_sdp_gaps[np.isfinite(cora_sdp_gaps)]
        finite_relative_gaps = cora_sdp_relative_gaps[
            np.isfinite(cora_sdp_relative_gaps)
        ]
        finite_refined_local_objectives = cora_refined_local_objectives[
            np.isfinite(cora_refined_local_objectives)
        ]
        invalid_reason_lines = [
            f"- event {event_index}: {', '.join(reasons)}"
            for event_index, reasons in sorted(cora_invalid_reasons_by_event.items())
        ]
        cora_lines = [
            "CORA diagnostics:",
            "- backend formulation: stationary-window range-aided QCQP lifted to dense SDP",
            "- CORA uses no artificial target smoothness: True",
            "- certificate status: SDP diagnostic only; local refinement is reported separately",
            f"- cora_anchor_mode: {cfg.cora_anchor_mode}",
            f"- physical_bounds_enabled: True",
            f"- cora_second_moment_bounds: {cfg.cora_second_moment_bounds}",
            f"- cora_bound_xy_m: {cfg.cora_bound_xy_m}",
            f"- cora_bound_z_boat_m: {cfg.cora_bound_z_boat_m}",
            f"- cora_bound_z_target_m: ({cfg.cora_bound_z_target_min_m}, {cfg.cora_bound_z_target_max_m})",
            f"- cora_window_size: {cfg.cora_window_size}",
            f"- cora_solve_stride: {cfg.cora_solve_stride}",
            f"- cora_solver: {cfg.cora_solver}",
            f"- cora_refine_with_full: {cfg.cora_refine_with_full}",
            f"- stationary_ping_windows: {cfg.stationary_ping_windows}",
            f"- num_events: {num_events}",
            f"- USBL samples fed to dense snapshot SDP: {int(np.sum(cora_feed_mask))}",
            f"- USBL inactive outside ping/update: {int(len(cora_feed_mask) - np.sum(cora_feed_mask))}",
            f"- USBL out-of-range samples dropped: {int(np.sum(data.get('usbl_out_of_range_mask', [])))}",
            f"- USBL out-of-coverage samples dropped: {int(np.sum(data.get('usbl_out_of_coverage_mask', [])))}",
            f"- SDP statuses: {sorted(set(cora_statuses))}",
            f"- SDP event graphs solved: {cora_window_count}",
            f"- SDP valid event graphs: {cora_sdp_valid_count}/{cora_window_count}",
            f"- rank-tight event graphs: {cora_rank_tight_count}/{cora_window_count}",
            f"- certified-tight event graphs: {cora_certified_tight_count}/{cora_window_count}",
            f"- sdp_all_valid: {cora_sdp_all_valid}",
            f"- sdp_diagnostic_success: {cora_sdp_diagnostic_success}",
            f"- local_refinement_success: {cora_local_refinement_success}",
            f"- pipeline_success: {cora_pipeline_success}",
            f"- published_estimate_source: {cora_published_estimate_source}",
            f"- published_cost: {cora_published_cost:.6g}",
            f"- sdp_objective_sum: {cora_sdp_objective_sum:.6g}",
            f"- refined_local_objective_sum: {cora_refined_local_objective_sum:.6g}",
            f"- SDP recovery methods: {sorted(set(recovery_methods))}",
            f"- SDP rank ratio mean: {float(np.mean(finite_rank_ratios)) if len(finite_rank_ratios) else float('nan'):.6g}",
            f"- SDP rank ratio max: {float(np.max(finite_rank_ratios)) if len(finite_rank_ratios) else float('nan'):.6g}",
            f"- SDP slack sum total: {float(np.sum(finite_slacks)) if len(finite_slacks) else float('nan'):.6g}",
            f"- SDP lower-bound objective sum: {float(np.sum(finite_objectives)) if len(finite_objectives) else float('nan'):.6g}",
            f"- SDP recovered primal objective sum: {float(np.sum(finite_primal_objectives)) if len(finite_primal_objectives) else float('nan'):.6g}",
            f"- SDP recovered primal minus SDP gap sum: {float(np.sum(finite_gaps)) if len(finite_gaps) else float('nan'):.6g}",
            f"- SDP recovered primal relative gap max: {float(np.max(finite_relative_gaps)) if len(finite_relative_gaps) else float('nan'):.6g}",
            f"- refined local USBL objective sum: {float(np.sum(finite_refined_local_objectives)) if len(finite_refined_local_objectives) else float('nan'):.6g}",
            "- final published estimates use refined USBL range+bearing when refinement is enabled.",
            "- SDP rank/gap diagnostics apply to L_sdp, not to L_final when refinement is used.",
            *(
                [
                    "- SDP diagnostic invalid; published estimate comes from local USBL refinement."
                ]
                if not cora_sdp_diagnostic_success
                and cora_published_estimate_source == "local_refinement"
                else []
            ),
            "Invalid SDP reasons:",
            *(invalid_reason_lines or ["- none"]),
            "- previous event factors persist into new event graphs: False",
            "Archived event landmarks:",
            *(event_lines or ["- none"]),
            "- certificate note: certified_tight requires both SDP validity and rank tightness; this is still not a full Riemannian Staircase certificate.",
        ]

    if cfg.optimizer_backend == "snapshot_sdp_diagnostic":
        optimizer_lines = [
            f"pipeline_success: {cora_pipeline_success}",
            f"sdp_diagnostic_success: {cora_sdp_diagnostic_success}",
            f"local_refinement_success: {cora_local_refinement_success}",
            f"published_estimate_source: {cora_published_estimate_source}",
            f"published_cost: {cora_published_cost:.6g}",
            f"sdp_objective_sum: {cora_sdp_objective_sum:.6g}",
            f"refined_local_objective_sum: {cora_refined_local_objective_sum:.6g}",
            f"optimizer iterations: {data['optimizer'].nfev}",
        ]
    else:
        optimizer_lines = [
            f"optimizer success: {data['optimizer'].success}",
            f"optimizer cost: {data['optimizer'].cost:.4f}",
            f"optimizer iterations: {data['optimizer'].nfev}",
        ]

    path.write_text(
        "\n".join(
            [
                "3D moving range-aided pose estimation demo",
                "",
                "Setup:",
                *setup_lines,
                "- Object A pose is estimated from a DLiO-like VLP-16 + Xsens stack.",
                "- Simulated acoustic measurements are generated from true A and true B.",
                "- The optimizer receives the noisy DLiO-like A estimate as its boat pose input.",
                target_line,
                "- Moving-time measurements are logged but excluded from CORA event graphs.",
                "- The USBL transceiver is fixed to the boat with a configured extrinsic transform.",
                "- Measurements are generated with T_AS_true and optimized with T_AS_est.",
                "- Measurements are predicted in the USBL sensor frame, not the map frame.",
                "- The range residual at time i is ||q_i|| - measured_range_i.",
                "- q_i = R_S_i.T @ (B_i - p_S_i), where S_i = A_i * T_A_S.",
                f"- smoothness factor enabled: {cfg.use_smoothness_factor}",
                "- smoothness ignored by dense snapshot SDP: "
                f"{cfg.optimizer_backend == 'snapshot_sdp_diagnostic'}",
                f"- depth factor enabled: {cfg.use_depth_factor}",
                f"- USBL angle factors enabled: {cfg.use_usbl_angles}",
                f"- optimizer backend: {cfg.optimizer_backend}",
                estimate_line,
                "- Range alone does not estimate attitude; outputs focus on position.",
                "",
                f"steps: {cfg.steps}",
                f"boat_usbl_name: {cfg.boat_usbl_name}",
                f"boat_usbl_role: {cfg.boat_usbl_role}",
                f"target_transponder_name: {cfg.target_transponder_name}",
                f"target_transponder_role: {cfg.target_transponder_role}",
                f"boat_usbl_frequency_band_khz: {cfg.boat_usbl_frequency_band_khz}",
                f"target_transponder_frequency_band_khz: {cfg.target_transponder_frequency_band_khz}",
                f"max_range_m: {cfg.max_range_m}",
                f"boat_usbl_depth_rating_m: {cfg.boat_usbl_depth_rating_m}",
                f"target_transponder_depth_rating_m: {cfg.target_transponder_depth_rating_m}",
                f"acoustic_coverage_deg: {cfg.acoustic_coverage_deg}",
                f"target_transponder_beam_shape_deg: {cfg.target_transponder_beam_shape_deg}",
                f"range_sigma_m: {cfg.range_sigma_m}",
                "- range_sigma_m is raw range precision for the factor, not final system position accuracy.",
                "- Micro-Ranger 2 system-level positioning can be on the order of percent-of-slant-range; this is a factor-level simulation.",
                f"angular_accuracy_deg: {np.rad2deg(cfg.angular_accuracy_rad):.2f}",
                f"update_rate_hz: {cfg.boat_usbl_update_rate_hz}",
                f"boat_usbl_weight_air_kg: {cfg.boat_usbl_weight_air_kg}",
                f"boat_usbl_weight_water_kg: {cfg.boat_usbl_weight_water_kg}",
                f"target_transponder_weight_air_kg: {cfg.target_transponder_weight_air_kg}",
                f"target_transponder_weight_water_kg: {cfg.target_transponder_weight_water_kg}",
                f"boat_usbl_typical_power_w: {cfg.boat_usbl_typical_power_w}",
                f"boat_usbl_max_power_w: {cfg.boat_usbl_max_power_w}",
                f"boat_usbl_mount_offset_m: {cfg.boat_usbl_mount_offset_m}",
                f"boat_usbl_mount_rpy_deg: {tuple(np.rad2deg(cfg.boat_usbl_mount_rpy_rad))}",
                f"usbl_mount_bias_m: {cfg.usbl_mount_bias_m}",
                f"usbl_mount_bias_rpy_deg: {tuple(np.rad2deg(cfg.usbl_mount_bias_rpy_rad))}",
                f"usbl_mount_prior_sigma_m: {cfg.usbl_mount_prior_sigma_m}",
                f"usbl_mount_prior_sigma_deg: {tuple(np.rad2deg(cfg.usbl_mount_prior_sigma_rad))}",
                f"smoothness_accel_sigma_mps2: {cfg.smoothness_accel_sigma_mps2}",
                f"depth_sigma_m: {cfg.depth_sigma_m}",
                f"initial_B_first_position_m: {tuple(data['b_init_xyz'][0])}",
                f"estimated_B_first_available_index: {first_available_idx}",
                f"estimated_B_first_available_position_m: {tuple(data['b_est_pose'][first_available_idx, :3]) if first_available_idx >= 0 else 'none'}",
                f"max_clean_range_m: {float(np.max(clean_ranges)):.4f}",
                f"max_abs_B_depth_m: {float(np.max(np.abs(b_true[:, 2]))):.4f}",
                f"A position sigma xyz m: {cfg.a_position_sigma_m}",
                f"A orientation sigma rpy deg: {tuple(np.rad2deg(cfg.a_orientation_sigma_rad))}",
                f"A estimated position RMSE: {a_position_rmse:.4f} m",
                f"A estimated orientation RMSE deg: {np.rad2deg(a_orientation_rmse):.4f}",
                f"A position error mean before ping: {a_error_before:.4f} m",
                f"A position error mean during ping: {a_error_during:.4f} m",
                f"A position error mean after ping: {a_error_after:.4f} m",
                f"initial trajectory RMSE: {init_rmse:.4f} m",
                f"event-available estimate RMSE: {est_rmse:.4f} m",
                f"B estimation error max: {b_error_max:.4f} m",
                f"B estimation error mean: {b_error_mean:.4f} m",
                f"B estimation error std dev: {b_error_std:.4f} m",
                f"active_ping_B_error_max: {b_error_active_max:.4f} m",
                f"active_ping_B_error_mean: {b_error_active_mean:.4f} m",
                f"active_ping_B_error_std_dev: {b_error_active_std:.4f} m",
                f"range consistency RMSE: {range_rmse:.4f} m",
                f"azimuth consistency RMSE deg: {np.rad2deg(azimuth_rmse):.4f}",
                f"elevation consistency RMSE deg: {np.rad2deg(elevation_rmse):.4f}",
                f"depth consistency RMSE: {depth_rmse:.4f} m",
                f"smoothness acceleration RMS: {smoothness_rms:.4f} m/s^2",
                f"mean B 2-sigma position sphere radius: {mean_2sigma:.4f} m",
                *optimizer_lines,
                f"optimizer backend note: {data.get('optimizer_backend_note', 'none')}",
                *cora_lines,
                "",
            ]
        )
    )
