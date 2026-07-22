#!/usr/bin/env python3
"""Run a pinned official CORA build and ingest its machine-readable result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from range_aid.certification.pyfg_export import (
    OFFICIAL_CORA_COMMIT,
    OFFICIAL_CORA_REPOSITORY,
)
from range_aid.certification.objective_parity import evaluate_objective_parity
from range_aid.archive import atomic_json_write
from range_aid.certification import validate_finite_tree, verify_pinned_repository


def _compile_adapter(source: Path, repository: Path, build: Path) -> Path:
    library = build / "lib" / "libCORA.so"
    ildl = build / "lib" / "libILDL.so"
    if not library.is_file() or not ildl.is_file():
        raise ValueError("official CORA build is missing libCORA or libILDL")
    destination = build / "bin" / "grande_official_cora_adapter"
    newest_input = max(
        source.stat().st_mtime,
        library.stat().st_mtime,
        ildl.stat().st_mtime,
        Path(__file__).stat().st_mtime,
    )
    if destination.is_file() and destination.stat().st_mtime >= newest_input:
        return destination
    include = repository / "libs"
    command = [
        "c++",
        "-std=gnu++17",
        "-O3",
        "-DNDEBUG",
        "-march=native",
        str(source),
        "-I{}".format(repository / "include"),
        "-I/usr/local/include/eigen3",
        "-I/usr/include/suitesparse",
        "-I{}".format(include / "Optimization" / "include"),
        "-I{}".format(include / "Preconditioners" / "SymILDL" / "SymILDL"),
        "-I{}".format(include / "Preconditioners"),
        "-I{}".format(
            include / "Preconditioners" / "Preconditioners" / "ILDL" / "include"
        ),
        str(library),
        str(ildl),
        "-lspqr",
        "-lcholmod",
        "-lblas",
        "-lf77blas",
        "-latlas",
        "-Wl,-rpath,{}".format(build / "lib"),
        "-o",
        str(destination),
    ]
    subprocess.run(command, check=True)
    return destination


def _load_inputs(pyfg: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(pyfg.read_bytes()).hexdigest()
    if digest != str(manifest.get("pyfg_sha256", "")):
        raise ValueError("PyFG content does not match its manifest")
    if str(manifest.get("required_official_cora_commit", "")) != OFFICIAL_CORA_COMMIT:
        raise ValueError("manifest does not require the pinned official CORA commit")
    return manifest


def _validate_result(result: dict, manifest: dict, seed: int) -> None:
    validate_finite_tree(result)
    if result.get("backend") != "official_cora" or not result.get("solved"):
        raise ValueError("official CORA did not return a solved result")
    for key in (
        "certified",
        "objective",
        "relaxation_rank",
        "returned_solution_rank",
        "poses",
        "landmarks",
    ):
        if key not in result:
            raise ValueError("official CORA result is missing {}".format(key))
    for key in ("objective", "objective_recomputed", "theta", "solve_time_sec"):
        if not math.isfinite(float(result[key])):
            raise ValueError("official CORA result contains non-finite {}".format(key))
    if abs(float(result["objective"]) - float(result["objective_recomputed"])) > 1e-8:
        raise ValueError("official CORA objective changed during result ingestion")
    if int(result.get("seed", -1)) != seed:
        raise ValueError("official CORA result seed changed")
    expected_poses = int(manifest["pose_count"])
    expected_pose_names = {"A{}".format(index) for index in range(expected_poses)}
    expected_pose_names.add("O0")
    if set(result["poses"]) != expected_pose_names:
        raise ValueError("official CORA result pose count changed")
    expected_landmark_names = {
        "L{}".format(index) for index in range(int(manifest["landmark_count"]))
    }
    if set(result["landmarks"]) != expected_landmark_names:
        raise ValueError("official CORA result landmark count changed")
    if int(result.get("range_factor_count", -1)) != int(manifest["range_factor_count"]):
        raise ValueError("official CORA result range factor count changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("pyfg", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cora-repo", type=Path, required=True)
    parser.add_argument("--cora-build", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--state-objective-absolute-tolerance", type=float, default=1e-3
    )
    parser.add_argument(
        "--state-objective-relative-tolerance", type=float, default=0.05
    )
    args = parser.parse_args()
    if args.seed < 0 or args.seed > 2**32 - 1:
        raise ValueError("seed must fit an unsigned 32-bit integer")
    if args.state_objective_absolute_tolerance < 0.0:
        raise ValueError("state objective absolute tolerance must be nonnegative")
    if args.state_objective_relative_tolerance < 0.0:
        raise ValueError("state objective relative tolerance must be nonnegative")

    verify_pinned_repository(
        args.cora_repo, OFFICIAL_CORA_REPOSITORY, OFFICIAL_CORA_COMMIT
    )
    manifest = _load_inputs(args.pyfg, args.manifest)
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if str(snapshot.get("snapshot_id", "")) != str(manifest["snapshot_id"]):
        raise ValueError("snapshot content does not match the PyFG manifest")
    package_path = Path(
        subprocess.check_output(["rospack", "find", "range_aid"], text=True).strip()
    )
    source = package_path / "tools" / "official_cora_adapter.cpp"
    if not source.is_file():
        raise ValueError("installed official CORA adapter source is missing")
    executable = _compile_adapter(source, args.cora_repo, args.cora_build)
    with tempfile.TemporaryDirectory(prefix="official-cora-result-") as directory:
        raw_result = Path(directory) / "result.json"
        completed = subprocess.run(
            [str(executable), str(args.pyfg), str(raw_result), str(args.seed)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            print(completed.stdout, file=sys.stderr, end="")
            print(completed.stderr, file=sys.stderr, end="")
            raise RuntimeError("official CORA executable failed")
        result = json.loads(raw_result.read_text(encoding="utf-8"))
    _validate_result(result, manifest, args.seed)
    objective_parity = evaluate_objective_parity(snapshot, result)
    state_delta = abs(float(objective_parity["gtsam_cora_state_objective_delta"]))
    state_scale = max(
        1.0,
        abs(float(objective_parity["official_cora_independent"])),
        abs(float(objective_parity["gtsam_state_same_objective"])),
    )
    state_tolerance = max(
        args.state_objective_absolute_tolerance,
        args.state_objective_relative_tolerance * state_scale,
    )
    state_agreement = {
        "absolute_delta": state_delta,
        "tolerance": state_tolerance,
        "absolute_tolerance": args.state_objective_absolute_tolerance,
        "relative_tolerance": args.state_objective_relative_tolerance,
        "passed": bool(state_delta <= state_tolerance),
    }
    formal_gate_passed = bool(
        result["certified"] and objective_parity["passed"] and state_agreement["passed"]
    )
    report = {
        **result,
        "snapshot_id": str(manifest["snapshot_id"]),
        "pyfg_sha256": str(manifest["pyfg_sha256"]),
        "official_cora_repository": OFFICIAL_CORA_REPOSITORY,
        "official_cora_commit": OFFICIAL_CORA_COMMIT,
        "objective_parity": objective_parity,
        "gtsam_cora_state_agreement": state_agreement,
        "official_certificate": bool(result["certified"]),
        "certificate_check": "post_solve_official_library_recheck",
        "formal_gate_passed": formal_gate_passed,
        "certificate_scope": "exact_exported_nonrobust_pose_range_instance",
        "sensor_correctness_certified": False,
    }
    if not objective_parity["passed"]:
        report["rejection_reason"] = "official_cora_objective_parity_failed"
    elif not state_agreement["passed"]:
        report["rejection_reason"] = "certified_solution_disagrees_with_gtsam"
    atomic_json_write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
