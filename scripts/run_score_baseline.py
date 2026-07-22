#!/usr/bin/env python3
"""Run the pinned official SCORE SOCP baseline on an exported PyFG snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from range_aid.archive import atomic_json_write
from range_aid.certification import verify_pinned_repository

OFFICIAL_SCORE_REPOSITORY = "https://github.com/MarineRoboticsGroup/score.git"
OFFICIAL_SCORE_COMMIT = "41626b49702d27a8fca03982533ff52f6306278d"
OFFICIAL_PYFACTORGRAPH_REPOSITORY = (
    "https://github.com/MarineRoboticsGroup/PyFactorGraph.git"
)
OFFICIAL_PYFACTORGRAPH_COMMIT = "87e18e9bab56b08dfe95e998c801226acba2439b"


def _array_rows(values):
    return {
        key: np.asarray(value, dtype=float).tolist() for key, value in values.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pyfg", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--score-repo", type=Path, required=True)
    parser.add_argument("--pyfactorgraph-repo", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify_pinned_repository(
        args.score_repo, OFFICIAL_SCORE_REPOSITORY, OFFICIAL_SCORE_COMMIT
    )
    verify_pinned_repository(
        args.pyfactorgraph_repo,
        OFFICIAL_PYFACTORGRAPH_REPOSITORY,
        OFFICIAL_PYFACTORGRAPH_COMMIT,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pyfg_sha256 = hashlib.sha256(args.pyfg.read_bytes()).hexdigest()
    if pyfg_sha256 != str(manifest.get("pyfg_sha256", "")):
        raise ValueError("PyFG content does not match its manifest")
    sys.path.insert(0, str(args.pyfactorgraph_repo))
    sys.path.insert(0, str(args.score_repo))
    report = {
        "schema_version": 1,
        "backend": "official_score_socp",
        "snapshot_id": str(manifest.get("snapshot_id", "")),
        "pyfg_sha256": pyfg_sha256,
        "score_repository": OFFICIAL_SCORE_REPOSITORY,
        "score_commit": OFFICIAL_SCORE_COMMIT,
        "pyfactorgraph_repository": OFFICIAL_PYFACTORGRAPH_REPOSITORY,
        "pyfactorgraph_commit": OFFICIAL_PYFACTORGRAPH_COMMIT,
        "initialization_only": True,
        "formal_certificate": False,
    }
    try:
        from py_factor_graph.io.pyfg_text import read_from_pyfg_text
        from score.solve_score import solve_score
        from score.utils.gurobi_utils import SOCP_RELAXATION
    except ImportError as exc:
        report.update(
            {
                "available": False,
                "solved": False,
                "reason": "missing_external_dependency:{}".format(exc.name),
            }
        )
        atomic_json_write(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if args.check_only else 2
    report["available"] = True
    if args.check_only:
        report.update({"solved": False, "reason": "check_only"})
    else:
        graph = read_from_pyfg_text(str(args.pyfg))
        result = solve_score(graph, relaxation_type=SOCP_RELAXATION)
        report.update(
            {
                "solved": bool(result.solved),
                "solver_cost": (
                    float(result.solver_cost)
                    if result.solver_cost is not None
                    else None
                ),
                "solve_time_sec": float(result.total_time),
                "poses": _array_rows(result.poses),
                "landmarks": _array_rows(result.landmarks),
            }
        )
    atomic_json_write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
