#!/usr/bin/env python3
"""Verify a range_aid archive or extract an immutable certification snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from range_aid.archive import read_archive_records, rebuild_full_batch, verify_archive
from range_aid.models import load_online_config


def _extract_snapshot(archive_path: Path, snapshot_id: str, output: Path) -> dict:
    matches = []
    with archive_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            record = json.loads(raw_line)
            if record.get("event_type") != "certification_snapshot_submitted":
                continue
            payload = dict(record.get("payload", {}) or {})
            snapshot = dict(payload.get("snapshot", {}) or {})
            if snapshot_id and str(snapshot.get("snapshot_id", "")) != snapshot_id:
                continue
            matches.append((line_number, snapshot))
    if not matches:
        raise ValueError("no matching certification snapshot in archive")
    if not snapshot_id and len(matches) != 1:
        raise ValueError("archive has multiple snapshots; provide --snapshot-id")
    line_number, snapshot = matches[-1]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "archive_path": str(archive_path),
        "archive_line": line_number,
        "snapshot_id": snapshot.get("snapshot_id", ""),
        "output_path": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("archive", type=Path)
    extract_parser = subparsers.add_parser("extract-snapshot")
    extract_parser.add_argument("archive", type=Path)
    extract_parser.add_argument("output", type=Path)
    extract_parser.add_argument("--snapshot-id", default="")
    rebuild_parser = subparsers.add_parser("rebuild-full-batch")
    rebuild_parser.add_argument("archive", type=Path)
    rebuild_parser.add_argument("config", type=Path)
    rebuild_parser.add_argument("output", type=Path)
    rebuild_parser.add_argument("--epoch", type=int)
    args = parser.parse_args()
    if args.command == "verify":
        report = verify_archive(args.archive)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] else 1
    verification = verify_archive(args.archive)
    if not verification["valid"]:
        raise ValueError(
            "archive verification failed: {}".format(verification["errors"])
        )
    if args.command == "rebuild-full-batch":
        report = rebuild_full_batch(
            read_archive_records(args.archive),
            load_online_config(args.config),
            epoch=args.epoch,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            json.dumps(
                {
                    "archive_path": str(args.archive),
                    "output_path": str(args.output),
                    "graph_epoch": report["graph_epoch"],
                    "pose_count": report["pose_count"],
                    "range_factor_count": report["range_factor_count"],
                    "loop_closure_count": report["loop_closure_count"],
                    "final_objective": report["final_objective"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = _extract_snapshot(args.archive, args.snapshot_id, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
