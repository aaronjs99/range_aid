#!/usr/bin/env python3
"""Export one immutable range_aid snapshot to official CORA's PyFG format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from range_aid.certification.pyfg_export import export_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    report = export_snapshot(snapshot, args.destination)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
