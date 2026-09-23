"""Write the deterministic source-data audit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from composites.data import audit_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing X_bp.xlsx and X_nup.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "data_audit.json",
        help="Destination JSON report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_dataset(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    joined = report["inner_join"]
    print(
        f"Audit written to {args.output}: "
        f"{joined['rows']} rows x {joined['content_columns']} content columns"
    )


if __name__ == "__main__":
    main()
