"""Persist the fixed holdout and 10-fold CV index manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from composites.split import create_split_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "splits.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_split_manifest(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts = manifest["counts"]
    print(
        f"Split written to {args.output}: "
        f"train={counts['train']}, test={counts['test']}"
    )


if __name__ == "__main__":
    main()
