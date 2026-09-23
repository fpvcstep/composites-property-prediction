"""Generate EDA artifacts from the fixed training partition only."""

from __future__ import annotations

import argparse
from pathlib import Path

from composites.eda import run_training_eda

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--split", type=Path, default=PROJECT_ROOT / "data" / "splits.json"
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "eda"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=PROJECT_ROOT / "figures" / "eda"
    )
    parser.add_argument(
        "--documentation", type=Path, default=PROJECT_ROOT / "docs" / "eda.md"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training_eda(
        args.data_dir,
        args.split,
        args.reports_dir,
        args.figures_dir,
        args.documentation,
    )
    print(
        f"Training-only EDA written: train={result['train_rows']}, "
        f"test rows used={result['test_rows_used']}"
    )


if __name__ == "__main__":
    main()
