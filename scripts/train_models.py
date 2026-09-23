"""Run train-only model selection and persist frozen pre-test artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from composites.training import run_training

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--split", type=Path, default=PROJECT_ROOT / "data" / "splits.json"
    )
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "modeling"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training(
        PROJECT_ROOT,
        data_dir=args.data_dir,
        split_path=args.split,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
    )
    for task, item in result["selection"]["tasks"].items():
        print(
            f"{task}: {item['selected_family']} / {item['selected_variant']} / "
            f"CV RMSE={item['selected_cv_rmse']:.6g}"
        )
    print(
        f"Train-only selection complete in {result['metadata']['elapsed_seconds']:.1f}s; "
        "test rows used=0"
    )


if __name__ == "__main__":
    main()
