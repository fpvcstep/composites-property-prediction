"""Evaluate all frozen models on full train/test and create final artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from composites.evaluation import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--split", type=Path, default=PROJECT_ROOT / "data" / "splits.json"
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "reports" / "modeling" / "model_selection.json",
    )
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "evaluation"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=PROJECT_ROOT / "figures" / "evaluation"
    )
    parser.add_argument(
        "--results-doc", type=Path, default=PROJECT_ROOT / "docs" / "results.md"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_evaluation(
        PROJECT_ROOT,
        data_dir=args.data_dir,
        split_path=args.split,
        selection_path=args.selection,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        results_doc=args.results_doc,
    )
    for row in result["metrics"]:
        print(
            f"{row['task']} {row['partition']}: RMSE={row['rmse']:.6g}, "
            f"MAE={row['mae']:.6g}, R2={row['r2']:.6g}"
        )
    print(f"Evaluation complete in {result['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
