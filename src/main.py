"""Entry point for reproducing the experiments in the paper.

Each experiment maps to a result table / figure:

  cp_regression              -> Section 4.1.1, Table 1, Figure 2  (class-agnostic regression)
  cp_classwise_regression    -> Section 4.1.2, Table 2            (class-wise regression, oracle class)
  cp_two_step                -> Section 4.1.3, Table 4            (two-step RAPS + regression)
  cp_classification          -> Section 4.1.3, Table 3 + Figure 3 (APS / RAPS prediction sets)

Run with, e.g.::

    python -m src.main --config configs/kitti.yaml --experiment cp_regression
    python -m src.main --config configs/bdd.yaml   --experiment cp_regression cp_classwise_regression
"""
import argparse
from pathlib import Path

import yaml

from src.experiments import (
    cp_classification,
    cp_classwise_regression,
    cp_regression,
    cp_two_step,
)

EXPERIMENTS = {
    "cp_regression": cp_regression.run,
    "cp_classwise_regression": cp_classwise_regression.run,
    "cp_two_step": cp_two_step.run,
    "cp_classification": cp_classification.run,
}


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a YAML config file (see configs/).")
    parser.add_argument(
        "--experiment",
        nargs="+",
        choices=sorted(EXPERIMENTS.keys()),
        default=["cp_regression"],
        help="One or more experiments to run (in order).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print(f"Dataset: {config['dataset']}")

    for name in args.experiment:
        print(f"\nRunning {name}")
        EXPERIMENTS[name](config)

    print("\nExperiments successfully completed.")


if __name__ == "__main__":
    main()
