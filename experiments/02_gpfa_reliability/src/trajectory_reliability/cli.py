from __future__ import annotations

import argparse
import json

from .pipeline import inspect_data, load_config, run_pipeline
from .conditions import evaluate_behavior_conditioned_prior
from .saturation import run_saturation, run_split_count_saturation


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensorium GPFA trajectory reliability")
    parser.add_argument(
        "command",
        choices=("inspect", "smoke", "run", "saturation", "split-saturation", "condition-prior"),
    )
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    if args.command == "inspect":
        config, root = load_config(args.config)
        result = inspect_data(config, root)
    elif args.command == "saturation":
        result = run_saturation(args.config)
    elif args.command == "split-saturation":
        result = run_split_count_saturation(args.config)
    elif args.command == "condition-prior":
        result = evaluate_behavior_conditioned_prior(args.config)
    else:
        result = run_pipeline(args.config, smoke=args.command == "smoke")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
