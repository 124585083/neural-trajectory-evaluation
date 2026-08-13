from __future__ import annotations

import argparse

from .common import DEFAULT_CONFIG, load_config
from .evaluate import evaluate_gpfa_models, evaluate_traditional_models
from .gpfa_pipeline import fit_gpfa_and_reliability
from .predictions import generate_predictions
from .protocol import prepare_protocol
from .sensitivity import run_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description="Static versus parameter-matched Dynamic evaluation")
    parser.add_argument(
        "command",
        choices=(
            "lock",
            "predict",
            "traditional",
            "gpfa",
            "sensitivity",
            "gpfa-evaluate",
            "extended-predict",
            "questions",
            "all",
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command in {"lock", "all"}:
        print(prepare_protocol(config))
    if args.command in {"predict", "all"}:
        print(generate_predictions(config))
    if args.command in {"traditional", "all"}:
        print(evaluate_traditional_models(config))
    if args.command in {"gpfa", "all"}:
        print(fit_gpfa_and_reliability(config))
    if args.command in {"sensitivity", "all"}:
        print(run_sensitivity(config))
    if args.command in {"gpfa-evaluate", "all"}:
        print(evaluate_gpfa_models(config))
    if args.command == "extended-predict":
        from .extended_predictions import generate_extended_predictions

        print(generate_extended_predictions(config))
    if args.command == "questions":
        from .question_analysis import run_question_analysis

        print(run_question_analysis(config))


if __name__ == "__main__":
    main()
