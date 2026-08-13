from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .audit import run_parameter_audit
from .config import DEFAULT_CONFIG, load_config, resolve_project_path
from .data import DynamicSensoriumDataModule
from .evaluation.response import correlation_summary
from .export import export_aligned_trials
from .hooks import RepresentationHooks
from .models import build_model
from .training import benchmark_reference, load_checkpoint_model, train_reference


MODELS = ("dynamic_reference", "static_reference")


def _evaluate(config: dict) -> dict:
    data = DynamicSensoriumDataModule(config)
    device = torch.device(config["project"]["device"] if torch.cuda.is_available() else "cpu")
    bundle = data.make_loader("oracle", frames=int(config["data"]["frames"]), batch_size=1, shuffle=False)
    model_results = {}
    for name in MODELS:
        model, checkpoint = load_checkpoint_model(name, config, data, device)
        manifest, targets, predictions, stimulus_ids = export_aligned_trials(
            model,
            bundle,
            data,
            config,
            checkpoint_path=str(resolve_project_path(config, Path("checkpoints") / name / "final.pt")),
            device=device,
        )
        metrics = correlation_summary(targets, predictions, stimulus_ids)
        model_results[name] = {
            "single_trial_mean": metrics["single_trial_mean"],
            "trial_average_mean": metrics["trial_average_mean"],
            "repeat_group_count": metrics["repeat_group_count"],
            "manifest": str(resolve_project_path(config, Path("predictions") / name / "oracle" / "manifest.json")),
            "checkpoint_validation_correlation": checkpoint["validation_correlation"],
        }
    result = {
        "session": data.session_key,
        "models": model_results,
        "dynamic_gain_single_trial": model_results["dynamic_reference"]["single_trial_mean"]
        - model_results["static_reference"]["single_trial_mean"],
        "dynamic_gain_trial_average": model_results["dynamic_reference"]["trial_average_mean"]
        - model_results["static_reference"]["trial_average_mean"],
    }
    output = resolve_project_path(config, Path("artifacts") / "evaluation_metrics.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _hook_smoke(config: dict) -> dict:
    data = DynamicSensoriumDataModule(config)
    batch = next(iter(data.make_loader("oracle", batch_size=1, shuffle=False).loader))
    result = {}
    for name in MODELS:
        model, _ = build_model(name, config, data, "cpu")
        model.eval()
        with RepresentationHooks(model) as hooks, torch.no_grad():
            output = model(batch["video"], batch["behavior"], batch["pupil_center"])
            result[name] = {"output_shape": list(output.shape), "representations": hooks.metadata(batch["video"].shape[2])}
    path = resolve_project_path(config, Path("artifacts") / "representation_hooks.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensorium Phase-1 official baseline workflow")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--model", choices=MODELS + ("both",), default="both")
    train = subparsers.add_parser("train")
    train.add_argument("--model", choices=MODELS + ("both",), default="both")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("hooks")
    subparsers.add_parser("all")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "audit":
        result = run_parameter_audit(config)
    elif args.command == "benchmark":
        names = MODELS if args.model == "both" else (args.model,)
        data = DynamicSensoriumDataModule(config)
        result = {name: benchmark_reference(name, config, data) for name in names}
    elif args.command == "train":
        names = MODELS if args.model == "both" else (args.model,)
        result = {name: train_reference(name, config) for name in names}
    elif args.command == "evaluate":
        result = _evaluate(config)
    elif args.command == "hooks":
        result = _hook_smoke(config)
    else:
        audit = run_parameter_audit(config)
        data = DynamicSensoriumDataModule(config)
        benchmarks = {name: benchmark_reference(name, config, data) for name in MODELS}
        training = {name: train_reference(name, config) for name in MODELS}
        evaluation = _evaluate(config)
        hooks = _hook_smoke(config)
        result = {"audit": audit, "benchmarks": benchmarks, "training": training, "evaluation": evaluation, "hooks": hooks}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

