"""Audit, predict and evaluate without changing historical phase commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .unified_contract import (cache_identity, canonical_digest, load_cache,
                               validate_bundle, validate_manifest)

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT / "configs" / "unified_v1.json"
UNFITTED_GPFA = canonical_digest("sensorium_part3_prediction_stage_gpfa_not_fitted")


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified frozen Sensorium comparison; historical runs remain separate")
    parser.add_argument("command", choices=("audit", "synthetic", "predict", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--static-checkpoint", type=Path)
    parser.add_argument("--dynamic-checkpoint", type=Path)
    parser.add_argument("--sensorium-source", type=Path, help="Pinned external sensorium_2023 source directory")
    parser.add_argument("--neuralpredictors-source", type=Path, help="Pinned external neuralpredictors source directory")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    config = _load(args.config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new empty output directory; completed artifacts are never overwritten")
    checkpoints = {name: getattr(args, f"{name}_checkpoint") for name in ("static", "dynamic")}
    if args.command == "audit":
        if args.data_root is None:
            parser.error("audit requires --data-root")
        from .unified_audit import audit_dataset
        from .unified_reporting import write_inventory_status
        manifest = audit_dataset(args.data_root, config, args.output_dir, checkpoint_paths=checkpoints)
        write_inventory_status(manifest, args.output_dir)
        result = validate_manifest(manifest, require_test=False)
        print(json.dumps({"status": "ready" if result["ready"] else "blocked",
                          "output_dir": str(args.output_dir), "counts": result["counts"],
                          "reasons": manifest["eligibility"]["reasons"]}, indent=2))
        raise SystemExit(0 if result["ready"] else 2)
    if args.command == "synthetic":
        from .unified_fixture import synthetic_fixture
        from .unified_analysis import run_analysis
        bundle, manifest, config = synthetic_fixture(config)
        summary = run_analysis(bundle, manifest, config, args.output_dir, synthetic=True)
    else:
        if args.manifest is None:
            parser.error("predict and run require --manifest from a successful audit")
        manifest = _load(args.manifest)
        if manifest.get("synthetic"):
            parser.error("Artificial data are accepted only by the synthetic command")
        if manifest["config_sha256"] != canonical_digest(config):
            parser.error("The configuration differs from the audited manifest")
        try:
            validate_manifest(manifest)
        except ValueError as exc:
            if str(exc).startswith("Scientific evaluation is blocked:"):
                reasons = "; ".join(manifest["eligibility"]["reasons"])
                parser.exit(2, f"Scientific evaluation is blocked: {reasons or 'input eligibility failed'}. "
                               "See the manifest for trial-level support and provenance.\n")
            parser.exit(2, f"{exc}\n")
        if args.command == "predict":
            if args.data_root is None or not all(checkpoints.values()):
                parser.error("predict requires --data-root and both checkpoint files")
            from .unified_predictions import export_predictions
            summary = export_predictions(config, manifest, args.data_root, checkpoints,
                                         args.output_dir, args.device,
                                         args.sensorium_source, args.neuralpredictors_source)
        else:
            if args.cache is None:
                parser.error("run requires the new protocol's stamped --cache")
            bundle = load_cache(args.cache, cache_identity(manifest, UNFITTED_GPFA))
            validate_bundle(bundle, manifest)
            from .unified_analysis import run_analysis
            summary = run_analysis(bundle, manifest, config, args.output_dir, synthetic=False)
    print(json.dumps(summary, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
