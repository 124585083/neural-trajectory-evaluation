"""Portable entry for diagnostics attached to the completed Sensorium run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .evidence_layout import evidence_file


MODULE = Path(__file__).resolve().parents[2]
REPOSITORY = MODULE.parents[1]


def inventory(public_dir):
    """Verify saved diagnostic records without loading or fitting any model."""
    root = Path(public_dir)
    rows = {}
    for family in ("gpfa", "encoder", "reversal"):
        directory = root / family
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            rows[family] = {"status": "not_complete", "verified_files": 0}
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, expected in manifest["files"].items():
            path = evidence_file(directory, relative)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Diagnostic artifact differs from its manifest: {family}/{relative}")
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else manifest
        rows[family] = {"status": summary.get("status", "recorded"),
                        "verified_files": len(manifest["files"]),
                        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    return {"primary_run": "retrained_v1", "supplementary_diagnostics": rows,
            "gpfa_sensitivity": "withdrawn_from_current_conclusions; retained records only; no further fitting",
            "scope": "File verification; primary completion and scientific scores are unchanged"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reversal", "encoder-predict", "encoder-score", "status"))
    parser.add_argument("--run-root", type=Path, default=MODULE / "outputs/retrained_v1")
    parser.add_argument("--public-dir", type=Path,
                        default=REPOSITORY / "results/sensorium/evidence/04_model_comparison/retrained_v1_diagnostics")
    parser.add_argument("--config", type=Path, help="Explicit fixed diagnostic config; defaults depend on stage")
    parser.add_argument("--data-root")
    parser.add_argument("--sensorium-source")
    parser.add_argument("--neuralpredictors-source")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage == "status":
        print(json.dumps(inventory(args.public_dir), indent=2))
        return
    family = args.stage.split("-")[0]
    config = args.config or MODULE / f"configs/supplementary_{family}_v1.json"
    if family == "encoder":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from .supplementary_encoder import main as encoder_main
        values = [args.stage.split("-")[1], "--run-root", str(args.run_root), "--config", str(config),
                  "--output-dir", str(args.run_root / "diagnostics/encoder"),
                  "--public-dir", str(args.public_dir / family), "--device", args.device]
        for flag, value in (("--data-root", args.data_root), ("--sensorium-source", args.sensorium_source),
                            ("--neuralpredictors-source", args.neuralpredictors_source)):
            if value is not None:
                values.extend((flag, value))
        encoder_main(values)
    else:
        from .supplementary_reversal import run
        run(args.run_root, args.public_dir / family, config)


if __name__ == "__main__":
    main()
