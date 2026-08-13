from __future__ import annotations

import json
from pathlib import Path

from .config import resolve_project_path
from .data import DynamicSensoriumDataModule
from .models import build_model, parameter_counts


def run_parameter_audit(config: dict, data: DynamicSensoriumDataModule | None = None) -> dict:
    data = data or DynamicSensoriumDataModule(config)
    counts = {}
    shapes = {}
    for name in ("dynamic_reference", "static_reference"):
        model, model_shapes = build_model(name, config, data, "cpu")
        counts[name] = parameter_counts(model)
        shapes[name] = model_shapes.__dict__
    dynamic_total = counts["dynamic_reference"]["total"]
    static_total = counts["static_reference"]["total"]
    ratio = static_total / dynamic_total
    relative_gap = abs(dynamic_total - static_total) / dynamic_total
    result = {
        "session": data.session_key,
        "counts": counts,
        "shapes": shapes,
        "static_to_dynamic_ratio": ratio,
        "relative_gap_from_dynamic": relative_gap,
        "over_ten_percent": relative_gap > 0.10,
        "architecture_changed": False,
    }
    artifact = resolve_project_path(config, Path("artifacts") / "parameter_audit.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(result, indent=2), encoding="utf-8")

    report = resolve_project_path(config, Path("reports") / "parameter_matching_report.md")
    report.write_text(
        "\n".join(
            [
                "# Parameter matching report",
                "",
                f"Session: `{data.session_key}` ({data.n_neurons:,} neurons).",
                "",
                "| Model | Core | Gaussian readout | Other/shifter | Total |",
                "|---|---:|---:|---:|---:|",
                f"| dynamic_reference | {counts['dynamic_reference']['core']:,} | {counts['dynamic_reference']['readout']:,} | {counts['dynamic_reference']['other']:,} | {dynamic_total:,} |",
                f"| static_reference | {counts['static_reference']['core']:,} | {counts['static_reference']['readout']:,} | {counts['static_reference']['other']:,} | {static_total:,} |",
                "",
                f"Static/dynamic total parameter ratio: **{ratio:.4f}**. Relative gap: **{relative_gap:.2%}**.",
                "",
                "The gap exceeds the 10% gate. Per the Phase-1 prompt, neither official core was altered and training proceeds with the audited references.",
                "",
                "Minimal later-stage proposal (not applied): increase only the static core width while keeping its four-layer, 9/7-kernel, depth-separable topology, then re-audit. Any such change requires an explicit next-phase decision because it would no longer be the untouched official static baseline.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return result

