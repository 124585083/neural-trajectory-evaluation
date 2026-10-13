from trajectory_model_eval.evidence_layout import evidence_file
from trajectory_model_eval.supplementary_cli import inventory
import hashlib
import json


def test_original_manifest_verifies_relocated_figure(tmp_path):
    root = tmp_path / "results/sensorium/evidence/run"
    family = root / "gpfa"
    family.mkdir(parents=True)
    figure = tmp_path / "results/sensorium/figures/run/gpfa/repeat.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"original figure bytes")
    checksum = hashlib.sha256(figure.read_bytes()).hexdigest()
    (family / "manifest.json").write_text(json.dumps({"status": "complete", "files": {"repeat.png": checksum}}))
    assert evidence_file(family, "repeat.png") == figure
    assert inventory(root)["supplementary_diagnostics"]["gpfa"]["verified_files"] == 1
    assert not evidence_file(family, "missing.csv").exists()
