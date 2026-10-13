import hashlib
import json

import pytest

from trajectory_model_eval.supplementary_cli import inventory


def test_status_distinguishes_missing_diagnostics_and_changed_outputs(tmp_path):
    assert inventory(tmp_path)["supplementary_diagnostics"]["gpfa"]["status"] == "not_complete"
    directory = tmp_path / "reversal"
    directory.mkdir()
    score = directory / "score.csv"
    score.write_bytes(b"metric,value\nposition,0.5\n")
    record = {"status": "complete", "files": {"score.csv": hashlib.sha256(score.read_bytes()).hexdigest()}}
    (directory / "manifest.json").write_text(json.dumps(record), encoding="utf-8")
    result = inventory(tmp_path)["supplementary_diagnostics"]["reversal"]
    assert result["status"] == "complete" and result["verified_files"] == 1
    score.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs"):
        inventory(tmp_path)


def test_withdrawn_sensitivity_has_no_active_fitting_command():
    from trajectory_model_eval.supplementary_cli import main
    with pytest.raises(SystemExit) as error:
        main(["gpfa"])
    assert error.value.code == 2
