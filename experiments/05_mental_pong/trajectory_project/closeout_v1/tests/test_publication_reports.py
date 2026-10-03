"""Ensure report rendering preserves complete narratives and scientific inputs."""
from pathlib import Path
import hashlib

from sklearn.linear_model import LinearRegression

from trajectory_project import publication_reports as reports


def test_saved_tables_render_all_complete_published_documents(tmp_path, monkeypatch):
    def reject_fit(*args, **kwargs):
        raise AssertionError("Saved-table report rendering must not fit a model.")

    monkeypatch.setattr(LinearRegression, "fit", reject_fit)
    roots = [reports.PROJECT / "closeout_v1", reports.PROJECT / "condition_endpoint_fa_gpfa_v1"]
    inputs = [p for root in roots for p in root.rglob("*")
              if p.is_file() and p.suffix in (".csv", ".gz", ".json", ".npz")]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    paths = reports.render_closeout(roots[0], tmp_path / "closeout_v1")
    paths += reports.render_condition(roots[1], tmp_path / "condition_endpoint_fa_gpfa_v1")
    paths += reports.render_descriptive(roots[0], tmp_path / "closeout_v1")
    assert len(paths) == 9
    for path in paths:
        expected = reports.PROJECT / path.relative_to(tmp_path)
        assert path.read_bytes() == expected.read_bytes(), str(expected)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}


def test_unfilled_table_tokens_fail_before_writing(tmp_path):
    import pytest

    output = tmp_path / "incomplete.md"
    with pytest.raises(ValueError, match="Template/table mismatch"):
        reports.render_template("closeout_report", output, [])
    assert not output.exists()
