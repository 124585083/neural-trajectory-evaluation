"""Ensure historical report rendering preserves saved tables and source inputs."""
from pathlib import Path
import hashlib

from sklearn.linear_model import LinearRegression

from trajectory_project import publication_reports as reports


def test_saved_tables_render_historical_editions_without_changing_inputs(tmp_path, monkeypatch):
    def reject_fit(*args, **kwargs):
        raise AssertionError("Saved-table report rendering must not fit a model.")

    monkeypatch.setattr(LinearRegression, "fit", reject_fit)
    roots = [reports.PROJECT / "closeout_v1", reports.PROJECT / "condition_endpoint_fa_gpfa_v1"]
    inputs = [p for root in roots + [reports.results_directory(root) for root in roots] for p in root.rglob("*")
              if p.is_file() and p.suffix in (".csv", ".gz", ".json", ".npz")]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    paths = reports.render_closeout(roots[0], tmp_path / "closeout_v1")
    paths += reports.render_condition(roots[1], tmp_path / "condition_endpoint_fa_gpfa_v1")
    paths += reports.render_descriptive(roots[0], tmp_path / "closeout_v1")
    assert len(paths) == 9
    for path in paths:
        assert path.is_file()
        assert "{{TABLE_" not in path.read_text(encoding="utf-8")
    # Numerical tables remain derived from saved rows even when an old report
    # edition is no longer part of the current documentation tree.
    table = reports.markdown(reports.closeout_tables(roots[0])[0])
    assert table in (tmp_path / "closeout_v1/FINAL_SUMMARY.md").read_text(encoding="utf-8")
    condition_text = (tmp_path / "condition_endpoint_fa_gpfa_v1/REPORT.md").read_text(encoding="utf-8")
    assert all(reports.markdown(table) in condition_text for table in reports.condition_tables(roots[1]))
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}


def test_custom_source_is_not_replaced_by_bundled_tables(tmp_path):
    import pytest

    explicit = tmp_path / "closeout_v1"
    assert reports.results_directory(explicit) == explicit / "results"
    with pytest.raises(FileNotFoundError):
        reports.closeout_tables(explicit)


def test_unfilled_table_tokens_fail_before_writing(tmp_path):
    import pytest

    output = tmp_path / "incomplete.md"
    with pytest.raises(ValueError, match="Template/table mismatch"):
        reports.render_template("closeout_report", output, [])
    assert not output.exists()
