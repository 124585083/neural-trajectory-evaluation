import sys

import pytest

from trajectory_model_eval import retrained_cli, retrained_reporting


def test_separate_run_cannot_overwrite_primary_public_records(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(retrained_reporting, "publish_status", lambda *args, **kwargs: called.append((args, kwargs)))
    monkeypatch.setattr(sys, "argv", ["run_retrained.py", "report", "--output-dir", str(tmp_path / "other")])
    with pytest.raises(SystemExit) as error:
        retrained_cli.main()
    assert error.value.code == 2
    assert not called


def test_separate_run_report_uses_explicit_destination(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(retrained_reporting, "publish_status", lambda *args, **kwargs: called.append((args, kwargs)))
    root, output = tmp_path / "other", tmp_path / "compact"
    monkeypatch.setattr(sys, "argv", ["run_retrained.py", "report", "--output-dir", str(root), "--public-dir", str(output)])
    retrained_cli.main()
    assert called == [((root.resolve(), output.resolve()), {"matching_root": None})]


@pytest.mark.parametrize("destination", [".", "matching", "matching/public"])
def test_checkpoint_publication_cannot_overlap_run(tmp_path, destination):
    with pytest.raises(ValueError, match="separate"):
        retrained_reporting.publish_checkpoint_status(tmp_path / "matching", tmp_path / destination)
