"""Real payloads are optional until derived-data redistribution is resolved."""
import importlib.util
import os
from pathlib import Path

import pytest

MINI = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('mini_replay_test_core', MINI/'replay.py')
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


@pytest.fixture
def real_bundle():
    path = Path(os.environ.get('MENTAL_PONG_TEST_MINI_BUNDLE', MINI))
    if not (path/'bundle/arrays.npz').is_file():
        pytest.skip('Real payload withheld or unavailable; no synthetic scientific substitute is used.')
    replay.load_bundle(path)
    return path
