import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trajectory_project.trial_endpoint_fa_gpfa_v1.trial_data import endpoint_candidates, deduplicate_records, fit_condition_geometry


def test_no_bounce_own_endpoint_and_initial_anchor():
    xy = np.array([[0., 2.], [5., 3.], [10., 4.]])
    y = endpoint_candidates(xy, [-2., 7.], x_start=0, y_start=2, x_end=10)
    np.testing.assert_array_equal(y[:, :, 0], np.tile(xy[:, 0], (2, 1)))
    np.testing.assert_allclose(y[:, 0, 1], 2)
    np.testing.assert_allclose(y[:, -1, 1], [-2, 7])
    np.testing.assert_allclose(y[:, 1, 1], [0, 4.5])


def test_collision_prefix_unchanged_and_continuity():
    xy = np.array([[0., 0.], [3., 6.], [5., 10.], [7., 6.], [10., 0.]])
    result = endpoint_candidates(xy, [-8., 12.], x_start=0, y_start=0, x_end=10,
                                 collision_x=5, collision_y=10)
    np.testing.assert_array_equal(result[:, :3], np.tile(xy[:3], (2, 1, 1)))
    np.testing.assert_allclose(result[:, -1, 1], [-8, 12])
    # Endpoint 12 is deliberately not clipped to the physical wall.
    assert result[1, -1, 1] == 12


def test_missing_endpoint_is_not_physical_reference():
    xy = np.array([[0., 0.], [10., 3.]])
    result = endpoint_candidates(xy, [np.nan], x_start=0, y_start=0, x_end=10)
    assert np.isnan(result).all()


def test_degenerate_collision_anchor_rejected():
    with pytest.raises(ValueError):
        endpoint_candidates(np.array([[1., 2.]]), [3], x_start=0, y_start=0,
                            x_end=10, collision_x=10, collision_y=2)


def test_same_endpoint_reproduces_straight_objective():
    xy = np.c_[np.arange(11.), 1 + .2*np.arange(11.)]
    candidate = endpoint_candidates(xy, [3.], x_start=0, y_start=1, x_end=10)
    np.testing.assert_allclose(candidate[0], xy, atol=1e-15)


def test_duplicate_session_trial_dedup_and_conflict():
    records = pd.DataFrame({"animal": ["a", "a"], "session": ["s", "s"],
                            "t_sync_on_mw": [1., 1.], "source_trial_row": [0, 1], "paddle_y": [2., 2.]})
    result, n = deduplicate_records(records)
    assert len(result) == 1 and n == 1
    records.loc[1, "paddle_y"] = 3.
    with pytest.raises(ValueError):
        deduplicate_records(records)


def synthetic_inputs(bounce=False):
    times = np.arange(1, 21)*50.
    center = times - 25.5
    x = .01*center
    y = np.where(x < 5, 2*x, 20-2*x) if bounce else 1+.2*x
    p = {"bin_available_ms": times, "xy": np.array([np.c_[x, y]]),
         "valid": np.ones((1, len(times)), bool)}
    e = pd.Series({"condition_index": 0, "condition_id": 1, "split": "train",
                   "metadata_bounce_count": int(bounce),
                   "bounce_class": "visible_bounce" if bounce else "no_bounce",
                   "events_json": '[{"pre_bin":9,"post_bin":10}]' if bounce else "[]"})
    meta = pd.Series({"x0_mwk": 0, "y0_mwk": 0 if bounce else 1, "t_f": 17, "t_occ": 8})
    return e, meta, p


def test_bin_mean_to_right_edge_is_fixed_not_lag_search():
    e, m, p = synthetic_inputs()
    g, xy, mask = fit_condition_geometry(e, m, p)
    assert g["geometry_valid"]
    np.testing.assert_allclose(xy[:, 0], .01*p["bin_available_ms"], atol=1e-12)
    np.testing.assert_allclose(xy[:, 1], 1+.2*xy[:, 0], atol=1e-12)
    assert abs(g["T_ms"]-1000) < 1e-9
    assert abs(g["initial_y_fit_error"]) < 1e-12


def test_collision_intersection_and_end_time():
    e, m, p = synthetic_inputs(True)
    g, xy, mask = fit_condition_geometry(e, m, p)
    assert g["geometry_valid"]
    np.testing.assert_allclose([g["collision_x"],g["collision_y"],g["collision_time_ms"]], [5,10,500], atol=1e-9)
    np.testing.assert_allclose(xy[9], [5,10], atol=1e-10)


def test_unresolved_collision_remains_invalid():
    e, m, p = synthetic_inputs()
    e["metadata_bounce_count"] = 1
    e["bounce_class"] = "unresolved_terminal_bounce"
    g, _, mask = fit_condition_geometry(e, m, p)
    assert not g["geometry_valid"]
    assert not mask.any()
    assert "unresolved" in g["invalid_reason"]


def test_known_reflection_needs_only_one_pure_post_point():
    e, m, p = synthetic_inputs(True)
    p["valid"][0, 12:] = False
    g, _, mask = fit_condition_geometry(e, m, p)
    assert g["geometry_valid"]
    assert g["n_post_branch_points"] == 1
    assert g["collision_estimation_method"] == "known_single_elastic_reflection_with_one_post_point"
    np.testing.assert_allclose([g["collision_x"], g["collision_y"]], [5, 10], atol=1e-10)
