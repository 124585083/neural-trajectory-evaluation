"""Real terminal behavior records and endpoint-constrained trajectory labels.

No neural response is selected, fitted, or paired to a real trial here. The
released response arrays are condition means; trial labels remain independent.
"""
from __future__ import annotations

import gc
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PILOT = PROJECT.parent
from trajectory_project.runtime_paths import configured_root
DATA = configured_root("source_data")
from trajectory_project._dependencies import verified_file

MODE = "trial_labels_with_mean_neural"
GEOMETRY_PROTOCOL = {
    "version": 1,
    "objective_source": "released TRUE condition-bin means; no neural results",
    "timestamp": "mean integer-ms centers 50*k+24.5; evaluate physical branches at 50*k+50",
    "x_model": "OLS straight x(time) from existing conservative valid physical bins",
    "x_end": 10.0,
    "arrival": "T=(10-x_intercept)/x_slope, retained alongside design T; no timing search",
    "no_bounce": "y(x) straight OLS; candidate starts at actual metadata x0,y0",
    "known_single_collision_pre_support": "indices < audited pre_bin-1, at least 2 points",
    "known_single_collision_post_support": "indices >= audited post_bin+1, at least 1 point",
    "collision_two_or_more_post_points": "OLS each y(x) branch and intersect",
    "collision_one_post_point": "Elastic vertical reflection: post_slope=-pre_slope; single post point determines intercept; intersect with pre branch",
    "collision_no_post_event_or_points": "unknown, no invented anchor or wall",
    "collision_accuracy": "Estimated from 50ms released means, retain audit interval; never exact raw trial log",
    "physical_support": "existing conservative valid support, completed bins <=estimated interception T",
    "neural_score_accessed": False,
    "behavior_endpoint_used_for_physical_geometry": False,
}
PHYSICAL_FIELDS = ["py_x0", "py_y0", "py_heading0", "py_speed", "py_n_bounce",
                   "x0_mworks", "y0_mworks", "yf_mworks", "ball_offset_x",
                   "ball_offset_y", "ball_heading", "ball_speed"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def record(path, purpose):
    return {"path": str(Path(path).resolve()), "sha256": sha(path), "purpose": purpose}


def deduplicate_records(frame):
    """Only identical animal/session/sync records can be collapsed.

    Conflicting repeated records are errors, never arbitrarily selected trials.
    Source row numbers are excluded from the equality comparison.
    """
    keys = ["animal", "session", "t_sync_on_mw"]
    if frame[keys].isna().any().any():
        raise ValueError("Cannot establish stable real-record identity without animal/session/sync")
    dup = frame.duplicated(keys, keep=False)
    for _, group in frame.loc[dup].groupby(keys, sort=False):
        check = group.drop(columns=["source_trial_row"], errors="ignore")
        if len(check.drop_duplicates()) != 1:
            raise ValueError("Conflicting records sharing real-trial key")
    return frame.drop_duplicates(keys, keep="first").copy(), int(frame.duplicated(keys).sum())


def endpoint_candidates(xy_objective, endpoint, *, x_start, y_start, x_end,
                        collision_x=None, collision_y=None):
    """Independent endpoint labels; no movement/stability/event detection.

    The input x coordinates and output x coordinates are bit-identical.
    A non-finite endpoint produces missing labels rather than a physical copy.
    """
    xy = np.asarray(xy_objective, dtype=np.float64)
    endpoints = np.asarray(endpoint, dtype=np.float64).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Expected objective time x xy")
    has_collision = collision_x is not None or collision_y is not None
    anchor_x, anchor_y = (collision_x, collision_y) if has_collision else (x_start, y_start)
    if (not np.isfinite([anchor_x, anchor_y, x_end]).all()
            or float(x_end) <= float(anchor_x)):
        raise ValueError("Finite ordered physical anchor and intercept plane required")
    labels = np.broadcast_to(xy, (len(endpoints), *xy.shape)).copy()
    after = xy[:, 0] >= anchor_x if has_collision else np.ones(len(xy), bool)
    alpha = (xy[after, 0] - anchor_x) / (x_end - anchor_x)
    labels[:, after, 1] = anchor_y + alpha[None, :] * (endpoints[:, None] - anchor_y)
    labels[~np.isfinite(endpoints)] = np.nan
    return labels


def fit_condition_geometry(event, metadata, prepared):
    """Convert published bin-mean physical reference to fixed bin-end labels.

    The publisher bins integer-ms samples [50k,50k+49]. Straight branches
    therefore have mean timestamp 50k+24.5; predictions are available at 50k+50.
    Fit x(t) and y(x) branches from audited uncontaminated physical support,
    and evaluate at that fixed right endpoint. No neural values are accessed.
    """
    ci, cid = int(event.condition_index), int(event.condition_id)
    times = np.asarray(prepared["bin_available_ms"], float)
    centers = times - 25.5
    xy = np.asarray(prepared["xy"][ci], float)
    valid = np.asarray(prepared["valid"][ci], bool) & np.isfinite(xy).all(axis=-1)
    ix = np.flatnonzero(valid)
    observed = json.loads(event.events_json)
    count = int(event.metadata_bounce_count)
    kind = str(event.bounce_class)
    row = {"condition_index": ci, "condition_id": cid, "original_split": str(event.split),
           "bounce_class": kind, "metadata_bounce_count": count,
           "x0": float(metadata.x0_mwk), "y0": float(metadata.y0_mwk), "x_end": 10.,
           "T_design_ms": float(300 + 41 * metadata.t_f),
           "occ_design_ms": float(300 + 41 * metadata.t_occ),
           "bin_mean_center_to_available_ms": 25.5,
           "collision_source": "published_50ms_physical_branch_intersection",
           "exact_trial_collision_log": False, "geometry_valid": False,
           "invalid_reason": "", "collision_time_ms": np.nan,
           "collision_low_ms": np.nan, "collision_high_ms": np.nan,
           "collision_x": np.nan, "collision_y": np.nan}
    objective = np.full(xy.shape, np.nan, float)
    if len(ix) < 3:
        row["invalid_reason"] = "insufficient_published_objective_support"
        return row, objective, np.zeros(len(times), bool)
    xcoef = np.polyfit(centers[ix], xy[ix, 0], 1)
    if not np.isfinite(xcoef).all() or xcoef[0] <= 0:
        row["invalid_reason"] = "invalid_horizontal_motion"
        return row, objective, np.zeros(len(times), bool)
    T = float((row["x_end"] - xcoef[1]) / xcoef[0])
    launch = float((row["x0"] - xcoef[1]) / xcoef[0])
    objective[:, 0] = np.polyval(xcoef, times)
    row.update(T_ms=T, launch_estimate_ms=launch, vx_per_ms=float(xcoef[0]),
               x_time_intercept=float(xcoef[1]),
               x_branch_rmse=float(np.sqrt(np.mean((xy[ix, 0] - np.polyval(xcoef, centers[ix]))**2))))
    if count == 0 and len(observed) == 0 and kind == "no_bounce":
        pre = np.polyfit(xy[ix, 0], xy[ix, 1], 1)
        objective[:, 1] = np.polyval(pre, objective[:, 0])
        row.update(y_pre_slope=float(pre[0]), y_pre_intercept=float(pre[1]),
                   y_branch_rmse=float(np.sqrt(np.mean((xy[ix, 1] - np.polyval(pre, xy[ix, 0]))**2))),
                   initial_y_fit_error=float(np.polyval(pre, row["x0"]) - row["y0"]))
    elif count == 1 and len(observed) == 1:
        e = observed[0]
        before = ix[ix < int(e["pre_bin"]) - 1]
        after = ix[ix >= int(e["post_bin"]) + 1]
        row.update(n_pre_branch_points=int(len(before)), n_post_branch_points=int(len(after)))
        if len(before) < 2 or len(after) < 1:
            row["invalid_reason"] = "insufficient_unmixed_collision_branches"
            return row, objective, np.zeros(len(times), bool)
        pre = np.polyfit(xy[before, 0], xy[before, 1], 1)
        if len(after) >= 2:
            post = np.polyfit(xy[after, 0], xy[after, 1], 1)
            row["collision_estimation_method"] = "two_independent_linear_branches"
        else:
            post = np.array([-pre[0], xy[after[0], 1] + pre[0]*xy[after[0], 0]])
            row["collision_estimation_method"] = "known_single_elastic_reflection_with_one_post_point"
        if abs(pre[0] - post[0]) < 1e-12:
            row["invalid_reason"] = "no_unique_branch_intersection"
            return row, objective, np.zeros(len(times), bool)
        xb = float((post[1] - pre[1]) / (pre[0] - post[0]))
        yb = float(np.polyval(pre, xb))
        tb = float((xb - xcoef[1]) / xcoef[0])
        low = float((int(e["pre_bin"]) - 1) * 50)
        high = float((int(e["post_bin"]) + 1) * 50)
        if not (row["x0"] < xb < row["x_end"] and launch < tb < T):
            row["invalid_reason"] = "collision_outside_launch_to_intercept"
            return row, objective, np.zeros(len(times), bool)
        objective[:, 1] = np.where(objective[:, 0] < xb,
                                   np.polyval(pre, objective[:, 0]), np.polyval(post, objective[:, 0]))
        error = np.r_[xy[before, 1] - np.polyval(pre, xy[before, 0]),
                      xy[after, 1] - np.polyval(post, xy[after, 0])]
        row.update(collision_x=xb, collision_y=yb, collision_time_ms=tb,
                   collision_low_ms=min(low, np.floor(tb/50)*50),
                   collision_high_ms=max(high, np.ceil(tb/50)*50),
                   y_pre_slope=float(pre[0]), y_pre_intercept=float(pre[1]),
                   y_post_slope=float(post[0]), y_post_intercept=float(post[1]),
                   y_branch_rmse=float(np.sqrt(np.mean(error**2))),
                   initial_y_fit_error=float(np.polyval(pre, row["x0"]) - row["y0"]))
    else:
        row["invalid_reason"] = "unresolved_terminal_or_unsupported_collision_history"
        return row, objective, np.zeros(len(times), bool)
    # Conservative release support and fully completed neural bins are shared.
    # No feedback/end-crossing bin is rescued through interpolation or padding.
    support = valid & (times > launch) & (times <= T)
    row.update(geometry_valid=True, n_physical_bins=int(support.sum()),
               objective_end_y=float(np.polyval(post if count else pre, row["x_end"])),
               T_minus_design_ms=float(T - row["T_design_ms"]))
    return row, objective, support


def _load_raw_table(animal):
    source = verified_file(DATA, f"{animal}_hand_dmfc_dataset_50ms.pkl")
    cache = HERE / f"{animal}_raw_trial_meta.csv.gz"
    # The initial audit cache is all rows/all original fields, without pruning.
    if cache.exists():
        frame = pd.read_csv(cache)
    else:
        raw = pd.read_pickle(source)
        frame = raw["trial_meta"].copy()
        frame["source_trial_row"] = np.arange(len(frame), dtype=np.int64)
        frame.to_csv(cache, index=False)
        del raw
        gc.collect()
    return frame, source, cache


def build_trial_data(output_root=HERE):
    """Write actual trial labels and geometry; return memory-efficient payloads."""
    out = Path(output_root)
    for folder in ("artifacts", "results", "sources"):
        (out/folder).mkdir(parents=True, exist_ok=True)
    geometry_protocol_path = out/"configs/label_geometry_protocol.json"
    if geometry_protocol_path.exists():
        if json.loads(geometry_protocol_path.read_text(encoding="utf-8")) != GEOMETRY_PROTOCOL:
            raise ValueError("Frozen label geometry protocol changed")
    else:
        save_json(geometry_protocol_path, GEOMETRY_PROTOCOL)
    frozen_path = PROJECT/"configs/frozen_representation.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    event_path = PROJECT/"results/condition_event_table.csv"
    events = pd.read_csv(event_path)
    condition_ids = events.condition_id.to_numpy(np.int64)
    meta_path = verified_file(DATA, "valid_meta_sample_full.pkl")
    meta = pd.read_pickle(meta_path).set_index("meta_index")
    outputs, audit, geometry_rows, condition_rows = {}, {}, [], []
    sources = [record(geometry_protocol_path, "fixed before decoder outcomes, physical-only geometry estimation"),
               record(frozen_path, "frozen identities and conservative support paths"),
               record(event_path, "physical collision audit, never neural turning"),
               record(meta_path, "task initial coordinates and condition timing")]
    for animal in ("mahler", "perle"):
        raw, source, cache = _load_raw_table(animal)
        sources.extend([record(source, "published animal dataset"), record(cache, "complete scalar record extraction")])
        ids_ok = raw.py_meta_index.isin(condition_ids)
        category_ok = raw.occ_alpha.eq(1)
        frame = raw.loc[ids_ok & category_ok].copy()
        frame.insert(0, "animal", animal)
        n_before_dedup = len(frame)
        frame, duplicates = deduplicate_records(frame)
        frame = frame.sort_values("source_trial_row").reset_index(drop=True)
        frame["condition_id"] = frame.py_meta_index.astype(np.int64)
        lookup = {int(cid): i for i, cid in enumerate(condition_ids)}
        frame["condition_index"] = frame.condition_id.map(lookup).astype(np.int64)
        frame["source_row"] = frame.source_trial_row.astype(np.int64)
        frame["session_id"] = frame.session.astype(str)
        frame["trial_id"] = [f"{animal}:source_row_{int(i)}" for i in frame.source_row]
        frame["source_file"] = str(source)
        frame["record_identity"] = [f"{animal}|{s}|{t:.17g}" for s, t in zip(frame.session, frame.t_sync_on_mw)]
        frame["data_mode"] = MODE
        frame["neural_match_status"] = "no_released_trial_response_or_unit_half_membership"
        frame["endpoint_source"] = "trial_meta.paddle_y_assigned_from_joystick_output"
        frame["endpoint_units"] = "MWorks_centered_display_coordinate_units"
        frame["endpoint_time_status"] = "terminal_scalar_exact_pre_feedback_time_unverified"
        frame["endpoint_valid"] = np.isfinite(frame.paddle_y) & frame.ignore.eq(0)
        frame["endpoint_status"] = np.where(frame.endpoint_valid, "published_terminal_scalar",
                                            np.where(frame.ignore.ne(0), "unknown_uncompleted_terminal_scalar", "missing_endpoint"))
        frame["original_split"] = frame.condition_id.map(events.set_index("condition_id")["split"])
        # This is a physical-field audit, not a neural or behavior-effect filter.
        variations = frame.groupby("condition_id")[PHYSICAL_FIELDS].nunique(dropna=False)
        if (variations > 1).any().any():
            raise ValueError("Trial physical parameters vary within condition; broadcasting would be false")
        variations.to_csv(out/"results"/f"{animal}_within_condition_physical_nunique.csv")
        paddle_x = frame.paddle_pos_x_from_mwk.to_numpy(float)
        if not np.isfinite(paddle_x).all() or not np.allclose(paddle_x, 10, atol=1e-9, rtol=0):
            raise ValueError("Released common physical paddle plane is not x=10")
        prepared_path = Path(frozen["animals"][animal]["prepared_path"])
        if sha(prepared_path) != frozen["animals"][animal]["prepared_sha256"]:
            raise ValueError("Frozen physical support source changed")
        with np.load(prepared_path, allow_pickle=False) as data:
            p = {k: data[k].copy() for k in ("condition_ids", "bin_available_ms", "xy", "valid", "visible", "hidden")}
        np.testing.assert_array_equal(p["condition_ids"], condition_ids)
        times = p["bin_available_ms"].astype(float)
        objective = np.full((len(condition_ids), len(times), 2), np.nan, float)
        geometric = np.zeros((len(condition_ids), len(times)), bool)
        behavior = np.full((len(frame), len(times), 2), np.nan, float)
        common = np.zeros((len(frame), len(times)), bool)
        epoch_condition = {name: np.zeros_like(geometric) for name in ("full", "visible", "hidden", "bounce", "no_bounce", "post_bounce")}
        animal_geometry = []
        for _, e in events.iterrows():
            ci, cid = int(e.condition_index), int(e.condition_id)
            g, xy, mask = fit_condition_geometry(e, meta.loc[cid], p)
            g["animal"] = animal
            trial_ref = frame[frame.condition_id.eq(cid)]
            g["trial_x0"] = float(trial_ref.py_x0.iloc[0])
            g["trial_y0"] = float(trial_ref.py_y0.iloc[0])
            g["initial_x_metadata_vs_trial_error"] = g["x0"] - g["trial_x0"]
            g["initial_y_metadata_vs_trial_error"] = g["y0"] - g["trial_y0"]
            g["trial_yf_mworks"] = float(trial_ref.yf_mworks.iloc[0])
            g["objective_end_y_minus_trial_yf"] = g.get("objective_end_y", np.nan) - g["trial_yf_mworks"]
            geometry_rows.append(g)
            animal_geometry.append(g)
            objective[ci], geometric[ci] = xy, mask
            idx = np.flatnonzero(frame.condition_index.to_numpy() == ci)
            eligible_idx = idx[frame.endpoint_valid.to_numpy()[idx]]
            epoch_condition["full"][ci] = mask
            epoch_condition["visible"][ci] = mask & p["visible"][ci]
            epoch_condition["hidden"][ci] = mask & p["hidden"][ci]
            epoch_condition["bounce" if int(e.metadata_bounce_count) else "no_bounce"][ci] = mask
            if g["geometry_valid"]:
                kwargs = dict(x_start=g["x0"], y_start=g["y0"], x_end=g["x_end"])
                if int(e.metadata_bounce_count):
                    kwargs.update(collision_x=g["collision_x"], collision_y=g["collision_y"])
                    epoch_condition["post_bounce"][ci] = mask & (times >= g["collision_time_ms"])
                endpoints = frame.paddle_y.to_numpy(float)[eligible_idx]
                behavior[eligible_idx] = endpoint_candidates(xy, endpoints, **kwargs)
                common[eligible_idx] = mask
                # Exact end-point identity is tested outside the observed bins.
                endxy = np.array([[g["x_end"], g["objective_end_y"]]], float)
                reconstructed_endpoint = endpoint_candidates(endxy, endpoints, **kwargs)[:, 0, 1]
                np.testing.assert_allclose(reconstructed_endpoint, endpoints, atol=1e-12, rtol=0)
            frame.loc[idx, "geometry_valid"] = bool(g["geometry_valid"])
            frame.loc[idx, "geometry_invalid_reason"] = g["invalid_reason"]
            frame.loc[idx, "bounce_class"] = g["bounce_class"]
            frame.loc[idx, "T_ms"] = g.get("T_ms", np.nan)
            frame.loc[idx, "collision_time_ms"] = g["collision_time_ms"]
            frame.loc[idx, "collision_x"] = g["collision_x"]
            frame.loc[idx, "collision_y"] = g["collision_y"]
            frame.loc[idx, "x_start"] = g["x0"]
            frame.loc[idx, "y_start"] = g["y0"]
            frame.loc[idx, "x_end"] = g["x_end"]
            condition_rows.append({"animal": animal, "condition_id": cid, "condition_index": ci,
                                   "n_records": int(len(idx)), "n_valid_endpoints": int(len(eligible_idx)),
                                   "n_complete_trial_labels": int(len(eligible_idx) if g["geometry_valid"] else 0),
                                   "n_label_rows": int(common[idx].sum()),
                                   "n_neural_trials": 0, "data_mode": MODE,
                                   "geometry_valid": bool(g["geometry_valid"]), "invalid_reason": g["invalid_reason"]})
        condition_index = frame.condition_index.to_numpy(np.int64)
        frame["n_label_bins"] = common.sum(axis=1)
        frame["candidate_valid"] = frame.n_label_bins.gt(0)
        records_path = out/"results"/f"{animal}_trial_records.csv.gz"
        frame.to_csv(records_path, index=False, float_format="%.17g")
        artifact_path = out/"artifacts"/f"{animal}_trial_labels.npz"
        np.savez_compressed(artifact_path, condition_ids=condition_ids, times_ms=times,
                            trial_id=frame.trial_id.to_numpy(str), condition_index=condition_index,
                            endpoint=frame.paddle_y.to_numpy(float), endpoint_valid=frame.endpoint_valid.to_numpy(bool),
                            objective_xy=objective, behavior_xy=behavior, common_mask=common,
                            **{f"epoch_condition_{k}": v for k, v in epoch_condition.items()})
        epoch_trial = {k: v[condition_index] & common for k, v in epoch_condition.items()}
        outputs[animal] = {"animal": animal, "condition_ids": condition_ids, "times_ms": times,
                           "trial_table": frame, "condition_index": condition_index,
                           "objective_xy": objective, "behavior_xy": behavior,
                           "common_mask": common, "epoch_masks": epoch_trial,
                           "epoch_condition_masks": epoch_condition,
                           "endpoint": frame.paddle_y.to_numpy(float), "geometry": pd.DataFrame(animal_geometry),
                           "data_mode": MODE, "n_neural_trials": 0,
                           "labels_path": str(artifact_path), "trial_records_path": str(records_path)}
        complete = frame.ignore.eq(0)
        completed_difference = np.abs(frame.loc[complete, "paddle_y"] - frame.loc[complete, "paddle_pos_y_from_mwk"])
        ignored_difference = np.abs(frame.loc[~complete, "paddle_y"] - frame.loc[~complete, "paddle_pos_y_from_mwk"])
        audit[animal] = {"published_scalar_rows": int(len(raw)), "all79_occ_records_before_dedup": n_before_dedup,
                         "duplicate_real_records_removed": duplicates, "all79_occ_records": len(frame),
                         "n_sessions": int(frame.session.nunique()), "n_valid_terminal_endpoints": int(frame.endpoint_valid.sum()),
                         "n_success": int((frame.success == 1).sum()), "n_failure": int((frame.failure == 1).sum()),
                         "n_terminal_status_unknown": int((~frame.endpoint_valid).sum()),
                         "n_trials_with_candidate_labels": int(frame.candidate_valid.sum()),
                         "n_label_rows": int(common.sum()), "all79_identity_retained": True,
                         "condition_physical_parameters_max_nunique": int(variations.to_numpy().max()),
                         "paddle_scalar_vs_display_completed_median_abs_difference": float(completed_difference.median()),
                         "paddle_scalar_vs_display_completed_max_abs_difference": float(completed_difference.max()),
                         "paddle_scalar_vs_display_ignore_max_abs_difference": float(ignored_difference.max()),
                         "endpoint_equals_joystick_output_max_error": float(np.max(np.abs(frame.paddle_y - frame.joystick_output))),
                         "data_mode": MODE, "n_independent_neural_trials": 0,
                         "terminal_time_verified_pre_feedback": False,
                         "sources": {"raw": str(source), "prepared": str(prepared_path)}}
        sources.append(record(prepared_path, "physical coordinate means and conservative mask; no response values accessed"))
        print(f"{animal}: {len(frame)} real records, {int(frame.endpoint_valid.sum())} endpoints, "
              f"{int(frame.candidate_valid.sum())} trial labels, {int(common.sum())} label rows", flush=True)
        del raw
        gc.collect()
    pd.DataFrame(geometry_rows).to_csv(out/"results/condition_geometry.csv", index=False, float_format="%.17g")
    pd.DataFrame(condition_rows).to_csv(out/"results/trial_label_coverage.csv", index=False)
    global_audit = {"route": "B", "data_mode": MODE, "route_A_available": False,
                    "reason": "Release has neuron x condition x time arrays; no real trial responses or per-unit stable/half trial-membership map.",
                    "endpoint_semantics": "paddle_y assigned from joystick_output and checked against display scalar; exact terminal sample time is not released.",
                    "ignore_policy": "Retain incomplete records in audit; predominantly zero terminal scalar conflicts with display, so unknown endpoint, never a valid zero target.",
                    "units": "MWorks centered display coordinate units, common interception plane x=10",
                    "objective_reference": "Physical parameters identical within condition; broadcast condition reference. Actual frame-by-frame trial timing is not released.",
                    "label_clock": "Published means use integer-ms center 50k+24.5; physical branch models evaluated at fixed bin end 50k+50.",
                    "arrival_time": "Solve released x(t) straight branch at physical paddle plane x=10; design discrete-step T also retained for comparison.",
                    "collision_clock": "Estimated from published 50ms branch intersection; uncertainty interval retained, not exact original single-trial log.",
                    "feedback_boundary": "Exclude completed bins ending after estimated physical interception and release conservative support; no exact original feedback log available.",
                    "old_stability_proxy_used": False, "endpoint_correctness_filter": False,
                    "independent_neural_trial_count": 0, "animals": audit}
    save_json(out/"results/data_pairing_audit.json", global_audit)
    save_json(out/"sources/trial_data_sources.json", sources)
    return outputs


def load_trial_data(output_root=HERE):
    out = Path(output_root)
    geometry = pd.read_csv(out/"results/condition_geometry.csv")
    result = {}
    for animal in ("mahler", "perle"):
        path = out/"artifacts"/f"{animal}_trial_labels.npz"
        with np.load(path, allow_pickle=False) as saved:
            a = {k: saved[k].copy() for k in saved.files}
        trial_table = pd.read_csv(out/"results"/f"{animal}_trial_records.csv.gz")
        masks = {k.removeprefix("epoch_condition_"): v for k, v in a.items() if k.startswith("epoch_condition_")}
        result[animal] = {"animal": animal, **a, "trial_table": trial_table,
                          "epoch_condition_masks": masks,
                          "epoch_masks": {k: v[a["condition_index"]] & a["common_mask"] for k, v in masks.items()},
                          "geometry": geometry[geometry.animal.eq(animal)].copy(),
                          "data_mode": MODE, "n_neural_trials": 0, "labels_path": str(path),
                          "trial_records_path": str(out/"results"/f"{animal}_trial_records.csv.gz")}
    return result


def write_sample_index(payload, path, common_mask=None, batch_trials=1024):
    """Compact trial×time identity manifest; neural inputs are referenced, not copied.

    Caller supplies the final FA/GPFA shared support after representation checks.
    Numeric unique_neural_input_id = condition_index * n_time + time_index.
    """
    mask = payload["common_mask"] if common_mask is None else np.asarray(common_mask, bool)
    if mask.shape != payload["common_mask"].shape or np.any(mask & ~payload["common_mask"]):
        raise ValueError("Common support must be a subset of physical/endpoint label support")
    trials = payload["trial_table"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        header = True
        for start in range(0, len(trials), batch_trials):
            local_trial, time_index = np.nonzero(mask[start:start+batch_trials])
            index = local_trial + start
            ci = payload["condition_index"][index]
            rows = pd.DataFrame({"trial_index": index, "trial_id": trials.trial_id.to_numpy()[index],
                                 "condition_id": payload["condition_ids"][ci], "condition_index": ci,
                                 "time_index": time_index, "time_ms": payload["times_ms"][time_index],
                                 "unique_neural_input_id": ci*len(payload["times_ms"])+time_index,
                                 "objective_condition_index": ci, "behavior_trial_index": index})
            rows.to_csv(stream, index=False, header=header)
            header = False
    return {"path": str(path), "sha256": sha(path), "n_rows": int(mask.sum()),
            "n_unique_neural_inputs": int(np.any(mask.reshape(mask.shape), axis=0).sum()) if not len(trials) else
            int(len(np.unique((payload["condition_index"][:, None]*mask.shape[1]+np.arange(mask.shape[1]))[mask]))),
            "n_neural_trials": 0, "data_mode": MODE}


def validate_actual_labels(output_root=HERE):
    """Read saved labels and test actual identities before decoder fitting."""
    out = Path(output_root)
    payloads = load_trial_data(out)
    checks, variability = {}, []
    for animal, a in payloads.items():
        frame = a["trial_table"]
        mask = a["common_mask"]
        ci = a["condition_index"]
        endpoint_error = anchor_error = x_error = prefix_error = identity_rel_error = 0.
        for c, g in a["geometry"].set_index("condition_index").iterrows():
            c = int(c)
            idx = np.flatnonzero((ci == c) & frame.candidate_valid.to_numpy(bool))
            if not len(idx):
                continue
            obj = a["objective_xy"][c]
            candidate = a["behavior_xy"][idx]
            x_error = max(x_error, float(np.max(np.abs(candidate[..., 0] - obj[None, :, 0]))))
            kwargs = dict(x_start=g.x0, y_start=g.y0, x_end=g.x_end)
            if int(g.metadata_bounce_count):
                kwargs.update(collision_x=g.collision_x, collision_y=g.collision_y)
                ax, ay = g.collision_x, g.collision_y
                pre = obj[:, 0] < ax
                prefix_error = max(prefix_error, float(np.max(np.abs(candidate[:, pre] - obj[None, pre]))))
            else:
                ax, ay = g.x0, g.y0
            endpoints = a["endpoint"][idx]
            end = endpoint_candidates(np.array([[g.x_end, g.objective_end_y]]), endpoints, **kwargs)
            anchor = endpoint_candidates(np.array([[ax, ay]]), endpoints, **kwargs)
            endpoint_error = max(endpoint_error, float(np.max(np.abs(end[:, 0, 1] - endpoints))))
            anchor_error = max(anchor_error, float(np.max(np.abs(anchor[:, 0, 1] - ay))))
            active = np.flatnonzero(mask[idx[0]])
            y = candidate[:, active, 1]
            mean = y.mean(axis=0)
            prediction = np.sin(np.arange(len(active), dtype=float))
            left = np.sum((y - prediction)**2)
            right = len(idx)*np.sum((mean-prediction)**2)+np.sum((y-mean)**2)
            identity_rel_error = max(identity_rel_error, float(abs(left-right)/max(1., abs(left))))
            variability.append({"animal": animal, "condition_id": int(g.condition_id),
                                "n_real_trials": len(idx), "n_sessions": int(frame.iloc[idx].session.nunique()),
                                "n_unique_endpoint_values": int(len(np.unique(endpoints))),
                                "endpoint_mean": float(endpoints.mean()), "endpoint_std": float(endpoints.std()),
                                "endpoint_min": float(endpoints.min()), "endpoint_max": float(endpoints.max()),
                                "y_obj_min": float(obj[active, 1].min()), "y_obj_max": float(obj[active, 1].max()),
                                "y_beh_min": float(y.min()), "y_beh_max": float(y.max()),
                                "y_beh_pooled_variance": float(y.var()),
                                "mean_trial_variance_at_common_times": float(y.var(axis=0).mean()),
                                "behavior_between_trial_sse": float(np.sum((y-mean)**2)),
                                "trial_permutation_within_condition_changes_B_fit": False})
        assert not np.isfinite(a["behavior_xy"][~frame.endpoint_valid.to_numpy(bool)]).any()
        assert endpoint_error < 1e-12 and anchor_error < 1e-12 and x_error == 0 and prefix_error == 0
        assert identity_rel_error < 1e-12
        assert len(a["condition_ids"]) == 79
        assert not frame.duplicated(["animal", "session", "t_sync_on_mw"]).any()
        checks[animal] = {"passed": True, "n_retained_condition_identities": 79,
                           "candidate_own_endpoint_max_abs_error": endpoint_error,
                           "candidate_anchor_max_abs_error": anchor_error,
                           "same_x_max_abs_error": x_error,
                           "before_collision_unchanged_max_abs_error": prefix_error,
                           "repeated_input_SSE_identity_max_relative_error": identity_rel_error,
                           "unknown_endpoints_are_all_missing": True,
                           "unique_real_record_keys": True,
                           "source_label_sha256": sha(a["labels_path"])}
    save_json(out/"results/actual_label_validation.json", checks)
    pd.DataFrame(variability).to_csv(out/"results/trial_label_variability.csv", index=False, float_format="%.17g")
    return checks


if __name__ == "__main__":
    build_trial_data()
