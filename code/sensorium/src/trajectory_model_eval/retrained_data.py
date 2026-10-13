"""Grouped Part 3 splits and normalization fitted only on the new training trials.

Original tiers remain provenance fields. No original dataset or statistics file
is edited, and the training adapter never constructs a loader for oracle test.
"""
from __future__ import annotations

from collections import Counter, defaultdict, namedtuple
import copy
import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .unified_contract import canonical_digest, file_sha256, manifest_digest

LOGGER = logging.getLogger(__name__)
MODALITIES = ("videos", "responses", "behavior", "pupil_center")
Batch = namedtuple("Batch", MODALITIES)


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_json(value):
    return json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else value


def _overlap_edges(data_root, representatives, minimum_frames=60, similarity=.999):
    """Find shifted shared clips using stimulus-only frame landmarks.

    Two seeded 32-bit random-hyperplane tables propose matching frames. Every
    proposed movie/lag pair is checked on its full common support, requiring at
    least 60 consecutive frames and grayscale cosine >= .999. The search is
    high-specificity, not a proof that all upstream source clips are different.
    """
    root = Path(data_root)
    movies = []
    for row in representatives:
        video_path = root / row["source_files"]["videos"]
        if file_sha256(video_path) != row["source_sha256"]["videos"]:
            raise ValueError("Stimulus file changed since the source inventory")
        video = np.load(video_path, mmap_mode="r", allow_pickle=False)
        movies.append(np.asarray(video[::4, ::4, :300].transpose(2, 0, 1), dtype=np.float32).reshape(300, -1))
    if not movies:
        return [], {"candidate_pairs": 0, "verified_overlaps": 0}
    values = np.stack(movies)
    frames = values.reshape(-1, values.shape[-1])
    centered = frames - frames.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norms, 1e-12)
    rng = np.random.default_rng(42)
    pairs = set()
    weights = (np.uint32(1) << np.arange(32, dtype=np.uint32))
    for table in range(2):
        projection = rng.standard_normal((normalized.shape[1], 32)).astype(np.float32)
        keys = ((normalized @ projection > 0).astype(np.uint32) * weights).sum(axis=1, dtype=np.uint32)
        lookup = defaultdict(list)
        for flat, key in enumerate(keys):
            if norms[flat, 0] > 1e-10:
                lookup[int(key)].append(flat)
        for movie in range(len(values)):
            for anchor in range(0, 300, 30):
                flat = movie * 300 + anchor
                for peer in lookup.get(int(keys[flat]), []):
                    other, other_frame = divmod(peer, 300)
                    if other <= movie:
                        continue
                    lag = other_frame - anchor
                    if 300 - abs(lag) >= minimum_frames:
                        pairs.add((movie, other, lag))
        LOGGER.info("Overlap landmark table %d: %d proposed movie/lag pairs", table + 1, len(pairs))
    edges = []
    accepted = set()
    for left, right, lag in sorted(pairs):
        if (left, right) in accepted:
            continue
        first_left, first_right = max(0, -lag), max(0, lag)
        count = 300 - abs(lag)
        a = values[left, first_left:first_left + count].reshape(-1).astype(np.float64)
        b = values[right, first_right:first_right + count].reshape(-1).astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        score = float(a @ b / denominator) if denominator else -1.
        if score >= similarity:
            accepted.add((left, right))
            edges.append({"left_movie_id": representatives[left]["movie_id"],
                          "right_movie_id": representatives[right]["movie_id"],
                          "left_session": representatives[left]["session"],
                          "right_session": representatives[right]["session"],
                          "lag_frames": lag, "common_frames": count, "similarity": score})
    return edges, {"candidate_pairs": len(pairs), "verified_overlaps": len(edges),
                   "minimum_common_frames": minimum_frames, "similarity_threshold": similarity,
                   "frame_landmark_stride": 30, "hash_tables": 2, "hash_bits": 32,
                   "proposal_seed": 42, "all_integer_lags_possible": True,
                   "method": "Frame-landmark proposals followed by full-overlap grayscale verification",
                   "limitation": "Approximate candidate search; shorter overlaps and unavailable upstream source identities remain unresolved"}


def build_split(data_root, prior_manifest, output_dir, seed=42, validation_fraction=.2,
                check_overlaps=True, config=None):
    """Freeze grouped roles from audited source records, without neural scoring."""
    prior = _load_json(prior_manifest)
    root, output = Path(data_root), Path(output_dir)
    rows = copy.deepcopy(prior["trials"])
    eligible = [r for r in rows if r["source_tier"] in ("train", "oracle")]
    if any(not r["eligible_labeled_trial"] for r in eligible):
        raise ValueError("A requested train/oracle trial lacks valid complete labeled support")
    for source in prior["metadata_files"]:
        if file_sha256(root / source["path"]) != source["sha256"]:
            raise ValueError("Source trial or neuron metadata changed since inventory")
    unique = {}
    for row in eligible:
        unique.setdefault(row["movie_id"], row)
    edges, overlap_audit = _overlap_edges(root, list(unique.values())) if check_overlaps else ([], {
        "status": "not_run", "limitation": "Synthetic fixture or explicitly omitted overlap search"})
    parent = {key: key for key in unique}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for edge in edges:
        a, b = find(edge["left_movie_id"]), find(edge["right_movie_id"])
        parent[max(a, b)] = min(a, b)
    test_groups = {find(r["movie_id"]) for r in eligible if r["source_tier"] == "oracle"}
    train_groups = sorted({find(r["movie_id"]) for r in eligible if r["source_tier"] == "train"} - test_groups)
    shuffled = np.random.RandomState(seed).permutation(train_groups)
    n_validation = max(1, int(np.ceil(len(train_groups) * validation_fraction)))
    validation_groups = set(shuffled[:n_validation])
    for row in rows:
        row["original_tier"] = row["source_tier"]
        row["condition_id"] = row["movie_id"]
        row["group_id"] = find(row["movie_id"]) if row["movie_id"] in parent else row["movie_id"]
        row["support"] = {"input_frame_start": 0, "input_frame_stop": 300,
                          "evaluation_frame_start": prior["time"]["frame_start"],
                          "evaluation_frame_stop": prior["time"]["frame_stop"]}
        if row["source_tier"] == "oracle":
            row["role"], row["exclusion_reason"] = "test", None
        elif row["source_tier"] != "train":
            row["role"], row["exclusion_reason"] = "excluded", "hidden_test_zero_response_placeholder"
        elif row["group_id"] in test_groups:
            row["role"], row["exclusion_reason"] = "excluded", "identified_stimulus_overlap_with_oracle_test"
        else:
            row["role"] = "validation" if row["group_id"] in validation_groups else "train"
            row["exclusion_reason"] = None
    neurons = copy.deepcopy(prior["session_neurons"])
    for session in prior["sessions"]:
        path = root / session / "meta/neurons/unit_ids.npy"
        ids = np.load(path, allow_pickle=False).astype(str).tolist()
        neurons[session]["all_ids"] = ids
        neurons[session]["all_indices"] = list(range(len(ids)))
        neurons[session]["source_sha256"] = file_sha256(path)
        coordinates = root / session / "meta/neurons/cell_motor_coordinates.npy"
        neurons[session]["cell_motor_coordinates_sha256"] = file_sha256(coordinates)
    manifest = {
        "schema_version": "sensorium_part3_retrained_v1", "protocol": "retrained_v1",
        "config_sha256": canonical_digest(config) if config is not None else None,
        "source_inventory_sha256": prior["split_sha256"],
        "session": prior["session"], "sessions": prior["sessions"], "time": prior["time"],
        "neurons": copy.deepcopy(prior["neurons"]), "session_neurons": neurons,
        "evaluation_scope": "retrained_holdout_evaluation_on_previously_explored_dataset",
        "metadata_files": prior["metadata_files"], "trials": rows,
        "split_rule": {"seed": int(seed), "validation_fraction": validation_fraction,
                       "unit": "connected stimulus group shared across sessions",
                       "algorithm": "RandomState(seed).permutation(sorted development group IDs); first ceil(0.2*N) validation",
                       "test": "all complete original oracle repeats", "test_conflicts": "quarantine development trials"},
        "movie_identity": {**prior["movie_identity"], "overlap_audit": overlap_audit,
                           "overlap_edges": edges, "upstream_source_clip_ids": "not available"},
    }
    manifest["split_sha256"] = manifest_digest(manifest)
    validate_split(manifest)
    _save_json(output / "split_manifest.json", manifest)
    summaries = []
    for session in manifest["sessions"]:
        for role in ("train", "validation", "test", "excluded"):
            selected = [r for r in rows if r["session"] == session and r["role"] == role]
            counts = Counter(r["condition_id"] for r in selected)
            summaries.append({"session": session, "role": role, "n_trials": len(selected),
                              "n_conditions": len(counts), "n_groups": len({r["group_id"] for r in selected}),
                              "repeat_counts": json.dumps(sorted(counts.values())) if role == "test" else "",
                              "split_sha256": manifest["split_sha256"]})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "split_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return manifest


def validate_split(manifest):
    if manifest["schema_version"] != "sensorium_part3_retrained_v1":
        raise ValueError("Wrong split protocol")
    if manifest_digest(manifest) != manifest["split_sha256"]:
        raise ValueError("Split digest mismatch")
    seen, groups = set(), defaultdict(set)
    for row in manifest["trials"]:
        key = row["session"], row["dataset_index"]
        if key in seen:
            raise ValueError("A trial appears more than once")
        seen.add(key)
        if row["role"] != "excluded":
            groups[row["group_id"]].add(row["role"])
        if row["role"] in ("train", "validation") and row["source_tier"] != "train":
            raise ValueError("Development loader contains an original non-train trial")
        if row["role"] == "test" and row["source_tier"] != "oracle":
            raise ValueError("Test does not preserve original oracle identity")
    if any(len(roles) > 1 for roles in groups.values()):
        raise ValueError("A recognized stimulus group crosses roles")
    return True


def role_rows(manifest, session, role):
    return [r for r in manifest["trials"] if r["session"] == session and r["role"] == role]


def fit_preprocessing(data_root, manifest, output_dir, verify_source_hashes=True):
    """Compute per-array-element population moments across new train trials.

    The original supplied statistics retain neuron/pixel/channel and time axes.
    These replacement moments do likewise, over the common finite 300 frames.
    Responses are scaled without centering or clipping. Small negative source
    values from upstream numerical processing remain unchanged in sign.
    """
    manifest = _load_json(manifest)
    validate_split(manifest)
    root, output = Path(data_root), Path(output_dir)
    record = {"schema_version": "sensorium_train_only_preprocessing_v1", "protocol": "retrained_v1",
              "split_sha256": manifest["split_sha256"], "frame_stop": 300,
              "formula": {"moments": "population mean/std across Part 3 train trials at each original array element; ddof=0",
                          "videos": "(x-train_mean)/train_std", "behavior": "x/train_std; no mean subtraction",
                          "pupil_center": "(x-train_mean)/train_std",
                          "responses": "x/max(train_std,0.01*mean(train_std)); no mean subtraction or clipping; retain source numerical negatives",
                          "zero_nonresponse_std": "use 1 where standard deviation <= float32 epsilon"},
              "statistics_source": "actual Part 3 train trial files only; never dataset-supplied meta/statistics",
              "initialization": "official builder samples only the passed Part 3 train loaders for mean activity; anatomical coordinates retain source order",
              "sessions": {}}
    for session in manifest["sessions"]:
        rows = role_rows(manifest, session, "train")
        if len(rows) < 2:
            raise ValueError("At least two training trials are needed for fixed normalization")
        means, m2 = {}, {}
        response_minimum, response_negative_values, response_values = float("inf"), 0, 0
        LOGGER.info("Fitting train-only normalization: %s (%d trials)", session, len(rows))
        for count, row in enumerate(rows, 1):
            for modality in MODALITIES:
                path = root / row["source_files"][modality]
                if verify_source_hashes and file_sha256(path) != row["source_sha256"][modality]:
                    raise ValueError(f"Training source changed: {session} {row['dataset_index']} {modality}")
                value = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False)[..., :300], dtype=np.float64)
                if value.shape[-1] != 300 or not np.isfinite(value).all():
                    raise ValueError("Nonfinite or incomplete training normalization support")
                if modality == "responses":
                    response_minimum = min(response_minimum, float(np.min(value)))
                    response_negative_values += int(np.count_nonzero(value < 0))
                    response_values += int(value.size)
                if count == 1:
                    means[modality] = value.copy()
                    m2[modality] = np.zeros_like(value)
                else:
                    delta = value - means[modality]
                    means[modality] += delta / count
                    m2[modality] += delta * (value - means[modality])
            if count % 50 == 0:
                LOGGER.info("Normalization %s: %d/%d train trials", session, count, len(rows))
        arrays, zero_counts = {}, {}
        for modality in MODALITIES:
            std = np.sqrt(np.maximum(m2[modality] / len(rows), 0))
            arrays[f"{modality}_mean"] = means[modality].astype(np.float32)
            if modality != "responses":
                zero = std <= np.finfo(np.float32).eps
                zero_counts[modality] = int(np.count_nonzero(zero))
                std[zero] = 1.
            arrays[f"{modality}_std"] = std.astype(np.float32)
        response_std = arrays["responses_std"]
        threshold = .01 * float(response_std.mean(dtype=np.float64))
        if threshold <= 0:
            raise ValueError("Training responses have no usable scale")
        arrays["response_precision"] = (1. / np.maximum(response_std, threshold)).astype(np.float32)
        arrays["response_threshold"] = np.asarray(threshold)
        arrays["train_dataset_indices"] = np.asarray([r["dataset_index"] for r in rows], dtype=np.int64)
        arrays["neuron_ids"] = np.asarray(manifest["session_neurons"][session]["all_ids"])
        path = output / "normalization" / f"{session}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        record["sessions"][session] = {
            "file": f"normalization/{session}.npz", "sha256": file_sha256(path),
            "n_train_trials": len(rows), "train_dataset_indices": arrays["train_dataset_indices"].tolist(),
            "train_trial_ids": [r["trial_id"] for r in rows],
            "source_sha256": canonical_digest([r["source_sha256"] for r in rows]),
            "neuron_ids_sha256": canonical_digest(manifest["session_neurons"][session]["all_ids"]),
            "zero_std_counts": zero_counts, "response_threshold": threshold,
            "source_response_minimum": response_minimum,
            "source_response_negative_values": response_negative_values,
            "source_response_values": response_values,
            "negative_response_policy": "Retain original values and sign; no clipping or mean subtraction",
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        }
    record["code_sha256"] = file_sha256(__file__)
    record["preprocessing_sha256"] = canonical_digest(record)
    _save_json(output / "preprocessing.json", record)
    return record


def load_preprocessing(preprocessing, session, artifact_root=None):
    if isinstance(preprocessing, (str, Path)):
        artifact_root = Path(preprocessing).parent
    record = _load_json(preprocessing)
    identity = dict(record)
    actual = identity.pop("preprocessing_sha256")
    if canonical_digest(identity) != actual:
        raise ValueError("Preprocessing record digest mismatch")
    if artifact_root is None:
        raise ValueError("artifact_root is required for an in-memory preprocessing record")
    path = Path(artifact_root) / record["sessions"][session]["file"]
    if file_sha256(path) != record["sessions"][session]["sha256"]:
        raise ValueError("Normalization artifact digest mismatch")
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


class RetrainedDataset:
    """Subset-local dataset with the official tensor layout and new fixed stats."""

    data_keys = list(MODALITIES)

    def __init__(self, data_root, manifest, preprocessing, session, role,
                 artifact_root=None, frames=80, offset=-1, to_cut=True, cuda=False):
        self.manifest = _load_json(manifest)
        validate_split(self.manifest)
        record = _load_json(preprocessing)
        if record["split_sha256"] != self.manifest["split_sha256"]:
            raise ValueError("Normalization and loader split identities differ")
        self.root, self.session, self.role = Path(data_root), session, role
        self.original_dataset_length = sum(r["session"] == session for r in self.manifest["trials"])
        self.rows = role_rows(self.manifest, session, role)
        if not self.rows or role == "excluded":
            raise ValueError("Requested loader has no eligible observations")
        self.dataset_indices = [r["dataset_index"] for r in self.rows]
        self.trial_ids = [r["trial_id"] for r in self.rows]
        self.frames, self.offset, self.to_cut, self.cuda = frames, offset, to_cut, cuda
        self.statistics_arrays = load_preprocessing(preprocessing, session, artifact_root)
        if record["sessions"][session]["train_dataset_indices"] != [r["dataset_index"] for r in role_rows(self.manifest, session, "train")]:
            raise ValueError("Normalization trial identities differ from train role")
        path = self.root / session / "meta/neurons"
        self.neurons = SimpleNamespace(unit_ids=np.load(path / "unit_ids.npy", allow_pickle=False),
                                       cell_motor_coordinates=np.load(path / "cell_motor_coordinates.npy", allow_pickle=False))
        if self.neurons.unit_ids.astype(str).tolist() != self.manifest["session_neurons"][session]["all_ids"]:
            raise ValueError("Neuron order differs from frozen manifest")
        if file_sha256(path / "cell_motor_coordinates.npy") != self.manifest["session_neurons"][session]["cell_motor_coordinates_sha256"]:
            raise ValueError("Readout anatomical initialization source changed")
        self.trial_info = SimpleNamespace(tiers=np.asarray([role] * len(self.rows)), trial_idx=np.asarray(self.trial_ids))
        self.preprocessing_sha256 = record["preprocessing_sha256"]
        self.split_sha256 = self.manifest["split_sha256"]

    @property
    def statistics(self):
        raise RuntimeError("Dataset-supplied statistics are forbidden in retrained_v1")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        import torch
        row = self.rows[item]
        arrays = {key: np.asarray(np.load(self.root / row["source_files"][key], mmap_mode="r", allow_pickle=False)[..., :300], dtype=np.float32)
                  for key in MODALITIES}
        if any(value.shape[-1] != 300 or not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Loader input has invalid 300-frame support")
        stats = self.statistics_arrays
        arrays["videos"] = (arrays["videos"] - stats["videos_mean"]) / stats["videos_std"]
        arrays["responses"] = arrays["responses"] * stats["response_precision"]
        arrays["behavior"] = arrays["behavior"] / stats["behavior_std"]
        arrays["pupil_center"] = (arrays["pupil_center"] - stats["pupil_center_mean"]) / stats["pupil_center_std"]
        if self.to_cut:
            start = np.random.randint(0, 300 - self.frames) if self.offset < 0 else self.offset
            if start < 0 or start + self.frames > 300:
                raise ValueError("Snippet falls outside its parent trial")
            arrays = {key: value[..., start:start + self.frames] for key, value in arrays.items()}
        video = arrays["videos"].transpose(2, 0, 1)[None]
        behavior_channels = np.broadcast_to(arrays["behavior"][:, :, None, None],
                                             (arrays["behavior"].shape[0], video.shape[1], video.shape[2], video.shape[3]))
        arrays["videos"] = np.concatenate((video, behavior_channels), axis=0)
        values = [torch.from_numpy(np.ascontiguousarray(arrays[key], dtype=np.float32)) for key in MODALITIES]
        if self.cuda:
            values = [value.cuda() for value in values]
        return Batch(*values)


def create_dataloaders(data_root, manifest, preprocessing, artifact_root=None,
                       batch_size=8, frames=80, offset=-1, to_cut=True, cuda=False,
                       sessions=None):
    """The historical 'oracle' closure key is new validation, never oracle test."""
    import torch
    manifest = _load_json(manifest)
    result = {"train": {}, "oracle": {}}
    for session in sessions or manifest["sessions"]:
        for role, key in (("train", "train"), ("validation", "oracle")):
            dataset = RetrainedDataset(data_root, manifest, preprocessing, session, role,
                                       artifact_root, frames, offset, to_cut, cuda)
            result[key][session] = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                              shuffle=role == "train", num_workers=0)
    return result


def create_role_loader(data_root, manifest, preprocessing, session, role, artifact_root=None,
                       batch_size=1, cuda=False):
    """Explicit full-trial loader for frozen validation calibration or test export."""
    import torch
    dataset = RetrainedDataset(data_root, manifest, preprocessing, session, role,
                               artifact_root=artifact_root, to_cut=False, cuda=cuda)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
