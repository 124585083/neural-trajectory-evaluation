from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SessionMetadata:
    session_id: str
    session_path: Path
    tiers: np.ndarray
    trial_ids: np.ndarray
    unit_ids: np.ndarray
    response_precision: np.ndarray

    def indices(self, tier: str) -> np.ndarray:
        return np.flatnonzero(self.tiers == tier)


@dataclass(frozen=True)
class OracleCondition:
    condition_id: str
    dataset_indices: tuple[int, ...]
    trial_ids: tuple[str, ...]


def load_session_metadata(session_path: str | Path) -> SessionMetadata:
    path = Path(session_path).resolve()
    tiers = np.load(path / "meta/trials/tiers.npy", allow_pickle=True)
    trial_ids = np.load(path / "meta/trials/trial_idx.npy", allow_pickle=True).astype(str)
    unit_ids = np.load(path / "meta/neurons/unit_ids.npy", allow_pickle=True)
    response_std = np.asarray(np.load(path / "meta/statistics/responses/all/std.npy"), dtype=np.float64)
    threshold = 0.01 * float(np.nanmean(response_std))
    precision = np.full_like(response_std, 1.0 / threshold)
    valid = response_std > threshold
    precision[valid] = 1.0 / response_std[valid]
    return SessionMetadata(path.name, path, tiers, trial_ids, unit_ids, precision)


def deterministic_neuron_order(metadata: SessionMetadata, seed: int) -> np.ndarray:
    """Return a train-independent nested order using stable unit-id hashes."""
    keys = []
    for index, unit_id in enumerate(metadata.unit_ids):
        digest = hashlib.sha256(f"{seed}:{unit_id}".encode()).digest()
        keys.append((digest, index))
    return np.asarray([index for _, index in sorted(keys)], dtype=np.int64)


def load_normalized_response(
    metadata: SessionMetadata,
    dataset_index: int,
    neuron_indices: np.ndarray,
    frame_start: int = 50,
    frame_stop: int = 300,
    response_precision: np.ndarray | None = None,
) -> np.ndarray:
    response = np.load(
        metadata.session_path / "data/responses" / f"{int(dataset_index)}.npy",
        mmap_mode="r",
    )
    if response.ndim != 2:
        raise ValueError(f"unexpected response shape {response.shape}")
    if response.shape[0] != metadata.unit_ids.size:
        raise AssertionError("neuron axis does not match metadata")
    if response.shape[1] < frame_stop:
        raise AssertionError(f"trial {dataset_index} has only {response.shape[1]} response frames")
    selected = np.asarray(response[neuron_indices, frame_start:frame_stop], dtype=np.float32)
    precision = metadata.response_precision if response_precision is None else np.asarray(response_precision)
    if precision.ndim == 1:
        selected_precision = precision if len(precision) == len(neuron_indices) else precision[neuron_indices]
        selected *= selected_precision[:, None].astype(np.float32)
    elif precision.ndim == 2:
        rows = np.arange(len(neuron_indices)) if precision.shape[0] == len(neuron_indices) else neuron_indices
        columns = (
            np.arange(frame_stop - frame_start)
            if precision.shape[1] == frame_stop - frame_start
            else np.arange(frame_start, frame_stop)
        )
        selected *= precision[np.ix_(rows, columns)].astype(np.float32)
    else:
        raise ValueError(f"unexpected response-statistics shape {precision.shape}")
    return selected.T.copy()


def load_response_trials(
    metadata: SessionMetadata,
    dataset_indices: Iterable[int],
    neuron_indices: np.ndarray,
    frame_start: int = 50,
    frame_stop: int = 300,
    response_precision: np.ndarray | None = None,
) -> np.ndarray:
    return np.stack(
        [
            load_normalized_response(
                metadata,
                index,
                neuron_indices,
                frame_start,
                frame_stop,
                response_precision,
            )
            for index in dataset_indices
        ]
    )


def _stimulus_signature(session_path: Path, dataset_index: int) -> np.ndarray:
    video = np.load(session_path / "data/videos" / f"{int(dataset_index)}.npy", mmap_mode="r")
    if video.ndim != 3 or video.shape[-1] < 300:
        raise ValueError(f"unexpected video shape {video.shape}")
    # Grayscale stimulus only: behavior must never enter the condition identity.
    signature = np.asarray(video[::4, ::4, :300:4], dtype=np.float64).reshape(-1)
    signature -= signature.mean()
    norm = np.linalg.norm(signature)
    if norm <= 1e-12:
        raise ValueError(f"constant stimulus in trial {dataset_index}")
    return signature / norm


def discover_oracle_conditions(
    metadata: SessionMetadata,
    similarity_threshold: float = 0.999,
) -> list[OracleCondition]:
    indices = metadata.indices("oracle")
    signatures = np.stack([_stimulus_signature(metadata.session_path, index) for index in indices])
    similarities = signatures @ signatures.T
    unseen = set(range(len(indices)))
    components: list[list[int]] = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        stack = [seed]
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = set(np.flatnonzero(similarities[current] >= similarity_threshold).tolist()) & unseen
            unseen -= neighbors
            stack.extend(sorted(neighbors, reverse=True))
        components.append(sorted(component))
    components.sort(key=lambda group: int(indices[group[0]]))
    conditions = []
    for condition_index, group in enumerate(components):
        group_indices = tuple(int(indices[item]) for item in group)
        conditions.append(
            OracleCondition(
                condition_id=f"movie_{condition_index:02d}",
                dataset_indices=group_indices,
                trial_ids=tuple(str(metadata.trial_ids[item]) for item in group_indices),
            )
        )
    return conditions


def load_oracle_conditions(
    metadata: SessionMetadata,
    conditions: list[OracleCondition],
    neuron_indices: np.ndarray,
    frame_start: int = 50,
    frame_stop: int = 300,
    response_precision: np.ndarray | None = None,
) -> list[np.ndarray]:
    return [
        load_response_trials(
            metadata,
            condition.dataset_indices,
            neuron_indices,
            frame_start,
            frame_stop,
            response_precision,
        )
        for condition in conditions
    ]


def compute_train_neuron_precision(
    metadata: SessionMetadata,
    dataset_indices: Iterable[int],
    neuron_indices: np.ndarray,
    frame_start: int = 50,
    frame_stop: int = 300,
) -> np.ndarray:
    """Compute one scale per neuron from real training samples only."""
    total = np.zeros(len(neuron_indices), dtype=np.float64)
    total_squared = np.zeros_like(total)
    count = np.zeros_like(total)
    for dataset_index in dataset_indices:
        response = np.load(
            metadata.session_path / "data/responses" / f"{int(dataset_index)}.npy",
            mmap_mode="r",
        )
        values = np.asarray(response[neuron_indices, frame_start:frame_stop], dtype=np.float64)
        finite = np.isfinite(values)
        safe = np.where(finite, values, 0.0)
        total += safe.sum(axis=1)
        total_squared += (safe * safe).sum(axis=1)
        count += finite.sum(axis=1)
    variance = (total_squared - total * total / np.maximum(count, 1.0)) / np.maximum(count - 1.0, 1.0)
    std = np.sqrt(np.clip(variance, 0.0, None))
    threshold = 0.01 * float(np.nanmean(std))
    std = np.where(std > threshold, std, threshold)
    precision = 1.0 / std
    if not np.isfinite(precision).all():
        raise ValueError("train-only response precision contains non-finite values")
    return precision


def split_train_calibration_indices(
    metadata: SessionMetadata,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = metadata.indices("train").copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    calibration_count = max(1, int(round(len(indices) * calibration_fraction)))
    calibration = np.sort(indices[:calibration_count])
    fit = np.sort(indices[calibration_count:])
    if not len(fit):
        raise ValueError("calibration split consumed all training trials")
    return fit, calibration


def load_behavior_features(
    metadata: SessionMetadata,
    dataset_indices: Iterable[int],
    frame_start: int = 50,
    frame_stop: int = 300,
) -> np.ndarray:
    """Trial-level covariates used only for condition classification.

    Median and robust temporal spread are computed for each provided behavior
    channel. Neural responses never enter these labels.
    """
    rows = []
    for dataset_index in dataset_indices:
        behavior = np.load(
            metadata.session_path / "data/behavior" / f"{int(dataset_index)}.npy",
            mmap_mode="r",
        )
        if behavior.ndim != 2 or behavior.shape[1] < frame_stop:
            raise ValueError(f"unexpected behavior shape {behavior.shape}")
        values = np.asarray(behavior[:, frame_start:frame_stop], dtype=np.float64)
        median = np.nanmedian(values, axis=1)
        spread = np.nanquantile(values, 0.75, axis=1) - np.nanquantile(values, 0.25, axis=1)
        rows.append(np.concatenate([median, spread]))
    result = np.asarray(rows)
    if not np.isfinite(result).all():
        raise ValueError("behavior classification features contain non-finite values")
    return result
