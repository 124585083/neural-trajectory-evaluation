from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from neuralpredictors.data.datasets import MovieFileTreeDataset
from neuralpredictors.data.transforms import (
    AddBehaviorAsChannels,
    ChangeChannelsOrder,
    CutVideos,
    ExpandChannels,
    NeuroNormalizer,
    ScaleInputs,
    Subsample,
    ToTensor,
)
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import resolve_project_path


@dataclass(frozen=True)
class SessionMetadata:
    session_key: str
    session_path: Path
    neuron_ids: np.ndarray
    cell_motor_coordinates: np.ndarray
    tier_counts: dict[str, int]
    input_shape: tuple[int, int, int]
    behavior_dim: int
    pupil_dim: int


class DeterministicEpochSampler(Sampler[int]):
    """A reproducible permutation that can be replayed for both models."""

    def __init__(self, size: int, seed: int, shuffle: bool) -> None:
        self.size = size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            return iter(range(self.size))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


class TrialWindowDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Applies one deterministic, aligned temporal window to every modality."""

    def __init__(
        self,
        base: MovieFileTreeDataset,
        trial_indices: Sequence[int],
        tier: str,
        frames: int | None,
        seed: int,
        random_window: bool,
        session_key: str,
    ) -> None:
        self.base = base
        self.trial_indices = np.asarray(trial_indices, dtype=np.int64)
        self.tier = tier
        self.frames = frames
        self.seed = int(seed)
        self.random_window = random_window
        self.session_key = session_key
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.trial_indices)

    def _offset(self, trial_index: int, time_points: int) -> int:
        if self.frames is None:
            return 0
        if time_points < self.frames:
            raise ValueError(f"trial {trial_index} has {time_points} frames, fewer than requested {self.frames}")
        if not self.random_window or time_points == self.frames:
            return 0
        # SeedSequence gives a stable mapping independent of process-global RNG state.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, trial_index]))
        # Match the official Subsequence high-exclusive convention.
        return int(rng.integers(0, time_points - self.frames))

    def __getitem__(self, local_index: int) -> dict[str, torch.Tensor | str]:
        trial_index = int(self.trial_indices[local_index])
        sample = self.base[trial_index]
        time_points = int(sample.responses.shape[-1])
        offset = self._offset(trial_index, time_points)
        stop = time_points if self.frames is None else offset + self.frames

        video = sample.videos[:, offset:stop, :, :].contiguous()
        responses = sample.responses[:, offset:stop].contiguous()
        behavior = sample.behavior[:, offset:stop].contiguous()
        pupil_center = sample.pupil_center[:, offset:stop].contiguous()
        selected_time = video.shape[1]
        return {
            "video": video,
            "behavior": behavior,
            "responses": responses,
            "pupil_center": pupil_center,
            "mask": torch.ones(selected_time, dtype=torch.bool),
            "frame_index": torch.arange(offset, stop, dtype=torch.int64),
            "trial_index": torch.tensor(trial_index, dtype=torch.int64),
            "window_offset": torch.tensor(offset, dtype=torch.int64),
            "session_key": self.session_key,
        }


@dataclass
class LoaderBundle:
    loader: DataLoader
    dataset: TrialWindowDataset
    sampler: DeterministicEpochSampler

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        self.sampler.set_epoch(epoch)


class DynamicSensoriumDataModule:
    """One authoritative Sensorium loader shared by dynamic and static models.

    The transform stack is the official Sensorium 2023 stack, except that the
    random Subsequence transform is replaced by a replayable aligned window.
    The public tensors are video [B,C,T,H,W], behavior [B,D,T], responses
    [B,N,T], plus pupil/mask/alignment metadata.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        data_cfg = config["data"]
        self.seed = int(config["project"]["seed"])
        self.frames = int(data_cfg["frames"])
        self.physical_batch_size = int(data_cfg["physical_batch_size"])
        self.num_workers = int(data_cfg["num_workers"])
        self.pin_memory = bool(data_cfg["pin_memory"])
        root = resolve_project_path(config, data_cfg["root"])
        self.session_path = (root / data_cfg["session"]).resolve()
        if not self.session_path.is_dir():
            raise FileNotFoundError(f"Sensorium session not found: {self.session_path}")
        self.session_key = self.session_path.name

        self.base_dataset = MovieFileTreeDataset(
            str(self.session_path), "videos", "responses", "behavior", "pupil_center"
        )
        neuron_indices = np.arange(len(self.base_dataset.neurons.cell_motor_coordinates))
        transforms = [
            NeuroNormalizer(self.base_dataset, stats_source=data_cfg["normalization_stats"], in_name="videos"),
            Subsample(neuron_indices, target_index=0),
            CutVideos(
                max_frame=None,
                frame_axis={key: -1 for key in ("videos", "responses", "behavior", "pupil_center")},
                target_groups=["videos", "responses", "behavior", "pupil_center"],
            ),
            ChangeChannelsOrder((2, 0, 1), in_name="videos"),
            ChangeChannelsOrder((1, 0), in_name="responses"),
            ChangeChannelsOrder((1, 0), in_name="behavior"),
            ChangeChannelsOrder((1, 0), in_name="pupil_center"),
            ChangeChannelsOrder((1, 0), in_name="responses"),
            ChangeChannelsOrder((1, 0), in_name="behavior"),
            ChangeChannelsOrder((1, 0), in_name="pupil_center"),
            ExpandChannels("videos"),
            AddBehaviorAsChannels("videos"),
            ScaleInputs(scale=float(data_cfg["scale"]), in_name="videos", channel_axis=-1),
            ToTensor(cuda=False),
        ]
        self.base_dataset.transforms.extend(transforms)

        tiers = np.asarray(self.base_dataset.trial_info.tiers).astype(str)
        self.tier_indices = {
            tier: np.flatnonzero(tiers == tier).astype(np.int64)
            for tier in sorted(set(tiers))
            if tier != "none"
        }
        first = self.base_dataset[int(self.tier_indices["train"][0])]
        neuron_ids = np.asarray(self.base_dataset.neurons.unit_ids)
        coordinates = np.asarray(self.base_dataset.neurons.cell_motor_coordinates)
        self.metadata = SessionMetadata(
            session_key=self.session_key,
            session_path=self.session_path,
            neuron_ids=neuron_ids,
            cell_motor_coordinates=coordinates,
            tier_counts={key: len(value) for key, value in self.tier_indices.items()},
            input_shape=(int(first.videos.shape[0]), int(first.videos.shape[-2]), int(first.videos.shape[-1])),
            behavior_dim=int(first.behavior.shape[0]),
            pupil_dim=int(first.pupil_center.shape[0]),
        )

    def make_loader(
        self,
        tier: str,
        *,
        frames: int | None = None,
        batch_size: int | None = None,
        shuffle: bool | None = None,
    ) -> LoaderBundle:
        if tier not in self.tier_indices:
            raise KeyError(f"unknown tier {tier!r}; available: {sorted(self.tier_indices)}")
        if frames is None:
            frames = self.frames
        if shuffle is None:
            shuffle = tier == "train"
        dataset = TrialWindowDataset(
            self.base_dataset,
            self.tier_indices[tier],
            tier=tier,
            frames=frames,
            seed=self.seed,
            random_window=tier == "train",
            session_key=self.session_key,
        )
        sampler = DeterministicEpochSampler(len(dataset), seed=self.seed, shuffle=shuffle)
        loader = DataLoader(
            dataset,
            batch_size=batch_size or self.physical_batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )
        return LoaderBundle(loader=loader, dataset=dataset, sampler=sampler)

    @property
    def n_neurons(self) -> int:
        return len(self.metadata.neuron_ids)

    def reference_mean_activity(self) -> torch.Tensor:
        bundle = self.make_loader("train", frames=self.frames, shuffle=True)
        bundle.set_epoch(0)
        batch = next(iter(bundle.loader))
        return batch["responses"].mean(dim=(0, 2))

