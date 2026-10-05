# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os
import random
from functools import cached_property

import numpy as np
import torch
from rich.console import Console
from rich.tree import Tree
from torch.utils.data import IterableDataset

from anemoi.models.distributed.balanced_partition import get_balanced_partition_range
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.training.data.data_reader import BaseAnemoiReader
from anemoi.training.data.usable_indices import compute_valid_anchors
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.training.utils.time_indices import offset_time_indices
from anemoi.utils.dates import frequency_to_timedelta

LOGGER = logging.getLogger(__name__)


class MultiDataset(IterableDataset):
    """Multi-dataset wrapper that returns synchronized samples from multiple data readers."""

    def __init__(
        self,
        data_readers: dict[str, BaseAnemoiReader],
        relative_date_indices: dict[str, TimeIndices],
        shuffle: bool = True,
        label: str = "multi",
        epoch: int = 0,
        rollout: int = 1,
        fake_dataloading: bool = False,
        dataset_offsets: dict[str, list[datetime.timedelta]] | None = None,
        date_filter: dict | None = None,
        epoch_sample: dict | None = None,
    ) -> None:
        """Initialize multi-dataset with synchronized data readers.

        Parameters
        ----------
        data_readers : dict[str, BaseAnemoiReader]
            Dictionary mapping dataset names to their data_readers
            Format: {"dataset_a": data_reader_a, "dataset_b": data_reader_b, ...}
        relative_date_indices : dict[str, TimeIndices]
            Precomputed relative date indices for each data reader
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "multi"
        epoch : int, optional
            Epoch used for deterministic epoch-dependent shuffling, by default 0
        rollout : int, optional
            Rollout length represented by the loaded relative date indices, by default 1
        fake_dataloading : bool, optional
            Load one real sample and reuse it for subsequent accesses, by default False
        """
        self.data_readers = data_readers
        self.label = label
        self.shuffle = shuffle
        self.dataset_names = list(data_readers.keys())
        self.epoch = epoch
        self.rollout = rollout
        self.fake_dataloading = fake_dataloading
        self.dataset_offsets = dataset_offsets
        self.date_filter = date_filter
        self.epoch_sample = epoch_sample
        if self.fake_dataloading:
            LOGGER.info("Using fake dataloading")

        # Guard against mixing single-sequence (NativeGridDataset, global time axis)
        # with multi-sequence (TrajectoryDataset, init x step axes).  The anchor
        # intersection would silently keep only sequence-0 samples and produce
        # semantically meaningless alignment between the two encoders.
        single_seq = [n for n, ds in data_readers.items() if ds.num_sequences == 1]
        multi_seq = [n for n, ds in data_readers.items() if ds.num_sequences > 1]
        if single_seq and multi_seq:
            msg = (
                "Currently mixing single-sequence datasets (global time axis) with "
                "Trajectory datasets (init x step axes) in the same MultiDataset is unsupported. "
                f"Single-sequence: {single_seq}. Trajectory: {multi_seq}. "
            )
            raise ValueError(msg)

        # Compute valid (sequence, position) anchors and a flat index over them
        # that the shuffle/shard logic operates on.
        if dataset_offsets is None:
            self.anchors = compute_valid_anchors(self.data_readers, relative_date_indices)
        else:
            if multi_seq:
                raise ValueError("Explicit dataset offsets currently require single-sequence readers")
            self.anchor_dataset_name = min(data_readers, key=lambda name: data_readers[name].frequency)
            reader = data_readers[self.anchor_dataset_name]
            offsets = dataset_offsets[self.anchor_dataset_name]
            if any(offset % reader.frequency for offset in offsets):
                raise ValueError("Anchor dataset offsets must be multiples of its native frequency")
            self.anchors = reader.compute_anchors([int(offset // reader.frequency) for offset in offsets])
            earliest_offset = min(offset for values in dataset_offsets.values() for offset in values)
            minimum_position = max(0, int(-earliest_offset // reader.frequency))
            self.anchors = self.anchors[self.anchors[:, 1] >= minimum_position]
            self.dataset_dates_ns = {
                name: np.asarray(reader.dates).astype("datetime64[ns]").astype(np.int64)
                for name, reader in data_readers.items()
            }
        self.valid_date_indices = np.arange(len(self.anchors), dtype=np.int64)
        if date_filter is not None:
            self.valid_date_indices = self._select_dates(self.valid_date_indices, date_filter)

        # Normalize the date indices to use slices where possible.
        self.relative_date_indices = {
            name: normalize_time_indices(indices) for name, indices in relative_date_indices.items()
        }

        self._lazy_init_model_and_reader_group_info()

    def set_epoch(
        self,
        epoch: int,
        *,
        rollout: int | None = None,
        relative_date_indices: dict[str, TimeIndices] | None = None,
    ) -> None:
        """Set epoch-dependent sampling state before DataLoader workers are launched."""
        self.epoch = epoch
        if rollout is not None:
            self.rollout = rollout
        if relative_date_indices is None:
            return

        # Recompute valid (sequence, position) anchors for the updated rollout.
        self.anchors = compute_valid_anchors(self.data_readers, relative_date_indices)
        self.valid_date_indices = np.arange(len(self.anchors), dtype=np.int64)

        # Normalize the date indices to use slices where possible.
        self.relative_date_indices = {
            name: normalize_time_indices(indices) for name, indices in relative_date_indices.items()
        }

    def _select_dates(self, indices: np.ndarray, config: dict) -> np.ndarray:
        """Filter forecast initialisation times against explicit event windows."""
        name = getattr(self, "anchor_dataset_name", config.get("dataset", self.dataset_names[0]))
        dates = np.asarray(self.data_readers[name].dates).astype("datetime64[ns]").astype(np.int64)
        dates = dates[self.anchors[indices, 1]]
        shift = int(frequency_to_timedelta(config.get("time_offset", "0s")).total_seconds() * 1e9)
        windows = []
        path = config.get("windows_path") or config.get("timestamps_path")
        if path is None:
            raise ValueError("Date filter requires windows_path or timestamps_path")
        with open(path) as source:
            for line in source:
                fields = line.strip().replace(",", " ").split()
                if not fields or fields[0].startswith("#"):
                    continue
                try:
                    start = np.datetime64(fields[0], "ns").astype(np.int64) + shift
                    end = np.datetime64(fields[1], "ns").astype(np.int64) + shift if config.get("windows_path") else start
                except (ValueError, IndexError):
                    continue
                radius = int(float(config.get("radius_minutes", 0.0)) * 60e9)
                windows.append((start - radius, end + radius))
        selected = np.zeros(len(indices), dtype=bool)
        for start, end in windows:
            inside = np.flatnonzero((dates >= start) & (dates <= end))
            if config.get("selection", "all") == "first_per_window":
                inside = inside[:1]
            selected[inside] = True
        if not selected.any():
            raise ValueError(f"Date filter {path!r} retained no forecast initialisations")
        return indices[selected]

    def _epoch_indices(self) -> np.ndarray:
        """Keep every mandatory anchor, then oversample or fill the requested epoch."""
        if self.epoch_sample is None:
            indices = self.valid_date_indices
        else:
            config = self.epoch_sample
            size = int(config["size"])
            mandatory_filter = config.get("mandatory_date_filter")
            mandatory = self._select_dates(self.valid_date_indices, mandatory_filter) if mandatory_filter else np.array([], dtype=np.int64)
            fraction = config.get("mandatory_fraction")
            count = len(mandatory) if fraction is None else int(round(size * float(fraction)))
            if not len(mandatory) <= count <= size:
                raise ValueError("Epoch sample must retain every mandatory anchor within its requested size")
            if count > len(mandatory):
                if not config.get("mandatory_with_replacement", False) or not len(mandatory):
                    raise ValueError("Repeated mandatory sampling requires a nonempty pool and mandatory_with_replacement")
                repeated = self.rng.choice(mandatory, size=count - len(mandatory), replace=True)
                mandatory = np.concatenate((mandatory, repeated))
            other = np.setdiff1d(self.valid_date_indices, mandatory)
            random_indices = self.rng.choice(other, size=size - count, replace=False)
            indices = np.concatenate((mandatory, random_indices))
        return self.rng.choice(indices, size=len(indices), replace=False) if self.shuffle else indices

    def _lazy_init_model_and_reader_group_info(self) -> None:
        """Lazy initialize model and reader group info."""
        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_id = 0
        self.global_rank = 0

        self.reader_group_rank = 0
        self.reader_group_size = 1

        self.sample_comm_num_groups = 1  # groups that work on the same sample / batch
        self.sample_comm_group_id = 0

        self.ens_comm_group_rank = 0
        self.ens_comm_num_groups = 1
        self.ens_comm_group_id = 0

        self.shard_sizes = None

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None

    def _collect(self, attr_name: str) -> dict:
        """Helper method to collect attributes from all data readers."""
        return {name: getattr(dataset, attr_name) for name, dataset in self.data_readers.items()}

    @cached_property
    def statistics(self) -> dict[str, dict]:
        """Return combined statistics from all data readers."""
        return self._collect("statistics")

    @cached_property
    def metadata(self) -> dict[str, dict]:
        """Return combined metadata from all data readers."""
        return self._collect("metadata")

    @cached_property
    def supporting_arrays(self) -> dict[str, dict]:
        """Return combined supporting arrays from all data readers."""
        return self._collect("supporting_arrays")

    @cached_property
    def variables(self) -> dict[str, list[str]]:
        """Return combined variables from all data readers."""
        return self._collect("variables")

    @property
    def data(self) -> dict:
        """Return data from all data readers as dictionary."""
        return self._collect("data")

    @cached_property
    def name_to_index(self) -> dict[str, dict]:
        """Return combined name_to_index mapping from all data readers."""
        return self._collect("name_to_index")

    @cached_property
    def resolution(self) -> dict[str, str]:
        """Return combined resolution from all data readers."""
        return self._collect("resolution")

    @cached_property
    def frequency(self) -> datetime.timedelta:
        """Return combined frequency from all data readers."""
        freqs = self._collect("frequency")
        if self.dataset_offsets is not None:
            return min(freqs.values())
        freq_ref = None
        for name, freq in freqs.items():
            if freq_ref is None:
                freq_ref = freq
            assert freq == freq_ref, f"Data reader '{name}' has different frequency than other data readers"
        return freq_ref

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
        shard_sizes: dict[str, ShardSizes],
    ) -> None:
        """Set model and reader communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        global_rank : int
            Global rank
        model_comm_group_id : int
            Model communication group ID
        model_comm_group_rank : int
            Model communication group rank
        model_comm_num_groups : int
            Number of model communication groups
        reader_group_rank : int
            Reader group rank
        reader_group_size : int
            Reader group size
        shard_sizes : dict[str, ShardSizes]
            Shard sizes for all datasets
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.sample_comm_group_id = model_comm_group_id
        self.sample_comm_num_groups = model_comm_num_groups

        self.shard_sizes = shard_sizes

        assert self.reader_group_size >= 1, f"reader_group_size(={self.reader_group_size}) must be positive"

        LOGGER.info(
            "NativeGridDataset.set_group_info(): global_rank %d, model_comm_group_id %d, "
            "model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d, "
            "sample_comm_group_id %d, sample_comm_num_groups %d",
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
            self.sample_comm_group_id,
            self.sample_comm_num_groups,
        )

    def set_ens_comm_group_info(
        self,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
    ) -> None:
        """Set ensemble communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        ens_comm_group_id : int
            Ensemble communication group ID
        ens_comm_group_rank : int
            Ensemble communication group rank
        ens_comm_num_groups : int
            Number of ensemble communication groups
        """
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups

        self.sample_comm_group_id = ens_comm_group_id
        self.sample_comm_num_groups = ens_comm_num_groups

        LOGGER.info(
            "NativeGridDataset.set_ens_comm_group_info(): global_rank %d, ens_comm_group_id %d, "
            "ens_comm_group_rank %d, ens_comm_num_groups %d, reader_group_rank %d, "
            "sample_comm_group_id %d, sample_comm_num_groups %d",
            self.global_rank,
            ens_comm_group_id,
            ens_comm_group_rank,
            ens_comm_num_groups,
            self.reader_group_rank,
            self.sample_comm_group_id,
            self.sample_comm_num_groups,
        )

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Initialize all data readers for this worker."""
        self.worker_id = worker_id

        # 1. divide valid date indices into shards for sample communication groups (DDP ranks)
        # note that we need even splits here across DDP ranks, so we might throw away some samples
        epoch_size = int(self.epoch_sample["size"]) if self.epoch_sample is not None else len(self.valid_date_indices)
        if self.epoch_sample is not None and epoch_size % self.sample_comm_num_groups:
            raise ValueError("Epoch sample size must be divisible by the number of sample groups")
        shard_size = epoch_size // self.sample_comm_num_groups
        shard_start = self.sample_comm_group_id * shard_size

        self.n_samples_per_worker = shard_size // n_workers

        # 2. partition the shard across workers (here we can have uneven splits, so we use a balanced partition)
        low, high = get_balanced_partition_range(shard_size, n_workers, worker_id, offset=shard_start)

        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        LOGGER.info(
            "Worker %d (pid %d, global_rank %d, model comm group %d)  has low/high range %d / %d",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            low,
            high,
        )

        base_seed = get_base_seed()
        # The datamodule checkpoints this epoch and restores it before new workers
        # start, so resuming from an epoch checkpoint derives the same seed.
        seed = derive_seed(base_seed, SeedContext.DATALOADER, self.epoch)

        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        sanity_rnd = self.rng.random(1)[0]
        LOGGER.info(
            ("Worker %d (%s, pid %d, epoch %d, rollout %d, seed %d, sanity rnd %f)"),
            worker_id,
            self.label,
            os.getpid(),
            self.epoch,
            self.rollout,
            seed,
            sanity_rnd,
        )

    def get_sample(self, index: int) -> dict[str, torch.Tensor]:
        sequence, position = (int(v) for v in self.anchors[index])
        x = {}
        for name, dataset in self.data_readers.items():
            if self.dataset_offsets is None:
                time_steps = offset_time_indices(position, self.relative_date_indices[name])
            else:
                anchor_ns = self.dataset_dates_ns[self.anchor_dataset_name][position]
                dates_ns = self.dataset_dates_ns[name]
                requested = anchor_ns + np.array([int(offset.total_seconds() * 1e9) for offset in self.dataset_offsets[name]])
                upper = np.searchsorted(dates_ns, requested).clip(0, len(dates_ns) - 1)
                lower = (upper - 1).clip(0, len(dates_ns) - 1)
                # The legacy loader chose the later timestamp on a half-step tie.
                time_steps = np.where(abs(dates_ns[upper] - requested) <= abs(dates_ns[lower] - requested), upper, lower)
                valid = abs(dates_ns[time_steps] - requested) <= int(dataset.frequency.total_seconds() * 5e8)
                valid &= ~np.isin(time_steps, list(dataset.missing_positions(sequence)))
            # self.shard_sizes is lazily initalised to None
            # This if statement guards against the case where shard_sizes is not set
            # (e.g. if set_comm_group_info hasn't been called yet)
            if self.shard_sizes is not None and self.shard_sizes[name] is not None:
                start, end = get_partition_range(self.shard_sizes[name], self.reader_group_rank)
                grid_indices = slice(start, end)
            else:
                grid_indices = slice(None)
            if self.dataset_offsets is None or valid.all():
                x[name] = dataset.get_sample(sequence, time_steps, grid_indices)
            else:
                good = np.flatnonzero(valid)
                if good.size:
                    loaded = dataset.get_sample(sequence, time_steps[good], grid_indices)
                else:
                    available = next(i for i in range(dataset.sequence_length(sequence)) if i not in dataset.missing_positions(sequence))
                    loaded = dataset.get_sample(sequence, [available], grid_indices)
                sample = loaded.new_full((len(time_steps), *loaded.shape[1:]), torch.nan)
                if good.size:
                    sample[good] = loaded
                x[name] = sample

        return x

    def __iter__(self) -> dict[str, torch.Tensor]:
        """Return an iterator that yields dictionaries of synchronized samples.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping dataset names to their tensor samples
            Format: {"dataset_a": tensor_a, "dataset_b": tensor_b, ...}
        """
        # Get the shuffled indices from the primary dataset
        # All data readers will use the same shuffled indices for synchronization
        shuffled_chunk_indices = self._epoch_indices()[self.chunk_index_range]

        LOGGER.debug(
            "%s worker pid %d, worker id %d, using synchronized indices[0:10]: %s",
            self.__class__.__name__,
            os.getpid(),
            self.worker_id,
            shuffled_chunk_indices[:10],
        )

        initial_batch = None

        # TODO(): improve this...
        for i in shuffled_chunk_indices:
            if not self.fake_dataloading:
                yield self.get_sample(i)
            elif initial_batch is None:
                initial_batch = self.get_sample(i)
                yield initial_batch
            else:
                yield initial_batch

    def __repr__(self) -> str:
        console = Console(record=True, width=120)
        with console.capture() as capture:
            console.print(self.tree())
        return capture.get()

    def tree(self) -> Tree:
        tree = Tree(f"{self.__class__.__name__}")
        for name, dataset in self.data_readers.items():
            subtree = dataset.tree(prefix=name)
            tree.add(subtree)
        return tree
