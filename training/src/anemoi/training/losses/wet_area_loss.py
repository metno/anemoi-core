# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.utils.enums import TensorDim


class WeightedSoftWetAreaLoss(FunctionalLoss):
    name: str = "weighted_soft_wet_area"

    def __init__(
        self,
        threshold: float = 0.0,
        temperature: float = 0.1,
        false_positive_weight: float = 1.0,
        false_negative_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.false_positive_weight = float(false_positive_weight)
        self.false_negative_weight = float(false_negative_weight)

    def _soft_wet_mask(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((x - self.threshold) / self.temperature)

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_wet = self._soft_wet_mask(pred)
        target_wet = self._soft_wet_mask(target)

        diff = pred_wet - target_wet
        fp = torch.relu(diff)
        fn = torch.relu(-diff)

        return self.false_positive_weight * fp**2 + self.false_negative_weight * fn**2


class MultiThresholdWetAreaLoss(FunctionalLoss):
    name: str = "multi_threshold_wet_area"

    def __init__(
        self,
        thresholds: tuple[float, ...] = (0.1, 1.0, 5.0),
        weights: tuple[float, ...] = (1.0, 0.5, 0.2),
        temperature: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if len(thresholds) != len(weights):
            raise ValueError("thresholds and weights must have the same length")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.thresholds = tuple(float(threshold) for threshold in thresholds)
        self.weights = tuple(float(weight) for weight in weights)
        self.temperature = float(temperature)

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == pred.ndim:
            target = target.select(TensorDim.ENSEMBLE_DIM, 0)

        loss = torch.zeros_like(target)
        for threshold, weight in zip(self.thresholds, self.weights, strict=True):
            pred_prob = torch.sigmoid((pred - threshold) / self.temperature).mean(dim=TensorDim.ENSEMBLE_DIM)
            obs_mask = (target > threshold).to(dtype=pred.dtype)
            loss = loss + weight * (pred_prob - obs_mask).square()
        return loss.unsqueeze(TensorDim.ENSEMBLE_DIM)


class NetatmoWetDryOnRadarGridLoss(BaseLoss):
    name: str = "netatmo_wet_dry_on_radar_grid"

    def __init__(
        self,
        source_dataset: str = "netatmo",
        source_nodes_name: str = "netatmo",
        target_nodes_name: str = "nordic_radar",
        source_variable: str = "rr",
        source_threshold: float = 0.1,
        prediction_threshold: float | None = None,
        temperature: float = 0.1,
        min_station_count: int = 1,
        max_distance_degrees: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if min_station_count < 1:
            raise ValueError(f"min_station_count must be >= 1, got {min_station_count}")
        self.source_dataset = str(source_dataset)
        self.source_nodes_name = str(source_nodes_name)
        self.target_nodes_name = str(target_nodes_name)
        self.source_variable = str(source_variable)
        self.source_threshold = float(source_threshold)
        self.prediction_threshold = float(source_threshold if prediction_threshold is None else prediction_threshold)
        self.temperature = float(temperature)
        self.min_station_count = int(min_station_count)
        self.max_distance_degrees = None if max_distance_degrees is None else float(max_distance_degrees)
        self.supports_sharding = False
        self.target_grid_size: int | None = None

    def set_graph_data(self, graph_data) -> "NetatmoWetDryOnRadarGridLoss":
        import math

        import numpy as np
        from scipy.spatial import cKDTree

        target_coords = graph_data[self.target_nodes_name].x.detach().cpu().numpy()
        source_coords = graph_data[self.source_nodes_name].x.detach().cpu().numpy()
        distances, indices = cKDTree(target_coords).query(source_coords, k=1)
        valid = np.ones(source_coords.shape[0], dtype=bool)
        if self.max_distance_degrees is not None:
            valid = distances <= math.radians(self.max_distance_degrees)

        self.target_grid_size = int(target_coords.shape[0])
        self.register_buffer(
            "source_to_target_index",
            torch.as_tensor(indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "source_to_target_valid",
            torch.as_tensor(valid, dtype=torch.bool),
            persistent=False,
        )
        return self

    def _source_values(
        self,
        targets_by_dataset: dict[str, torch.Tensor],
        data_indices_by_dataset,
        device: torch.device,
    ) -> torch.Tensor:
        if self.source_dataset not in targets_by_dataset:
            raise ValueError(f"Missing source dataset {self.source_dataset!r} in targets_by_dataset")
        source_target = targets_by_dataset[self.source_dataset]
        if source_target.ndim == 5:
            source_target = source_target.select(TensorDim.ENSEMBLE_DIM, 0)
        if source_target.ndim != 4:
            raise ValueError(
                "NetatmoWetDryOnRadarGridLoss expects source target shape "
                f"(batch, time, grid, variables), got {tuple(source_target.shape)}"
            )
        source_index = int(data_indices_by_dataset[self.source_dataset].data.output.name_to_index[self.source_variable])
        return source_target[..., source_index].to(device=device)

    def _mean_source_on_target_grid(self, source_values: torch.Tensor, target_grid_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_to_target_index = self.source_to_target_index.to(device=source_values.device)
        source_to_target_valid = self.source_to_target_valid.to(device=source_values.device)
        valid_source = torch.isfinite(source_values) & source_to_target_valid.view(1, 1, -1)
        values = torch.where(valid_source, source_values, torch.zeros_like(source_values))

        batch_size, n_steps, _ = values.shape
        target_index = source_to_target_index.view(1, 1, -1).expand(batch_size, n_steps, -1)
        sums = torch.zeros(batch_size, n_steps, target_grid_size, dtype=values.dtype, device=values.device)
        counts = torch.zeros_like(sums)
        sums.scatter_add_(2, target_index, values)
        counts.scatter_add_(2, target_index, valid_source.to(dtype=values.dtype))

        mean_values = sums / counts.clamp_min(1.0)
        valid_target = counts >= self.min_station_count
        return mean_values, valid_target

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group=None,
        targets_by_dataset: dict[str, torch.Tensor] | None = None,
        data_indices_by_dataset=None,
        **kwargs,
    ) -> torch.Tensor:
        del target, group
        if grid_shard_slice is not None:
            raise ValueError("NetatmoWetDryOnRadarGridLoss requires unsharded radar-grid predictions")
        if targets_by_dataset is None or data_indices_by_dataset is None:
            raise ValueError("NetatmoWetDryOnRadarGridLoss requires targets_by_dataset and data_indices_by_dataset")
        if not hasattr(self, "source_to_target_index") or self.target_grid_size is None:
            raise ValueError("NetatmoWetDryOnRadarGridLoss requires set_graph_data() before use")
        if pred.ndim != 5:
            raise ValueError(f"Expected pred shape (batch, time, ensemble, grid, variables), got {tuple(pred.shape)}")
        if pred.shape[TensorDim.GRID] != self.target_grid_size:
            raise ValueError(
                "Prediction grid size does not match target graph nodes: "
                f"pred={pred.shape[TensorDim.GRID]}, graph={self.target_grid_size}"
            )

        source_values = self._source_values(targets_by_dataset, data_indices_by_dataset, pred.device)
        source_mean, valid_target = self._mean_source_on_target_grid(source_values, self.target_grid_size)
        obs_wet = (source_mean > self.source_threshold).to(dtype=pred.dtype).unsqueeze(-1)
        pred_prob = torch.sigmoid((pred - self.prediction_threshold) / self.temperature).mean(
            dim=TensorDim.ENSEMBLE_DIM,
        )
        loss = (pred_prob - obs_wet).square()
        loss = loss.masked_fill(~valid_target.unsqueeze(-1), torch.nan if self.ignore_nans else 0.0)
        loss = loss.unsqueeze(TensorDim.ENSEMBLE_DIM)
        loss = self.scale(
            loss,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(loss, squash, group=None, squash_mode=kwargs.get("squash_mode", "avg"))

