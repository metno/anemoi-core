# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
import torch.nn.functional as F
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import reduce_tensor
from anemoi.training.losses.base import BaseLoss
from anemoi.training.utils.enums import TensorDim


class LightningTemporalIncrementHuberLoss(BaseLoss):
    """Huber loss on frame-to-frame lightning count increments."""

    name: str = "lightning_temporal_increment_huber"

    def __init__(
        self,
        delta: float = 1.0,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        self.delta = float(delta)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if pred.ndim != 5:
            raise ValueError(f"Expected pred shape (batch, time, ensemble, grid, vars), got {tuple(pred.shape)}")
        target = self.align_target_to_pred(pred, target)
        if target.ndim != 5:
            raise ValueError(f"Expected target shape compatible with pred, got {tuple(target.shape)}")
        if pred.shape[TensorDim.TIME] < 2 or target.shape[TensorDim.TIME] < 2:
            raise ValueError("LightningTemporalIncrementHuberLoss requires at least two output steps")

        pred_delta = pred[:, 1:] - pred[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        out = F.huber_loss(pred_delta, target_delta, reduction="none", delta=self.delta)
        out = self.scale(
            out,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(out, squash=squash, squash_mode="avg", group=group if grid_shard_slice is not None else None)


class LightningActiveMaskLoss(BaseLoss):
    """Soft occurrence loss for active lightning frames."""

    name: str = "lightning_active_mask"

    def __init__(
        self,
        threshold: float = 0.5,
        temperature: float = 0.5,
        false_positive_weight: float = 1.0,
        false_negative_weight: float = 1.0,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.false_positive_weight = float(false_positive_weight)
        self.false_negative_weight = float(false_negative_weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if pred.ndim != 5:
            raise ValueError(f"Expected pred shape (batch, time, ensemble, grid, vars), got {tuple(pred.shape)}")
        if target.ndim == 5:
            target = target.select(TensorDim.ENSEMBLE_DIM, 0)
        if target.ndim != 4:
            raise ValueError(f"Expected target shape (batch, time, grid, vars), got {tuple(target.shape)}")

        pred_prob = torch.sigmoid((pred - self.threshold) / self.temperature).mean(dim=TensorDim.ENSEMBLE_DIM)
        target_active = (target > self.threshold).to(dtype=pred.dtype)
        diff = pred_prob - target_active
        out = self.false_positive_weight * torch.relu(diff).square()
        out = out + self.false_negative_weight * torch.relu(-diff).square()

        if self.ignore_nans:
            out = out.masked_fill(~torch.isfinite(target), torch.nan)
        elif torch.isnan(target).any():
            out = out.masked_fill(~torch.isfinite(target), 0.0)

        out = out.unsqueeze(TensorDim.ENSEMBLE_DIM)
        out = self.scale(
            out,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(out, squash=squash, squash_mode="avg", group=group if grid_shard_slice is not None else None)


class LightningSpatialTemporalAggregatedHuberLoss(BaseLoss):
    """Huber loss on spatial and temporal aggregates of lightning counts.

    The loss samples fixed centre nodes, sums predictions and targets over all
    nodes within radius_km of each centre, then sums over a rolling time
    window before applying Huber loss.
    """

    name: str = "lightning_spatial_temporal_aggregated_huber"

    def __init__(
        self,
        nodes_name: str = "lightning",
        radius_km: float = 8.0,
        window_size: int = 2,
        num_points: int | None = None,
        fraction: float | None = None,
        seed: int = 0,
        delta: float = 1.0,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        if radius_km <= 0.0:
            raise ValueError("radius_km must be > 0")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if num_points is not None and int(num_points) <= 0:
            raise ValueError("num_points must be positive")
        if fraction is not None and not 0.0 < float(fraction) <= 1.0:
            raise ValueError("fraction must be in (0, 1]")

        self.nodes_name = str(nodes_name)
        self.radius_km = float(radius_km)
        self.window_size = int(window_size)
        self.num_points = None if num_points is None else int(num_points)
        self.fraction = None if fraction is None else float(fraction)
        self.seed = int(seed)
        self.delta = float(delta)
        self.supports_sharding = False

    def set_graph_data(self, graph_data) -> "LightningSpatialTemporalAggregatedHuberLoss":
        import numpy as np
        from scipy.spatial import cKDTree

        coords = graph_data[self.nodes_name].x.detach().cpu().numpy()
        grid_size = int(coords.shape[0])
        sample_size = grid_size
        if self.fraction is not None:
            sample_size = max(1, int(round(grid_size * self.fraction)))
        if self.num_points is not None:
            sample_size = min(sample_size, self.num_points) if self.fraction is not None else self.num_points
        sample_size = min(max(1, int(sample_size)), grid_size)

        if sample_size < grid_size:
            rng = np.random.default_rng(self.seed)
            centre_indices = np.sort(rng.choice(grid_size, size=sample_size, replace=False))
        else:
            centre_indices = np.arange(grid_size, dtype=np.int64)

        xyz = self._unit_xyz(coords)
        radius = 2.0 * np.sin((self.radius_km / 6371.0) / 2.0)
        neighbours = cKDTree(xyz).query_ball_point(xyz[centre_indices], r=radius)
        max_neighbours = max(len(item) for item in neighbours)
        neighbour_indices = np.zeros((sample_size, max_neighbours), dtype=np.int64)
        neighbour_mask = np.zeros((sample_size, max_neighbours), dtype=bool)
        for i, item in enumerate(neighbours):
            if not item:
                item = [int(centre_indices[i])]
            count = len(item)
            neighbour_indices[i, :count] = item
            neighbour_mask[i, :count] = True

        self.register_buffer("centre_indices", torch.as_tensor(centre_indices, dtype=torch.long), persistent=False)
        self.register_buffer("neighbour_indices", torch.as_tensor(neighbour_indices, dtype=torch.long), persistent=False)
        self.register_buffer("neighbour_mask", torch.as_tensor(neighbour_mask, dtype=torch.bool), persistent=False)
        return self

    @staticmethod
    def _unit_xyz(coords):
        import numpy as np

        cos_lat = np.cos(coords[:, 0])
        return np.column_stack((cos_lat * np.cos(coords[:, 1]), cos_lat * np.sin(coords[:, 1]), np.sin(coords[:, 0])))

    def _spatial_sum(self, values: torch.Tensor) -> torch.Tensor:
        neighbour_indices = self.neighbour_indices.to(device=values.device)
        neighbour_mask = self.neighbour_mask.to(device=values.device, dtype=values.dtype)
        flat_indices = neighbour_indices.reshape(-1)
        gathered = torch.index_select(values, TensorDim.GRID, flat_indices)
        gathered = gathered.reshape(*values.shape[:TensorDim.GRID], *neighbour_indices.shape, values.shape[-1])
        gathered = gathered * neighbour_mask.view(1, 1, 1, *neighbour_indices.shape, 1)
        return self.sum_function(gathered, dim=TensorDim.GRID + 1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del scaler_indices, without_scalers, kwargs
        if grid_shard_slice is not None:
            raise ValueError("LightningSpatialTemporalAggregatedHuberLoss requires unsharded predictions")
        if not hasattr(self, "neighbour_indices"):
            raise ValueError("LightningSpatialTemporalAggregatedHuberLoss requires set_graph_data() before use")
        if pred.ndim != 5:
            raise ValueError(f"Expected pred shape (batch, time, ensemble, grid, vars), got {tuple(pred.shape)}")
        target = self.align_target_to_pred(pred, target)
        if target.ndim != 5:
            raise ValueError(f"Expected target shape compatible with pred, got {tuple(target.shape)}")
        if pred.shape[TensorDim.TIME] < self.window_size or target.shape[TensorDim.TIME] < self.window_size:
            raise ValueError(
                f"window_size={self.window_size} exceeds available output steps "
                f"(pred={pred.shape[TensorDim.TIME]}, target={target.shape[TensorDim.TIME]})"
            )

        valid = torch.isfinite(pred) & torch.isfinite(target)
        pred = torch.where(valid, pred, torch.zeros_like(pred))
        target = torch.where(valid, target, torch.zeros_like(target))

        pred_spatial = self._spatial_sum(pred)
        target_spatial = self._spatial_sum(target)

        pred_acc = pred_spatial.unfold(dimension=TensorDim.TIME, size=self.window_size, step=1).sum(dim=-1)
        target_acc = target_spatial.unfold(dimension=TensorDim.TIME, size=self.window_size, step=1).sum(dim=-1)
        out = F.huber_loss(pred_acc, target_acc, reduction="none", delta=self.delta)

        if not self.ignore_nans:
            out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return self.reduce(out, squash=squash, squash_mode="avg", group=group)


class LightningSpatialTemporalAggregatedAlmostFairKernelCRPSLoss(LightningSpatialTemporalAggregatedHuberLoss):
    """Almost-fair CRPS on spatial and temporal aggregates of lightning counts."""

    name: str = "lightning_spatial_temporal_aggregated_afkcrps"

    def __init__(
        self,
        nodes_name: str = "lightning",
        radius_km: float = 8.0,
        window_size: int = 2,
        num_points: int | None = None,
        fraction: float | None = None,
        seed: int = 0,
        alpha: float = 1.0,
        no_autocast: bool = True,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        kwargs.pop("delta", None)
        super().__init__(
            nodes_name=nodes_name,
            radius_km=radius_km,
            window_size=window_size,
            num_points=num_points,
            fraction=fraction,
            seed=seed,
            delta=1.0,
            ignore_nans=ignore_nans,
            **kwargs,
        )
        self.alpha = float(alpha)
        self.no_autocast = bool(no_autocast)

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ens_size = preds.shape[-1]
        if ens_size <= 1:
            raise ValueError("Ensemble size must be greater than 1 for AKCRPS.")

        alpha = torch.as_tensor(self.alpha, dtype=preds.dtype, device=preds.device)
        epsilon = (1.0 - alpha) / ens_size
        var = torch.abs(preds.unsqueeze(dim=-1) - preds.unsqueeze(dim=-2))
        err = torch.abs(preds - targets.unsqueeze(dim=-1))
        err_pairs = err.unsqueeze(dim=-2).expand(*err.shape[:-1], ens_size, ens_size)
        diag = torch.eye(ens_size, dtype=torch.bool, device=preds.device)
        mem_err = err_pairs * ~diag

        coef = 1.0 / (2.0 * ens_size * (ens_size - 1))
        return coef * torch.sum(mem_err + mem_err.transpose(-1, -2) - (1.0 - epsilon) * var, dim=(-1, -2))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del scaler_indices, without_scalers, kwargs
        if grid_shard_slice is not None:
            raise ValueError("LightningSpatialTemporalAggregatedAlmostFairKernelCRPSLoss requires unsharded predictions")
        if not hasattr(self, "neighbour_indices"):
            raise ValueError("LightningSpatialTemporalAggregatedAlmostFairKernelCRPSLoss requires set_graph_data() before use")
        if pred.ndim != 5:
            raise ValueError(f"Expected pred shape (batch, time, ensemble, grid, vars), got {tuple(pred.shape)}")
        target = self.align_target_to_pred(pred, target)
        if target.ndim != 5:
            raise ValueError(f"Expected target shape compatible with pred, got {tuple(target.shape)}")
        if pred.shape[TensorDim.TIME] < self.window_size or target.shape[TensorDim.TIME] < self.window_size:
            raise ValueError(
                f"window_size={self.window_size} exceeds available output steps "
                f"(pred={pred.shape[TensorDim.TIME]}, target={target.shape[TensorDim.TIME]})"
            )

        valid = torch.isfinite(pred) & torch.isfinite(target)
        pred = torch.where(valid, pred, torch.zeros_like(pred))
        target = torch.where(valid, target, torch.zeros_like(target))

        pred_spatial = self._spatial_sum(pred)
        target_spatial = self._spatial_sum(target)

        pred_acc = pred_spatial.unfold(dimension=TensorDim.TIME, size=self.window_size, step=1).sum(dim=-1)
        target_acc = target_spatial.unfold(dimension=TensorDim.TIME, size=self.window_size, step=1).sum(dim=-1)
        target_acc = target_acc.select(TensorDim.ENSEMBLE_DIM, 0)

        pred_crps = pred_acc.permute(0, 1, 4, 3, 2)
        target_crps = target_acc.permute(0, 1, 3, 2)
        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                out = self._kernel_crps(pred_crps, target_crps)
        else:
            out = self._kernel_crps(pred_crps, target_crps)

        out = out.permute(0, 1, 3, 2).unsqueeze(TensorDim.ENSEMBLE_DIM)
        if not self.ignore_nans:
            out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return self.reduce(out, squash=squash, squash_mode="sum", group=group)

    @property
    def name(self) -> str:
        return f"lightning_spatial_temporal_aggregated_afkcrps{self.alpha:.2f}"
