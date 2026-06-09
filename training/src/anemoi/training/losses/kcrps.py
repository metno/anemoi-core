# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import reduce_tensor
from anemoi.training.losses.base import BaseLoss
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class KernelCRPS(BaseLoss):
    """Kernel CRPS loss."""

    def __init__(
        self,
        fair: bool = True,
        ignore_nans: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Latitude- and (inverse-)variance-weighted kernel CRPS loss.

        Parameters
        ----------
        fair : bool
            Calculate a "fair" (unbiased) score - ensemble variance component weighted by (ens-size-1)^-1.
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(ignore_nans=ignore_nans)

        self.fair = fair

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Kernel (ensemble) CRPS.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted ensemble, shape (batch_size, n_out_steps, n_vars, latlon, ens_size)
        targets : torch.Tensor
            Ground truth, shape (batch_size, n_out_steps, n_vars, latlon)

        Returns
        -------
        kCRPS : torch.Tensor
            The point-wise kernel CRPS, shape (batch_size, n_out_steps, n_vars, latlon).
        """
        ens_size = preds.shape[-1]
        mae = torch.mean(torch.abs(targets[..., None] - preds), dim=-1)

        assert ens_size > 1, "Ensemble size must be greater than 1."

        coef = -1.0 / (ens_size * (ens_size - 1)) if self.fair else -1.0 / (ens_size**2)

        ens_var = torch.zeros(size=preds.shape[:-1], device=preds.device)
        for i in range(ens_size):  # loop version to reduce memory usage
            ens_var += torch.sum(torch.abs(preds[..., i].unsqueeze(-1) - preds[..., i + 1 :]), dim=-1)
        ens_var = coef * ens_var

        return mae + ens_var

    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None

        y_target = einops.rearrange(y_target, "bs t latlon v -> bs t v latlon")
        y_pred = einops.rearrange(y_pred, "bs t e latlon v -> bs t v latlon e")

        if self.ignore_nans:
            nan_mask = torch.isnan(y_target)
            y_target = y_target.masked_fill(nan_mask, 0.0)
            # Expand mask for ensemble dimension: (bs, v, latlon) -> (bs, v, latlon, e)
            y_pred = y_pred.masked_fill(nan_mask.unsqueeze(-1), 0.0)

        kcrps_ = self._kernel_crps(y_pred, y_target)

        kcrps_ = einops.rearrange(kcrps_, "bs t v latlon -> bs t 1 latlon v")
        kcrps_ = self.scale(kcrps_, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        return self.reduce(kcrps_, squash=squash, squash_mode="sum", group=group if is_sharded else None)

    @property
    def name(self) -> str:
        f_str = "f" if self.fair else ""
        return f"{f_str}kcrps"


class AlmostFairKernelCRPS(BaseLoss):
    """Almost fair kernel CRPS loss."""

    def __init__(
        self,
        alpha: float = 1.0,
        alpha_scaler_name: str | None = None,
        no_autocast: bool = True,
        ignore_nans: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Latitude- and (inverse-)variance-weighted kernel CRPS loss.

        Parameters
        ----------
        alpha : float
            Factor for linear combination of fair (unbiased, ensemble variance component weighted by (ens-size-1)^-1)
            and standard CRPS (1.0 = fully fair, 0.0 = fully unfair)
        alpha_scaler_name : str | None
            Optional scaler name to turn into a per-point alpha map. When provided,
            the scaler values are used directly as the local alpha values.
        no_autocast : bool, optional
            Deactivate autocast for the kernel CRPS calculation
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(ignore_nans=ignore_nans)

        self.alpha = alpha
        self.alpha_scaler_name = alpha_scaler_name
        self.no_autocast = no_autocast

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor, alpha: float | torch.Tensor = 1.0) -> torch.Tensor:
        """Kernel (ensemble) CRPS.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted ensemble, shape (batch_size, n_out_steps, n_vars, latlon, ens_size)
        targets : torch.Tensor
            Ground truth, shape (batch_size, n_out_steps, n_vars, latlon)
        alpha : float | torch.Tensor
            Factor for linear combination of fair (unbiased, ensemble variance component weighted by (ens-size-1)^-1)
            and standard CRPS (1.0 = fully fair, 0.0 = fully unfair)

        Returns
        -------
        kCRPS : torch.Tensor
            The point-wise kernel CRPS, shape (batch_size, n_out_steps, n_vars, latlon).
        """
        ens_size = preds.shape[-1]

        alpha = torch.as_tensor(alpha, dtype=preds.dtype, device=preds.device)
        epsilon = (1.0 - alpha) / ens_size
        if epsilon.ndim > 0:
            epsilon = epsilon[..., None, None]

        var = torch.abs(preds.unsqueeze(dim=-1) - preds.unsqueeze(dim=-2))
        diag = torch.eye(ens_size, dtype=torch.bool, device=preds.device)
        err_r = einops.repeat(
            torch.abs(preds - targets.unsqueeze(dim=-1)),
            "batch t var latlon ens -> batch t var latlon n ens",
            n=ens_size,
        )

        mem_err = err_r * ~diag
        mem_err_transpose = mem_err.transpose(-1, -2)

        assert ens_size > 1, "Ensemble size must be greater than 1."

        coef = 1.0 / (2.0 * ens_size * (ens_size - 1))
        return coef * torch.sum(mem_err + mem_err_transpose - (1 - epsilon) * var, dim=(-1, -2))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None

        y_target = einops.rearrange(y_target, "bs t latlon v -> bs t v latlon")
        y_pred = einops.rearrange(y_pred, "bs t e latlon v -> bs t v latlon e")

        if self.ignore_nans:
            nan_mask = torch.isnan(y_target)
            y_target = y_target.masked_fill(nan_mask, 0.0)
            # Expand mask for ensemble dimension: (bs, v, latlon) -> (bs, v, latlon, e)
            y_pred = y_pred.masked_fill(nan_mask.unsqueeze(-1), 0.0)

        alpha = self.alpha
        if self.alpha_scaler_name is not None:
            keep_scaler = self.alpha_scaler_name
            if keep_scaler not in self.scaler.tensors:
                msg = f"Scaler {keep_scaler!r} not found for {self.__class__.__name__}"
                raise ValueError(msg)

            alpha_map = torch.ones(
                (y_pred.shape[0], y_pred.shape[1], 1, y_pred.shape[3], y_pred.shape[2]),
                dtype=y_pred.dtype,
                device=y_pred.device,
            )
            alpha_map = self.scale(
                alpha_map,
                scaler_indices,
                without_scalers=[name for name in self.scaler.tensors if name != keep_scaler],
                grid_shard_slice=grid_shard_slice,
            )
            alpha = einops.rearrange(alpha_map, "bs t 1 latlon v -> bs t v latlon")

        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                kcrps_ = self._kernel_crps(y_pred, y_target, alpha=alpha)
        else:
            kcrps_ = self._kernel_crps(y_pred, y_target, alpha=alpha)

        kcrps_ = einops.rearrange(kcrps_, "bs t v latlon -> bs t 1 latlon v")
        kcrps_ = self.scale(kcrps_, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        return self.reduce(kcrps_, squash=squash, squash_mode="sum", group=group if is_sharded else None)

    @property
    def name(self) -> str:
        return f"afkcrps{self.alpha:.2f}"


class SampledGridPointAlmostFairKernelCRPS(AlmostFairKernelCRPS):
    """Almost-fair kernel CRPS evaluated on a random subset of grid points."""

    def __init__(
        self,
        num_points: int | None = None,
        fraction: float | None = None,
        rescale: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if num_points is None and fraction is None:
            raise ValueError("SampledGridPointAlmostFairKernelCRPS requires num_points or fraction.")
        if num_points is not None and int(num_points) <= 0:
            raise ValueError("SampledGridPointAlmostFairKernelCRPS num_points must be positive.")
        if fraction is not None and not 0.0 < float(fraction) <= 1.0:
            raise ValueError("SampledGridPointAlmostFairKernelCRPS fraction must be in (0, 1].")
        self.num_points = None if num_points is None else int(num_points)
        self.fraction = None if fraction is None else float(fraction)
        self.rescale = bool(rescale)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **kwargs,
    ) -> torch.Tensor:
        grid_size = int(y_pred.shape[TensorDim.GRID])
        sample_size = grid_size
        if self.fraction is not None:
            sample_size = max(1, int(round(grid_size * self.fraction)))
        if self.num_points is not None:
            sample_size = min(sample_size, self.num_points) if self.fraction is not None else self.num_points
        sample_size = min(max(1, int(sample_size)), grid_size)

        if sample_size >= grid_size:
            return super().forward(
                y_pred,
                y_target,
                squash=squash,
                scaler_indices=scaler_indices,
                without_scalers=without_scalers,
                grid_shard_slice=grid_shard_slice,
                group=group,
                **kwargs,
            )

        indices = torch.randint(grid_size, (sample_size,), device=y_pred.device)
        local_indices = torch.arange(sample_size, device=y_pred.device)
        target_grid_dim = TensorDim.GRID if y_target.ndim == y_pred.ndim else TensorDim.GRID - 1
        y_pred = torch.index_select(y_pred, TensorDim.GRID, indices)
        y_target = torch.index_select(y_target, target_grid_dim, indices)

        if scaler_indices is None:
            sampled_scaler_indices = [slice(None)] * y_pred.ndim
        else:
            sampled_scaler_indices = []
            ellipsis_seen = False
            for item in scaler_indices:
                if item is Ellipsis:
                    if ellipsis_seen:
                        raise ValueError("Only one ellipsis is allowed in scaler_indices.")
                    ellipsis_seen = True
                    missing_dims = y_pred.ndim - (len(scaler_indices) - 1)
                    sampled_scaler_indices.extend([slice(None)] * missing_dims)
                else:
                    sampled_scaler_indices.append(item)
            if len(sampled_scaler_indices) != y_pred.ndim:
                raise ValueError("scaler_indices must resolve to the prediction tensor rank.")

        sampled_scaler_indices[TensorDim.GRID] = local_indices

        if y_target.ndim == y_pred.ndim:
            target_ens = y_target.shape[TensorDim.ENSEMBLE_DIM]
            pred_ens = y_pred.shape[TensorDim.ENSEMBLE_DIM]
            if target_ens not in (1, pred_ens):
                msg = (
                    "Prediction and target ensemble dimensions are incompatible: "
                    f"pred={tuple(y_pred.shape)}, target={tuple(y_target.shape)}"
                )
                raise ValueError(msg)
            y_target = y_target.select(TensorDim.ENSEMBLE_DIM, 0)

        y_target = einops.rearrange(y_target, "bs t latlon v -> bs t v latlon")
        y_pred = einops.rearrange(y_pred, "bs t e latlon v -> bs t v latlon e")

        if self.ignore_nans:
            nan_mask = torch.isnan(y_target)
            y_target = y_target.masked_fill(nan_mask, 0.0)
            y_pred = y_pred.masked_fill(nan_mask.unsqueeze(-1), 0.0)

        if self.alpha_scaler_name is not None:
            msg = "SampledGridPointAlmostFairKernelCRPS does not support alpha_scaler_name."
            raise ValueError(msg)

        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                kcrps_ = self._kernel_crps(y_pred, y_target, alpha=self.alpha)
        else:
            kcrps_ = self._kernel_crps(y_pred, y_target, alpha=self.alpha)

        kcrps_ = einops.rearrange(kcrps_, "bs t v latlon -> bs t 1 latlon v")
        kcrps_ = self._scale_sampled(
            kcrps_,
            tuple(sampled_scaler_indices),
            indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        loss = self.reduce(kcrps_, squash=squash, squash_mode="sum", group=None)
        if self.rescale:
            loss = loss * (grid_size / sample_size)
        return loss if group is None or grid_shard_slice is None else reduce_tensor(loss, group)

    def _scale_sampled(
        self,
        x: torch.Tensor,
        subset_indices: tuple[int, ...],
        grid_indices: torch.Tensor,
        *,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        def apply_subset(tensor: torch.Tensor) -> torch.Tensor:
            out = tensor
            for dim, index in enumerate(subset_indices):
                if index is Ellipsis:
                    continue
                if isinstance(index, slice):
                    slices = [slice(None)] * out.ndim
                    slices[dim] = index
                    out = out[tuple(slices)]
                    continue
                index = torch.as_tensor(index, dtype=torch.long, device=out.device).reshape(-1)
                index = index % out.shape[dim]
                out = torch.index_select(out, dim, index)
            return out

        if len(self.scaler) == 0:
            return apply_subset(x)

        scale_tensor = self.scaler
        if without_scalers is not None and len(without_scalers) > 0:
            if isinstance(without_scalers[0], str):
                scale_tensor = self.scaler.without(without_scalers)
            else:
                scale_tensor = self.scaler.without_by_dim(without_scalers)

        out = apply_subset(x).clone()
        tensors = scale_tensor.resolve(x.ndim).tensors
        for dims, scaler in tensors.values():
            dims = list(dims)
            if TensorDim.GRID in dims:
                grid_index = dims.index(TensorDim.GRID)
                if grid_shard_slice is not None and scaler.shape[grid_index] >= grid_shard_slice.stop:
                    slices = [slice(None)] * len(dims)
                    slices[grid_index] = grid_shard_slice
                    scaler = scaler[tuple(slices)]
                scaler = torch.index_select(scaler, grid_index, grid_indices)

            missing_dims = [d for d in range(x.ndim) if d not in dims]
            reshape = [1] * len(missing_dims)
            reshape.extend(scaler.shape)

            reshaped_scaler = scaler.reshape(reshape)
            reshaped_scaler = torch.moveaxis(reshaped_scaler, list(range(x.ndim)), (*missing_dims, *dims))
            reshaped_scaler = reshaped_scaler.expand_as(x)
            out = out * apply_subset(reshaped_scaler)

        return out

    @property
    def name(self) -> str:
        return f"sampled_grid_afkcrps{self.alpha:.2f}"


class SampledNetatmoAlmostFairKernelCRPSOnRadarGridLoss(AlmostFairKernelCRPS):
    """Almost-fair kernel CRPS at Netatmo stations using nearest radar-grid predictions."""

    def __init__(
        self,
        num_points: int | None = None,
        fraction: float | None = None,
        rescale: bool = True,
        source_dataset: str = "netatmo",
        source_nodes_name: str = "netatmo",
        target_nodes_name: str = "nordic_radar",
        source_variable: str = "rr",
        source_confidence_variable: str = "number_neighbours",
        min_station_count: int = 1,
        max_distance_degrees: float | None = None,
        source_distance_threshold_km: float | None = None,
        distance_weighted: bool = False,
        distance_weight_radius_km: float = 5.0,
        lambda_max: float = 0.8,
        neighbour_confidence_q_min: float = 0.25,
        neighbour_confidence_n0: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if num_points is not None and int(num_points) <= 0:
            raise ValueError("SampledNetatmoAlmostFairKernelCRPSOnRadarGridLoss num_points must be positive.")
        if fraction is not None and not 0.0 < float(fraction) <= 1.0:
            raise ValueError("SampledNetatmoAlmostFairKernelCRPSOnRadarGridLoss fraction must be in (0, 1].")
        if min_station_count < 1:
            raise ValueError(f"min_station_count must be >= 1, got {min_station_count}")
        if source_distance_threshold_km is not None and float(source_distance_threshold_km) <= 0.0:
            raise ValueError(f"source_distance_threshold_km must be > 0, got {source_distance_threshold_km}")
        if distance_weighted and source_distance_threshold_km is not None:
            raise ValueError("distance_weighted cannot be combined with source_distance_threshold_km")
        if distance_weight_radius_km <= 0.0:
            raise ValueError(f"distance_weight_radius_km must be > 0, got {distance_weight_radius_km}")
        if not 0.0 <= lambda_max <= 1.0:
            raise ValueError(f"lambda_max must be in [0, 1], got {lambda_max}")
        if not 0.0 <= neighbour_confidence_q_min <= 1.0:
            raise ValueError(f"neighbour_confidence_q_min must be in [0, 1], got {neighbour_confidence_q_min}")
        if neighbour_confidence_n0 <= 0.0:
            raise ValueError(f"neighbour_confidence_n0 must be > 0, got {neighbour_confidence_n0}")

        self.num_points = None if num_points is None else int(num_points)
        self.fraction = None if fraction is None else float(fraction)
        self.rescale = bool(rescale)
        self.source_dataset = str(source_dataset)
        self.source_nodes_name = str(source_nodes_name)
        self.target_nodes_name = str(target_nodes_name)
        self.source_variable = str(source_variable)
        self.source_confidence_variable = str(source_confidence_variable)
        self.min_station_count = int(min_station_count)
        self.max_distance_degrees = None if max_distance_degrees is None else float(max_distance_degrees)
        self.source_distance_threshold_km = (
            None if source_distance_threshold_km is None else float(source_distance_threshold_km)
        )
        self.distance_weighted = bool(distance_weighted)
        self.distance_weight_radius_km = float(distance_weight_radius_km)
        self.lambda_max = float(lambda_max)
        self.neighbour_confidence_q_min = float(neighbour_confidence_q_min)
        self.neighbour_confidence_n0 = float(neighbour_confidence_n0)
        self.supports_sharding = False
        self.target_grid_size: int | None = None

    def set_graph_data(self, graph_data) -> "SampledNetatmoAlmostFairKernelCRPSOnRadarGridLoss":
        import math

        import numpy as np
        from scipy.spatial import cKDTree

        target_coords = graph_data[self.target_nodes_name].x.detach().cpu().numpy()
        source_coords = graph_data[self.source_nodes_name].x.detach().cpu().numpy()
        distances, indices = cKDTree(target_coords).query(source_coords, k=1)
        valid = np.ones(source_coords.shape[0], dtype=bool)
        if self.max_distance_degrees is not None:
            valid = distances <= math.radians(self.max_distance_degrees)

        matched_target_coords = target_coords[indices]
        dlat = matched_target_coords[:, 0] - source_coords[:, 0]
        dlon = matched_target_coords[:, 1] - source_coords[:, 1]
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(source_coords[:, 0]) * np.cos(matched_target_coords[:, 0]) * np.sin(dlon / 2.0) ** 2
        )
        distance_km = 6371.0 * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, None))))
        use_source = np.ones(source_coords.shape[0], dtype=bool)
        if self.source_distance_threshold_km is not None:
            use_source = distance_km <= self.source_distance_threshold_km

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
        self.register_buffer(
            "source_to_target_use_source",
            torch.as_tensor(use_source, dtype=torch.bool),
            persistent=False,
        )
        if self.distance_weighted:
            source_xyz = self._unit_xyz(source_coords)
            target_xyz = self._unit_xyz(target_coords)
            _, target_to_source = cKDTree(source_xyz).query(target_xyz, k=1)
            target_to_source_coords = source_coords[target_to_source]
            target_to_source_distance_km = self._haversine_km(target_coords, target_to_source_coords)

            r = target_to_source_distance_km / self.distance_weight_radius_km
            kernel = np.where(r < 1.0, np.clip(1.0 - r, 0.0, None) ** 4 * (1.0 + 4.0 * r), 0.0)

            self.register_buffer(
                "target_to_source_index",
                torch.as_tensor(target_to_source, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "target_to_source_kernel",
                torch.as_tensor(kernel, dtype=torch.float32),
                persistent=False,
            )
        return self

    @staticmethod
    def _unit_xyz(coords):
        import numpy as np

        cos_lat = np.cos(coords[:, 0])
        return np.column_stack((cos_lat * np.cos(coords[:, 1]), cos_lat * np.sin(coords[:, 1]), np.sin(coords[:, 0])))

    @staticmethod
    def _haversine_km(source_coords, target_coords):
        import numpy as np

        dlat = target_coords[:, 0] - source_coords[:, 0]
        dlon = target_coords[:, 1] - source_coords[:, 1]
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(source_coords[:, 0]) * np.cos(target_coords[:, 0]) * np.sin(dlon / 2.0) ** 2
        )
        return 6371.0 * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, None))))

    def _source_values(
        self,
        targets_by_dataset: dict[str, torch.Tensor],
        data_indices_by_dataset,
        device: torch.device,
        source_variable: str | None = None,
    ) -> torch.Tensor:
        if self.source_dataset not in targets_by_dataset:
            raise ValueError(f"Missing source dataset {self.source_dataset!r} in targets_by_dataset")
        source_target = targets_by_dataset[self.source_dataset]
        if source_target.ndim == 5:
            source_target = source_target.select(TensorDim.ENSEMBLE_DIM, 0)
        if source_target.ndim != 4:
            raise ValueError(
                "SampledNetatmoAlmostFairKernelCRPSOnRadarGridLoss expects source target shape "
                f"(batch, time, grid, variables), got {tuple(source_target.shape)}"
            )
        source_variable = self.source_variable if source_variable is None else source_variable
        source_index = int(data_indices_by_dataset[self.source_dataset].data.output.name_to_index[source_variable])
        return source_target[..., source_index].to(device=device)

    def _mean_source_on_target_grid(self, source_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_to_target_index = self.source_to_target_index.to(device=source_values.device)
        source_to_target_valid = self.source_to_target_valid.to(device=source_values.device)
        valid_source = torch.isfinite(source_values) & source_to_target_valid.view(1, 1, -1)
        values = torch.where(valid_source, source_values, torch.zeros_like(source_values))

        batch_size, n_steps, _ = values.shape
        target_index = source_to_target_index.view(1, 1, -1).expand(batch_size, n_steps, -1)
        sums = torch.zeros(
            batch_size,
            n_steps,
            self.target_grid_size,
            dtype=values.dtype,
            device=values.device,
        )
        counts = torch.zeros_like(sums)
        sums.scatter_add_(2, target_index, values)
        counts.scatter_add_(2, target_index, valid_source.to(dtype=values.dtype))

        mean_values = sums / counts.clamp_min(1.0)
        valid_target = counts >= self.min_station_count
        return mean_values, valid_target

    def _resolve_sample_size(self, grid_size: int) -> int:
        sample_size = grid_size
        if self.fraction is not None:
            sample_size = max(1, int(round(grid_size * self.fraction)))
        if self.num_points is not None:
            sample_size = min(sample_size, self.num_points) if self.fraction is not None else self.num_points
        return min(max(1, int(sample_size)), grid_size)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        targets_by_dataset: dict[str, torch.Tensor] | None = None,
        data_indices_by_dataset=None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs

        if grid_shard_slice is not None:
            raise ValueError("SampledNetatmoAlmostFairKernelCRPSOnRadarGridLoss requires unsharded radar-grid predictions")
        if targets_by_dataset is None or data_indices_by_dataset is None:
            raise ValueError(
                "SampledNetatmoAlmostFairKernelCRPSOnRadarGridLoss requires targets_by_dataset and data_indices_by_dataset"
            )
        if not hasattr(self, "source_to_target_index") or self.target_grid_size is None:
            raise ValueError("SampledNetatmoAlmostFairKernelCRPSOnRadarGridLoss requires set_graph_data() before use")
        if y_pred.ndim != 5:
            raise ValueError(f"Expected pred shape (batch, time, ensemble, grid, variables), got {tuple(y_pred.shape)}")
        if y_pred.shape[TensorDim.GRID] != self.target_grid_size:
            raise ValueError(
                "Prediction grid size does not match target graph nodes: "
                f"pred={y_pred.shape[TensorDim.GRID]}, graph={self.target_grid_size}"
            )
        if self.alpha_scaler_name is not None:
            raise ValueError("SampledNetatmoAlmostFairKernelCRPSOnRadarGridLoss does not support alpha_scaler_name.")

        if self.distance_weighted:
            return self._forward_distance_weighted(
                y_pred,
                y_target,
                squash=squash,
                scaler_indices=scaler_indices,
                without_scalers=without_scalers,
                grid_shard_slice=grid_shard_slice,
                group=group,
                targets_by_dataset=targets_by_dataset,
                data_indices_by_dataset=data_indices_by_dataset,
            )

        source_values = self._source_values(targets_by_dataset, data_indices_by_dataset, y_pred.device)
        source_to_target_index = self.source_to_target_index.to(device=y_pred.device)
        source_to_target_valid = self.source_to_target_valid.to(device=y_pred.device)
        source_to_target_use_source = self.source_to_target_use_source.to(device=y_pred.device)
        station_indices = torch.nonzero(source_to_target_valid, as_tuple=False).squeeze(1)
        if station_indices.numel() == 0:
            raise ValueError("No valid Netatmo stations are mapped to the radar grid")

        target_indices = torch.index_select(source_to_target_index, 0, station_indices)
        use_source = torch.index_select(source_to_target_use_source, 0, station_indices)
        source_values = torch.index_select(source_values, 2, station_indices)
        y_pred = torch.index_select(y_pred, TensorDim.GRID, target_indices)
        if y_target.ndim == 5:
            y_target = y_target.select(TensorDim.ENSEMBLE_DIM, 0)
        if y_target.ndim != 4:
            raise ValueError(
                "SampledNetatmoAlmostFairKernelCRPSOnRadarGridLoss expects radar target shape "
                f"(batch, time, grid, variables), got {tuple(y_target.shape)}"
            )
        target_on_stations = torch.index_select(y_target, TensorDim.GRID - 1, target_indices)

        sampled_scaler_indices = scaler_indices
        if scaler_indices is None:
            sampled_scaler_indices = [slice(None)] * y_pred.ndim
        else:
            sampled_scaler_indices = []
            ellipsis_seen = False
            for item in scaler_indices:
                if item is Ellipsis:
                    if ellipsis_seen:
                        raise ValueError("Only one ellipsis is allowed in scaler_indices.")
                    ellipsis_seen = True
                    missing_dims = y_pred.ndim - (len(scaler_indices) - 1)
                    sampled_scaler_indices.extend([slice(None)] * missing_dims)
                else:
                    sampled_scaler_indices.append(item)
            if len(sampled_scaler_indices) != y_pred.ndim:
                raise ValueError("scaler_indices must resolve to the prediction tensor rank.")

        source_target = source_values.unsqueeze(-1).to(dtype=target_on_stations.dtype)
        if y_pred.shape[-1] != 1:
            source_target = source_target.expand(-1, -1, -1, y_pred.shape[-1])
        if self.source_distance_threshold_km is None:
            target_on_stations = source_target
        else:
            use_source = use_source.view(1, 1, -1, 1) & torch.isfinite(source_target)
            target_on_stations = torch.where(use_source, source_target, target_on_stations)

        y_target = einops.rearrange(target_on_stations, "bs t station v -> bs t v station")
        y_pred = einops.rearrange(y_pred, "bs t e station v -> bs t v station e")

        valid_target = torch.isfinite(target_on_stations)
        invalid_target = ~einops.rearrange(valid_target, "bs t station v -> bs t v station")
        y_target = y_target.masked_fill(invalid_target, 0.0)
        y_pred = y_pred.masked_fill(invalid_target.unsqueeze(-1), 0.0)

        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                kcrps_ = self._kernel_crps(y_pred, y_target, alpha=self.alpha)
        else:
            kcrps_ = self._kernel_crps(y_pred, y_target, alpha=self.alpha)

        kcrps_ = einops.rearrange(kcrps_, "bs t v station -> bs t 1 station v")
        invalid_loss = (~valid_target).unsqueeze(2)
        kcrps_ = kcrps_.masked_fill(invalid_loss, torch.nan if self.ignore_nans else 0.0)
        kcrps_ = self._scale_sampled(
            kcrps_,
            tuple(sampled_scaler_indices),
            target_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )

        loss = self.reduce(kcrps_, squash=squash, squash_mode="sum", group=None)
        return loss if group is None or grid_shard_slice is None else reduce_tensor(loss, group)

    def _forward_distance_weighted(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        squash: bool,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        targets_by_dataset: dict[str, torch.Tensor],
        data_indices_by_dataset,
    ) -> torch.Tensor:
        if not hasattr(self, "target_to_source_index") or not hasattr(self, "target_to_source_kernel"):
            raise ValueError("distance_weighted=True requires set_graph_data() before use")

        grid_size = int(y_pred.shape[TensorDim.GRID])
        sample_size = self._resolve_sample_size(grid_size)
        if sample_size < grid_size:
            target_indices = torch.randint(grid_size, (sample_size,), device=y_pred.device)
        else:
            target_indices = torch.arange(grid_size, device=y_pred.device)
        local_indices = torch.arange(sample_size, device=y_pred.device)

        if y_target.ndim == 5:
            y_target = y_target.select(TensorDim.ENSEMBLE_DIM, 0)
        if y_target.ndim != 4:
            raise ValueError(
                "SampledNetatmoAlmostFairKernelCRPSOnRadarGridLoss expects radar target shape "
                f"(batch, time, grid, variables), got {tuple(y_target.shape)}"
            )

        source_values = self._source_values(targets_by_dataset, data_indices_by_dataset, y_pred.device)
        confidence_values = self._source_values(
            targets_by_dataset,
            data_indices_by_dataset,
            y_pred.device,
            source_variable=self.source_confidence_variable,
        )

        target_to_source_index = self.target_to_source_index.to(device=y_pred.device)
        target_to_source_kernel = self.target_to_source_kernel.to(device=y_pred.device)
        source_indices = torch.index_select(target_to_source_index, 0, target_indices)
        distance_kernel = torch.index_select(target_to_source_kernel, 0, target_indices)

        y_pred = torch.index_select(y_pred, TensorDim.GRID, target_indices)
        radar_target = torch.index_select(y_target, TensorDim.GRID - 1, target_indices)
        source_target = torch.index_select(source_values, 2, source_indices).unsqueeze(-1).to(dtype=radar_target.dtype)
        confidence = torch.index_select(confidence_values, 2, source_indices).to(dtype=radar_target.dtype)

        confidence_valid = torch.isfinite(confidence)
        confidence_count = torch.where(confidence_valid, confidence.clamp_min(0.0), torch.zeros_like(confidence))
        source_confidence = self.neighbour_confidence_q_min + (1.0 - self.neighbour_confidence_q_min) * (
            1.0 - torch.exp(-confidence_count / self.neighbour_confidence_n0)
        )
        source_confidence = torch.where(
            confidence_valid,
            source_confidence,
            torch.full_like(source_confidence, self.neighbour_confidence_q_min),
        )

        lambda_weight = self.lambda_max * distance_kernel.to(dtype=radar_target.dtype).view(1, 1, -1) * source_confidence
        lambda_weight = lambda_weight.unsqueeze(-1)

        if y_pred.shape[-1] != 1:
            source_target = source_target.expand(-1, -1, -1, y_pred.shape[-1])
            lambda_weight = lambda_weight.expand_as(source_target)

        sampled_scaler_indices = scaler_indices
        if scaler_indices is None:
            sampled_scaler_indices = [slice(None)] * y_pred.ndim
        else:
            sampled_scaler_indices = []
            ellipsis_seen = False
            for item in scaler_indices:
                if item is Ellipsis:
                    if ellipsis_seen:
                        raise ValueError("Only one ellipsis is allowed in scaler_indices.")
                    ellipsis_seen = True
                    missing_dims = y_pred.ndim - (len(scaler_indices) - 1)
                    sampled_scaler_indices.extend([slice(None)] * missing_dims)
                else:
                    sampled_scaler_indices.append(item)
            if len(sampled_scaler_indices) != y_pred.ndim:
                raise ValueError("scaler_indices must resolve to the prediction tensor rank.")
        sampled_scaler_indices[TensorDim.GRID] = local_indices

        radar_valid = torch.isfinite(radar_target)
        source_valid = torch.isfinite(source_target)
        lambda_weight = torch.where(source_valid, lambda_weight, torch.zeros_like(lambda_weight))

        radar_target_crps = einops.rearrange(radar_target, "bs t point v -> bs t v point")
        source_target_crps = einops.rearrange(source_target, "bs t point v -> bs t v point")
        y_pred_crps = einops.rearrange(y_pred, "bs t e point v -> bs t v point e")
        lambda_weight = einops.rearrange(lambda_weight, "bs t point v -> bs t v point")
        radar_valid = einops.rearrange(radar_valid, "bs t point v -> bs t v point")
        source_valid = einops.rearrange(source_valid, "bs t point v -> bs t v point")

        radar_pred = y_pred_crps.masked_fill((~radar_valid).unsqueeze(-1), 0.0)
        radar_target_crps = radar_target_crps.masked_fill(~radar_valid, 0.0)
        source_pred = y_pred_crps.masked_fill((~source_valid).unsqueeze(-1), 0.0)
        source_target_crps = source_target_crps.masked_fill(~source_valid, 0.0)

        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                radar_kcrps = self._kernel_crps(radar_pred, radar_target_crps, alpha=self.alpha)
                source_kcrps = self._kernel_crps(source_pred, source_target_crps, alpha=self.alpha)
        else:
            radar_kcrps = self._kernel_crps(radar_pred, radar_target_crps, alpha=self.alpha)
            source_kcrps = self._kernel_crps(source_pred, source_target_crps, alpha=self.alpha)

        radar_weight = (1.0 - lambda_weight) * radar_valid.to(dtype=radar_kcrps.dtype)
        source_weight = lambda_weight * source_valid.to(dtype=source_kcrps.dtype)
        kcrps_ = radar_weight * radar_kcrps + source_weight * source_kcrps
        valid_loss = (radar_weight + source_weight) > 0.0

        kcrps_ = einops.rearrange(kcrps_, "bs t v point -> bs t 1 point v")
        invalid_loss = ~einops.rearrange(valid_loss, "bs t v point -> bs t 1 point v")
        kcrps_ = kcrps_.masked_fill(invalid_loss, torch.nan if self.ignore_nans else 0.0)
        kcrps_ = self._scale_sampled(
            kcrps_,
            tuple(sampled_scaler_indices),
            target_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )

        loss = self.reduce(kcrps_, squash=squash, squash_mode="sum", group=None)
        if self.rescale:
            loss = loss * (grid_size / sample_size)
        return loss if group is None or grid_shard_slice is None else reduce_tensor(loss, group)

    def _scale_sampled(
        self,
        x: torch.Tensor,
        subset_indices: tuple[int, ...],
        grid_indices: torch.Tensor,
        *,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        def apply_subset(tensor: torch.Tensor) -> torch.Tensor:
            out = tensor
            for dim, index in enumerate(subset_indices):
                if index is Ellipsis:
                    continue
                if isinstance(index, slice):
                    slices = [slice(None)] * out.ndim
                    slices[dim] = index
                    out = out[tuple(slices)]
                    continue
                index = torch.as_tensor(index, dtype=torch.long, device=out.device).reshape(-1)
                index = index % out.shape[dim]
                out = torch.index_select(out, dim, index)
            return out

        if len(self.scaler) == 0:
            return apply_subset(x)

        scale_tensor = self.scaler
        if without_scalers is not None and len(without_scalers) > 0:
            if isinstance(without_scalers[0], str):
                scale_tensor = self.scaler.without(without_scalers)
            else:
                scale_tensor = self.scaler.without_by_dim(without_scalers)

        out = apply_subset(x).clone()
        tensors = scale_tensor.resolve(x.ndim).tensors
        for dims, scaler in tensors.values():
            dims = list(dims)
            if TensorDim.GRID in dims:
                grid_index = dims.index(TensorDim.GRID)
                if grid_shard_slice is not None and scaler.shape[grid_index] >= grid_shard_slice.stop:
                    slices = [slice(None)] * len(dims)
                    slices[grid_index] = grid_shard_slice
                    scaler = scaler[tuple(slices)]
                scaler = torch.index_select(scaler, grid_index, grid_indices)

            missing_dims = [d for d in range(x.ndim) if d not in dims]
            reshape = [1] * len(missing_dims)
            reshape.extend(scaler.shape)

            reshaped_scaler = scaler.reshape(reshape)
            reshaped_scaler = torch.moveaxis(reshaped_scaler, list(range(x.ndim)), (*missing_dims, *dims))
            reshaped_scaler = reshaped_scaler.expand_as(x)
            out = out * apply_subset(reshaped_scaler)

        return out

    @property
    def name(self) -> str:
        return f"sampled_netatmo_station_afkcrps{self.alpha:.2f}_on_radar_grid"
