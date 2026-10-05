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


class LightningBinaryCrossEntropyLoss(BaseLoss):
    """Binary cross-entropy for lightning occurrence above a count threshold."""

    name: str = "lightning_binary_cross_entropy"

    def __init__(
        self,
        threshold: float = 0.5,
        temperature: float = 0.5,
        from_logits: bool = True,
        reduction: str = "weighted",
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
        self.from_logits = bool(from_logits)
        if reduction not in {"weighted", "mean", "balanced_mean"}:
            raise ValueError("reduction must be one of ['weighted', 'mean', 'balanced_mean']")
        self.reduction = reduction
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

        target = target.unsqueeze(TensorDim.ENSEMBLE_DIM)
        target_active = (target > self.threshold).to(dtype=pred.dtype).expand_as(pred)
        if self.from_logits:
            out = F.binary_cross_entropy_with_logits(pred / self.temperature, target_active, reduction="none")
        else:
            pred_prob = pred.clamp(torch.finfo(pred.dtype).eps, 1.0 - torch.finfo(pred.dtype).eps)
            out = F.binary_cross_entropy(pred_prob, target_active, reduction="none")
        if self.ignore_nans:
            valid = torch.isfinite(target).expand_as(pred)
            out = out.masked_fill(~valid, 0.0)
        elif torch.isnan(target).any():
            valid = torch.isfinite(target).expand_as(pred)
            out = out.masked_fill(~valid, 0.0)
        else:
            valid = None

        if self.reduction == "balanced_mean":
            if scaler_indices is not None:
                out = out[scaler_indices]
                target_active = target_active[scaler_indices]
                if valid is not None:
                    valid = valid[scaler_indices]
            if valid is None:
                valid = torch.ones_like(target_active, dtype=torch.bool)

            positive = valid & (target_active > 0)
            negative = valid & (target_active <= 0)
            positive_count = positive.to(dtype=out.dtype).sum()
            negative_count = negative.to(dtype=out.dtype).sum()
            positive_sum = (torch.nansum if self.ignore_nans else torch.sum)(out.masked_fill(~positive, 0.0))
            negative_sum = (torch.nansum if self.ignore_nans else torch.sum)(out.masked_fill(~negative, 0.0))
            if grid_shard_slice is not None and group is not None:
                positive_sum = reduce_tensor(positive_sum, group)
                negative_sum = reduce_tensor(negative_sum, group)
                positive_count = reduce_tensor(positive_count, group)
                negative_count = reduce_tensor(negative_count, group)

            positive_weight = torch.as_tensor(self.false_negative_weight, dtype=out.dtype, device=out.device)
            negative_weight = torch.as_tensor(self.false_positive_weight, dtype=out.dtype, device=out.device)
            has_positive = (positive_count > 0).to(dtype=out.dtype)
            has_negative = (negative_count > 0).to(dtype=out.dtype)
            numerator = (
                positive_weight * has_positive * positive_sum / positive_count.clamp_min(1.0)
                + negative_weight * has_negative * negative_sum / negative_count.clamp_min(1.0)
            )
            denominator = positive_weight * has_positive + negative_weight * has_negative
            return numerator / denominator.clamp_min(torch.finfo(out.dtype).eps)

        weights = torch.where(
            target_active > 0,
            torch.as_tensor(self.false_negative_weight, dtype=out.dtype, device=out.device),
            torch.as_tensor(self.false_positive_weight, dtype=out.dtype, device=out.device),
        )
        out = out * weights

        if self.reduction == "mean":
            if scaler_indices is not None:
                out = out[scaler_indices]
                if valid is not None:
                    valid = valid[scaler_indices]
            if valid is None:
                denominator = torch.as_tensor(out.numel(), dtype=out.dtype, device=out.device)
            else:
                out = out.masked_fill(~valid, 0.0)
                denominator = valid.to(dtype=out.dtype).sum().clamp_min(1.0)
            numerator = (torch.nansum if self.ignore_nans else torch.sum)(out)
            if grid_shard_slice is not None and group is not None:
                numerator = reduce_tensor(numerator, group)
                denominator = reduce_tensor(denominator, group)
            return numerator / denominator

        out = self.scale(
            out,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(out, squash=squash, squash_mode="avg", group=group if grid_shard_slice is not None else None)


class LightningEnsembleBinaryCrossEntropyLoss(BaseLoss):
    """Binary cross-entropy on the ensemble-mean occurrence probability."""

    name: str = "lightning_ensemble_binary_cross_entropy"

    def __init__(
        self,
        threshold: float = 0.5,
        window_size: int = 1,
        temperature: float = 1.0,
        reduction: str = "mean",
        false_positive_weight: float = 1.0,
        false_negative_weight: float = 1.0,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if reduction not in {"mean", "balanced_mean"}:
            raise ValueError("reduction must be one of ['mean', 'balanced_mean']")
        self.threshold = float(threshold)
        self.window_size = int(window_size)
        self.temperature = float(temperature)
        self.reduction = reduction
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

        target = target.unsqueeze(TensorDim.ENSEMBLE_DIM)
        valid = torch.isfinite(target)
        target_active = (target > self.threshold).to(dtype=pred.dtype)
        member_probability = torch.sigmoid(pred / self.temperature)
        if self.window_size > 1:
            if pred.shape[TensorDim.TIME] < self.window_size:
                raise ValueError(
                    f"window_size={self.window_size} exceeds available output steps "
                    f"(pred={pred.shape[TensorDim.TIME]}, target={target.shape[TensorDim.TIME]})"
                )
            member_probability = 1.0 - torch.prod(
                1.0 - member_probability.unfold(
                    dimension=TensorDim.TIME,
                    size=self.window_size,
                    step=1,
                ),
                dim=-1,
            )
            target_active = target_active.unfold(
                dimension=TensorDim.TIME,
                size=self.window_size,
                step=1,
            ).amax(dim=-1)
            valid = valid.unfold(
                dimension=TensorDim.TIME,
                size=self.window_size,
                step=1,
            ).all(dim=-1)
        probability = member_probability.mean(dim=TensorDim.ENSEMBLE_DIM, keepdim=True)
        eps = torch.finfo(probability.dtype).eps
        out = F.binary_cross_entropy(
            probability.clamp(eps, 1.0 - eps),
            target_active,
            reduction="none",
        )
        if self.ignore_nans:
            out = out.masked_fill(~valid, 0.0)
        elif torch.isnan(target).any():
            out = out.masked_fill(~valid, 0.0)

        if self.reduction == "balanced_mean":
            if scaler_indices is not None:
                out = out[scaler_indices]
                target_active = target_active[scaler_indices]
                valid = valid[scaler_indices]

            positive = valid & (target_active > 0)
            negative = valid & (target_active <= 0)
            positive_count = positive.to(dtype=out.dtype).sum()
            negative_count = negative.to(dtype=out.dtype).sum()
            positive_sum = (torch.nansum if self.ignore_nans else torch.sum)(out.masked_fill(~positive, 0.0))
            negative_sum = (torch.nansum if self.ignore_nans else torch.sum)(out.masked_fill(~negative, 0.0))
            if grid_shard_slice is not None and group is not None:
                positive_sum = reduce_tensor(positive_sum, group)
                negative_sum = reduce_tensor(negative_sum, group)
                positive_count = reduce_tensor(positive_count, group)
                negative_count = reduce_tensor(negative_count, group)

            positive_weight = torch.as_tensor(self.false_negative_weight, dtype=out.dtype, device=out.device)
            negative_weight = torch.as_tensor(self.false_positive_weight, dtype=out.dtype, device=out.device)
            has_positive = (positive_count > 0).to(dtype=out.dtype)
            has_negative = (negative_count > 0).to(dtype=out.dtype)
            numerator = (
                positive_weight * has_positive * positive_sum / positive_count.clamp_min(1.0)
                + negative_weight * has_negative * negative_sum / negative_count.clamp_min(1.0)
            )
            denominator = positive_weight * has_positive + negative_weight * has_negative
            return numerator / denominator.clamp_min(torch.finfo(out.dtype).eps)

        weights = torch.where(
            target_active > 0,
            torch.as_tensor(self.false_negative_weight, dtype=out.dtype, device=out.device),
            torch.as_tensor(self.false_positive_weight, dtype=out.dtype, device=out.device),
        )
        out = out * weights
        out = out / out.shape[TensorDim.TIME]
        out = self.scale(
            out,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(
            out,
            squash=squash,
            squash_mode="avg",
            group=group if grid_shard_slice is not None else None,
        )

    @property
    def name(self) -> str:
        if self.window_size == 1:
            return "lightning_ensemble_binary_cross_entropy"
        return f"lightning_ensemble_binary_cross_entropy_{self.window_size}step"


class LightningStormAreaEnsembleBinaryCrossEntropyLoss(BaseLoss):
    """Block-occurrence ensemble BCE split into positive, nearby, and distant regions."""

    name: str = "lightning_storm_area_ensemble_binary_cross_entropy"

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        radius_px: int = 15,
        threshold: float = 0.0,
        temperature: float = 1.0,
        positive_weight: float = 1.0,
        near_negative_weight: float = 1.0,
        far_negative_weight: float = 0.1,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        if x_dim <= 0 or y_dim <= 0:
            raise ValueError("x_dim and y_dim must be positive")
        if radius_px < 0:
            raise ValueError("radius_px must be >= 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if min(positive_weight, near_negative_weight, far_negative_weight) < 0.0:
            raise ValueError("Storm-area loss weights must be non-negative")
        if positive_weight + near_negative_weight + far_negative_weight <= 0.0:
            raise ValueError("At least one storm-area loss weight must be positive")
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.radius_px = int(radius_px)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.positive_weight = float(positive_weight)
        self.near_negative_weight = float(near_negative_weight)
        self.far_negative_weight = float(far_negative_weight)
        self.supports_sharding = False

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
        del squash, group, kwargs
        if grid_shard_slice is not None:
            raise ValueError("LightningStormAreaEnsembleBinaryCrossEntropyLoss requires unsharded predictions")
        if pred.ndim != 5:
            raise ValueError(f"Expected pred shape (batch, time, ensemble, grid, vars), got {tuple(pred.shape)}")
        if target.ndim == 5:
            target = target.select(TensorDim.ENSEMBLE_DIM, 0)
        if target.ndim != 4:
            raise ValueError(f"Expected target shape (batch, time, grid, vars), got {tuple(target.shape)}")
        if pred.shape[TensorDim.GRID] != self.x_dim * self.y_dim:
            raise ValueError(
                f"Expected {self.y_dim}x{self.x_dim}={self.y_dim * self.x_dim} grid points, "
                f"got {pred.shape[TensorDim.GRID]}"
            )

        target = target.unsqueeze(TensorDim.ENSEMBLE_DIM)
        target_valid = torch.isfinite(target)
        pred_valid = torch.isfinite(pred)
        probability_by_member = 1.0 - torch.prod(
            1.0 - torch.sigmoid(pred / self.temperature).masked_fill(~pred_valid, 0.0),
            dim=TensorDim.TIME,
            keepdim=True,
        )
        probability = probability_by_member.mean(dim=TensorDim.ENSEMBLE_DIM, keepdim=True)
        target_active = ((target > self.threshold) & target_valid).any(dim=TensorDim.TIME, keepdim=True)
        valid = target_valid.any(dim=TensorDim.TIME, keepdim=True) & pred_valid.any(
            dim=TensorDim.TIME,
            keepdim=True,
        ).all(dim=TensorDim.ENSEMBLE_DIM, keepdim=True)
        target_active &= valid
        active_maps = target_active.permute(0, 1, 2, 4, 3).reshape(
            target.shape[0],
            1,
            1,
            target.shape[-1],
            self.y_dim,
            self.x_dim,
        )
        storm_seed = active_maps.squeeze(1)
        if self.radius_px > 0:
            storm_area = F.max_pool2d(
                storm_seed.reshape(-1, 1, self.y_dim, self.x_dim).to(dtype=pred.dtype),
                kernel_size=2 * self.radius_px + 1,
                stride=1,
                padding=self.radius_px,
            ).to(dtype=torch.bool)
            storm_area = storm_area.reshape(storm_seed.shape)
        else:
            storm_area = storm_seed
        storm_area = storm_area.reshape(target.shape[0], 1, 1, target.shape[-1], -1).permute(0, 1, 2, 4, 3)

        eps = torch.finfo(probability.dtype).eps
        point_loss = F.binary_cross_entropy(
            probability.clamp(eps, 1.0 - eps),
            target_active.to(dtype=probability.dtype),
            reduction="none",
        )
        regions = (
            (target_active, self.positive_weight),
            (valid & ~target_active & storm_area, self.near_negative_weight),
            (valid & ~storm_area, self.far_negative_weight),
        )
        result = point_loss.sum() * 0.0
        for region, weight in regions:
            region = region.to(dtype=point_loss.dtype)
            scaled_loss = self.scale(
                point_loss * region,
                scaler_indices,
                without_scalers=without_scalers,
                grid_shard_slice=None,
            )
            scaled_region = self.scale(
                region,
                scaler_indices,
                without_scalers=without_scalers,
                grid_shard_slice=None,
            )
            denominator = (torch.nansum if self.ignore_nans else torch.sum)(scaled_region)
            present = (denominator > 0).to(dtype=point_loss.dtype)
            result = result + weight * present * (torch.nansum if self.ignore_nans else torch.sum)(scaled_loss) / denominator.clamp_min(eps)
        return result / (self.positive_weight + self.near_negative_weight + self.far_negative_weight)


class LightningEnsembleSoftCSILoss(BaseLoss):
    """Soft CSI for block-occurrence ensemble probabilities."""

    name: str = "lightning_ensemble_soft_csi"

    def __init__(
        self,
        threshold: float = 0.0,
        temperature: float = 1.0,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.supports_sharding = False

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
        del squash, scaler_indices, without_scalers, group, kwargs
        if grid_shard_slice is not None:
            raise ValueError("LightningEnsembleSoftCSILoss requires unsharded predictions")
        if pred.ndim != 5:
            raise ValueError(f"Expected pred shape (batch, time, ensemble, grid, vars), got {tuple(pred.shape)}")
        if target.ndim == 5:
            target = target.select(TensorDim.ENSEMBLE_DIM, 0)
        if target.ndim != 4:
            raise ValueError(f"Expected target shape (batch, time, grid, vars), got {tuple(target.shape)}")

        target = target.unsqueeze(TensorDim.ENSEMBLE_DIM)
        target_valid = torch.isfinite(target)
        pred_valid = torch.isfinite(pred)
        probability_by_member = 1.0 - torch.prod(
            1.0 - torch.sigmoid(pred / self.temperature).masked_fill(~pred_valid, 0.0),
            dim=TensorDim.TIME,
            keepdim=True,
        )
        probability = probability_by_member.mean(dim=TensorDim.ENSEMBLE_DIM, keepdim=True)
        target_active = ((target > self.threshold) & target_valid).any(dim=TensorDim.TIME, keepdim=True)
        valid = target_valid.any(dim=TensorDim.TIME, keepdim=True) & pred_valid.any(
            dim=TensorDim.TIME,
            keepdim=True,
        ).all(dim=TensorDim.ENSEMBLE_DIM, keepdim=True)

        valid_float = valid.to(dtype=probability.dtype)
        target_float = target_active.to(dtype=probability.dtype)
        intersection = (probability * target_float * valid_float).sum(dim=TensorDim.GRID)
        target_sum = (target_float * valid_float).sum(dim=TensorDim.GRID)
        union = (probability * valid_float).sum(dim=TensorDim.GRID) + target_sum - intersection
        positive_case = target_sum > 0
        loss = 1.0 - intersection / union.clamp_min(torch.finfo(probability.dtype).eps)
        if not positive_case.any():
            return probability.sum() * 0.0
        return loss.masked_select(positive_case).mean()
