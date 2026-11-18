# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spread/skill building blocks as :class:`BaseLoss` metrics.

Both metrics return *area-weighted means of squared quantities* so that the spread–skill ratio
can be assembled correctly **at aggregation time** (sqrt of means, never mean of sqrts):

    SSR = sqrt( (M+1)/M · ⟨s²⟩ / ⟨(x̄ − y)²⟩ )        (Fortin et al. 2014)

* :class:`EnsembleVariance`         → ⟨s²⟩ with the *unbiased* ensemble variance
  s² = 1/(M−1) Σ_i (x_i − x̄)²  (for M = 2: (x₁−x₂)²/2).  Accumulated in variance space — never
  log mean|x₁−x₂|/√2, which is biased low by √(2/π) ≈ 0.798 for Gaussian members.
* :class:`EnsembleMeanSquaredError` → ⟨(x̄ − y)²⟩ (squared space, not MAE).

Both are plain point-wise statistics of shape ``(bs, t, 1, latlon, v)`` fed through the standard
:meth:`BaseLoss.scale` (``node_weights`` for the area weighting, ``time_steps`` …) and
:meth:`BaseLoss.reduce` machinery, so they:

* work as ``validation_metrics`` (per variable through the ``training.metrics`` list, and per
  variable group / ``all``) — note that anemoi evaluates validation metrics in *post-processed
  (physical) space*, so ``all`` mixes units and only per-variable values are meaningful;
* can be evaluated on training batches (train-side SSR);
* are shard-safe: point-wise along the grid, reduced with the model comm group.

The ratio itself is **not** a per-batch metric — see :mod:`anemoi.training.spread.callbacks`.
"""

from __future__ import annotations

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.utils.enums import TensorDim


def _mask_nans_keep_ensemble(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-fill NaNs consistently across the ensemble (mirrors ``CRPS.mask_nans``)."""
    target_nan_mask = torch.isnan(target).any(dim=TensorDim.ENSEMBLE_DIM, keepdim=True)
    pred_nan_mask = torch.isnan(pred).any(dim=TensorDim.ENSEMBLE_DIM, keepdim=True)
    nan_mask = target_nan_mask | pred_nan_mask
    return pred.masked_fill(nan_mask, 0.0), target.masked_fill(nan_mask, 0.0)


class _EnsemblePointStatistic(BaseLoss):
    """Shared plumbing: point statistic → scale → reduce, computed in float32."""

    def __init__(self, ignore_nans: bool = False, no_autocast: bool = True) -> None:
        super().__init__(ignore_nans=ignore_nans)
        self.no_autocast = no_autocast
        self.supports_sharding = True

    def _point_statistic(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

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
        squash_mode: Squash_mode = "avg",
        **_kwargs,
    ) -> torch.Tensor:
        """Parameters follow :meth:`BaseLoss.forward`.

        ``y_pred``: ``(bs, t, e, latlon, v)``; ``y_target``: ``(bs, t, 1, latlon, v)``.
        """
        is_sharded = grid_shard_slice is not None
        if self.ignore_nans:
            y_pred, y_target = _mask_nans_keep_ensemble(y_pred, y_target)

        if self.no_autocast:
            with torch.amp.autocast(device_type=y_pred.device.type, enabled=False):
                out = self._point_statistic(y_pred.float(), y_target.float())
        else:
            out = self._point_statistic(y_pred, y_target)

        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        return self.reduce(out, squash=squash, squash_mode=squash_mode, group=group if is_sharded else None)


class EnsembleVariance(_EnsemblePointStatistic):
    """Unbiased ensemble variance s² = 1/(M−1) Σ_i (x_i − x̄)², point-wise, then area-weighted."""

    def _point_statistic(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        ens_size = y_pred.shape[TensorDim.ENSEMBLE_DIM]
        assert ens_size > 1, "EnsembleVariance needs at least two ensemble members."
        return torch.var(y_pred, dim=TensorDim.ENSEMBLE_DIM, unbiased=True, keepdim=True)

    @property
    def name(self) -> str:
        return "ensvar"


class EnsembleMeanSquaredError(_EnsemblePointStatistic):
    """Squared error of the ensemble mean (x̄ − y)², point-wise, then area-weighted."""

    def _point_statistic(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        ens_mean = torch.mean(y_pred, dim=TensorDim.ENSEMBLE_DIM, keepdim=True)
        return (ens_mean - y_target) ** 2

    @property
    def name(self) -> str:
        return "ensmse"


def spread_skill_ratio(mean_variance: float, mean_squared_error: float, ens_size: int) -> float:
    """SSR = sqrt((M+1)/M · ⟨s²⟩ / ⟨(x̄−y)²⟩) from *aggregated* means (Fortin et al. 2014).

    ``mean_variance`` and ``mean_squared_error`` must be means over the same sample of the two
    metrics above (same weights, same variables).  For a reliable ensemble the ratio is 1; the
    uncorrected ratio sqrt(⟨s²⟩/⟨(x̄−y)²⟩) of a reliable M = 2 ensemble is √(2/3) ≈ 0.816.
    """
    if ens_size < 2:
        msg = f"ens_size must be >= 2, got {ens_size}"
        raise ValueError(msg)
    if mean_squared_error <= 0.0:
        return float("nan")
    return float(((ens_size + 1) / ens_size * mean_variance / mean_squared_error) ** 0.5)
