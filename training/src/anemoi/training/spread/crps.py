# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spread-scaled (affinely widened) kernel CRPS.

The (almost-fair) kernel CRPS evaluated on the widened ensemble

    x̃_i = x̄ + a · (x_i − x̄),        x̄ = (1/M) Σ_j x_j,

i.e. every member is pushed away from the ensemble mean by the factor ``a``.  For a = 1 the
loss is *exactly* :class:`anemoi.training.losses.CRPS` (the widening branch is skipped, so the
result is bitwise identical).  For M = 2, with d = |x₁−x₂|/2 and ε = (1−α)/M, the loss reduces to

    L = ½ (|x̄ + a·d − y| + |x̄ − a·d − y|) − (1−ε) · a · |x₁−x₂| / 2 ,

the reference formula of the unit tests.  For Gaussian errors the expected loss is minimised at
the ensemble spread

    s*(a, ε)² = 2 (1−ε)² a² / [ (1+a²) (1 + a² − (1−ε)² a²) ]

(in units of the error variance), so a > 1 asks for a wider ensemble and a < 1 for a narrower one.
The score is proper only at a = 1: use the factor as a small, controlled correction (see
:class:`anemoi.training.spread.callbacks.CombinedAdaptiveSpreadController`), not as a remedy on
its own.

Design notes
------------
* The widening is a point-wise map along the ensemble dimension, so it is trivially shard-safe;
  the ensemble dimension is fully gathered before the loss in
  ``EnsembleTraining.compute_dataset_loss_metrics``.
* Tensor layout inside :meth:`_kernel_crps` is ``(bs, t, v, latlon, e)`` — the ensemble is the
  *last* dimension there (the public ``forward`` receives ``(bs, t, e, latlon, v)``).
* This class deliberately does **not** request ``data_indices`` from the loss factory: doing so
  would suppress the :class:`~anemoi.training.losses.variable_mapper.LossVariableMapper`
  wrapping that aligns the ``DATA_FULL``-layout targets with the ``MODEL_OUTPUT``-layout
  predictions.  Per-variable widening factors are therefore *resolved by a container* (see
  :class:`anemoi.training.spread.multiscale.MultiscaleSpreadLossWrapper`) or explicitly by the
  caller through :meth:`resolve_variable_spread_adjustment`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch

from anemoi.training.losses.kcrps import CRPS
from anemoi.training.losses.kcrps import CRPSBackend

LOGGER = logging.getLogger(__name__)


class SpreadScaledCRPS(CRPS):
    """(Almost-fair) kernel CRPS on the affinely widened ensemble ``x̄ + a (x − x̄)``.

    Parameters
    ----------
    spread_adjustment : float
        Default widening factor ``a`` (applied to every variable that has no per-variable
        override).  ``1.0`` recovers :class:`CRPS` exactly.  Must be > 0.
        A :class:`MultiscaleSpreadLossWrapper` re-sets this attribute band by band.
    per_variable_spread_adjustment : Mapping[str, float] | None
        Optional per-variable overrides ``{variable_name: a}`` (e.g. ``{tp: 1.0, tcc: 1.0}``
        to leave bounded variables un-widened).  Names are resolved to channel indices with
        :meth:`resolve_variable_spread_adjustment` (called by the container wrapper with
        ``data_indices.model.output.name_to_index``).  Using unresolved overrides in
        ``forward`` raises — nothing is silently ignored.
    **kwargs
        Forwarded to :class:`CRPS` (``alpha``, ``backend``, ``no_autocast``, ``ignore_nans``).
    """

    #: constructor kwargs accepted by the parent CRPS (its ``**kwargs`` silently swallows anything else,
    #: so we reject unknown keys here — a config typo must never silently disable the widening).
    _CRPS_KWARGS: frozenset[str] = frozenset({"alpha", "backend", "no_autocast", "ignore_nans"})

    def __init__(
        self,
        spread_adjustment: float = 1.0,
        per_variable_spread_adjustment: Mapping[str, float] | None = None,
        **kwargs,
    ) -> None:
        unknown = set(kwargs) - self._CRPS_KWARGS
        if unknown:
            msg = (
                f"SpreadScaledCRPS got unexpected keyword argument(s) {sorted(unknown)}; accepted: "
                f"spread_adjustment, per_variable_spread_adjustment, {sorted(self._CRPS_KWARGS)}"
            )
            raise TypeError(msg)
        super().__init__(**kwargs)
        self._a_default: float = 1.0
        self._a_overrides: dict[str, float] = {}
        self._a_index: dict[str, int] | None = None
        self._a_num_variables: int | None = None
        self._a_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

        self.spread_adjustment = spread_adjustment  # validated by the setter

        if per_variable_spread_adjustment:
            for name, value in per_variable_spread_adjustment.items():
                self._validate_a(value, what=f"per_variable_spread_adjustment[{name!r}]")
                self._a_overrides[str(name)] = float(value)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _validate_a(value: float, what: str = "spread_adjustment") -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"{what} must be a real number, got {value!r}"
            raise TypeError(msg)
        if not value > 0.0:
            msg = f"{what} must be > 0 (a = 1 recovers CRPS), got {value}"
            raise ValueError(msg)

    @property
    def spread_adjustment(self) -> float:
        """Default (scalar) widening factor ``a``."""
        return self._a_default

    @spread_adjustment.setter
    def spread_adjustment(self, value: float) -> None:
        self._validate_a(value)
        value = float(value)
        if value != self._a_default:
            self._a_cache.clear()
        self._a_default = value

    @property
    def per_variable_spread_adjustment(self) -> dict[str, float]:
        """Per-variable overrides (a copy)."""
        return dict(self._a_overrides)

    @per_variable_spread_adjustment.setter
    def per_variable_spread_adjustment(self, overrides: Mapping[str, float] | None) -> None:
        """Replace the per-variable overrides (used by adaptive controllers).

        Values are validated like the constructor's; if the names were already resolved, unknown
        names raise ``KeyError`` immediately (the index binding is kept, the cache is cleared).
        """
        new: dict[str, float] = {}
        for name, value in (overrides or {}).items():
            self._validate_a(value, what=f"per_variable_spread_adjustment[{name!r}]")
            new[str(name)] = float(value)
        if self._a_index is not None:
            missing = [n for n in new if n not in self._a_index]
            if missing:
                msg = f"per_variable_spread_adjustment names {missing} not found among the loss variables"
                raise KeyError(msg)
        self._a_overrides = new
        self._a_cache.clear()

    @property
    def variable_index(self) -> dict[str, int] | None:
        """Resolved ``{variable_name: channel_index}`` (``None`` until resolved)."""
        return dict(self._a_index) if self._a_index is not None else None

    @property
    def has_variable_overrides(self) -> bool:
        return bool(self._a_overrides)

    def resolve_variable_spread_adjustment(self, name_to_index: Mapping[str, int]) -> None:
        """Bind per-variable overrides to channel indices.

        Parameters
        ----------
        name_to_index : Mapping[str, int]
            Variable name → index along the variable dimension of the tensors this loss
            receives (for the anemoi loss factory: ``data_indices.model.output.name_to_index``).
            Overrides for names not present in the mapping raise ``KeyError`` (a silently
            ignored override is exactly the failure mode we want to avoid).
        """
        missing = [n for n in self._a_overrides if n not in name_to_index]
        if missing:
            msg = (
                f"per_variable_spread_adjustment names {missing} not found among the loss variables "
                f"{sorted(name_to_index)}"
            )
            raise KeyError(msg)
        self._a_index = {str(k): int(v) for k, v in name_to_index.items()}
        self._a_num_variables = (max(self._a_index.values()) + 1) if self._a_index else 0
        self._a_cache.clear()
        LOGGER.info(
            "%s: per-variable spread adjustment resolved (%d overrides over %d variables)",
            self.__class__.__name__,
            len(self._a_overrides),
            self._a_num_variables,
        )

    def _effective_spread_adjustment(self, preds: torch.Tensor) -> float | torch.Tensor:
        """Return ``a`` as a float, or a ``(1, 1, v, 1, 1)`` tensor when overrides exist.

        ``preds`` has the internal layout ``(bs, t, v, latlon, e)``.
        """
        if not self._a_overrides:
            return self._a_default
        if self._a_index is None:
            msg = (
                "per_variable_spread_adjustment set but variable names were never resolved; call "
                "resolve_variable_spread_adjustment(name_to_index) (done automatically by "
                "MultiscaleSpreadLossWrapper when built through the anemoi loss factory)."
            )
            raise RuntimeError(msg)
        n_vars = preds.shape[2]
        if n_vars != self._a_num_variables:
            msg = (
                f"per-variable spread adjustment was resolved for {self._a_num_variables} variables but the "
                f"loss received {n_vars}; per-variable widening requires the loss to see the full model "
                "output (do not combine with predicted_variables/target_variables filtering)."
            )
            raise RuntimeError(msg)
        key = (preds.device, preds.dtype)
        a = self._a_cache.get(key)
        if a is None:
            a = torch.full((n_vars,), self._a_default, dtype=preds.dtype, device=preds.device)
            for name, value in self._a_overrides.items():
                a[self._a_index[name]] = value
            a = a.view(1, 1, n_vars, 1, 1)
            self._a_cache[key] = a
        return a

    @staticmethod
    def widen(preds: torch.Tensor, a: float | torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Affine widening ``x̄ + a (x − x̄)`` along ``dim``."""
        mean = preds.mean(dim=dim, keepdim=True)
        return mean + a * (preds - mean)

    # ------------------------------------------------------------------ CRPS hook
    def _kernel_crps(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        alpha: float | None = None,
        backend: CRPSBackend | None = None,
    ) -> torch.Tensor:
        """Widen the ensemble (last dim of ``preds``) then evaluate the parent kernel CRPS."""
        a = self._effective_spread_adjustment(preds)
        # Skip the (identity) transform for the scalar a == 1 case so that
        # SpreadScaledCRPS(a=1) == CRPS bitwise.
        if not (isinstance(a, float) and a == 1.0):
            preds = self.widen(preds, a, dim=-1)
        return super()._kernel_crps(preds, targets, alpha=alpha, backend=backend)

    @property
    def name(self) -> str:
        return f"sscrps{self.alpha:.2f}a{self._a_default:.2f}"
