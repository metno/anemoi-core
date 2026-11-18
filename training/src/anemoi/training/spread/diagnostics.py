# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Train-side spread/skill diagnostics as a transparent wrapper around the training loss.

anemoi-training computes ``validation_metrics`` only in validation mode and exposes no
"training metrics" hook (``EnsembleTraining.compute_dataset_loss_metrics``
calls ``_compute_metrics`` only ``if validation_mode``; ``on_train_batch_end`` callbacks receive
just the loss).  The no-patch route is a *transparent* wrapper around the training loss:

* ``forward`` returns exactly the wrapped loss (bitwise — it is the very same module call), and
* on the same ``(y_pred_ens, y)`` tensors additionally evaluates
  :class:`~anemoi.training.spread.metrics.EnsembleVariance` and
  :class:`~anemoi.training.spread.metrics.EnsembleMeanSquaredError` per model-output variable
  (``squash=False``), area-weighted, under ``torch.no_grad()``, and stashes the detached vectors
  in :attr:`pending` for :class:`anemoi.training.spread.callbacks.SpreadSkillLogger` to drain and log.

Notes
-----
* The training loss sees **normalised** tensors (validation metrics see post-processed ones).
  For any *linear* normaliser (mean-std, std, max, none) both ⟨s²⟩ and ⟨(x̄−y)²⟩ of a variable
  scale by the same factor, so the per-variable SSR is identical in both spaces — which is what
  the train/val gap ``g`` needs.  Do not compare the raw train-side ⟨s²⟩ to the val-side one.
* Predictions arrive in ``MODEL_OUTPUT`` layout, targets in ``DATA_FULL`` layout; the metrics are
  built through the anemoi loss factory (with ``data_indices``) so they are
  ``LossVariableMapper``-wrapped and align the layouts themselves — the wrapper just forwards
  ``pred_layout`` / ``target_layout``.
* ``EnsembleTraining`` runs the loss region under ``torch.utils.checkpoint`` (non-reentrant), so
  ``forward`` executes twice per training step (forward + recompute in backward), and once per
  rollout step.  :attr:`pending` therefore receives ``2 × n_rollout_steps`` entries per training
  batch, in the order ``fwd_0 … fwd_{n-1}, recompute_{n-1} … recompute_0``; the logger keeps the
  first ``n = len(task.steps("training"))`` (identical values are recomputed).  Everything is
  ``.detach()``-ed, so the double execution is harmless.
* Loss-factory integration follows the ``CombinedLoss`` pattern (``factory_context_keys``): the
  wrapper receives the full scaler dict and ``data_indices`` and builds the wrapped loss with
  ``get_loss_function`` — i.e. it becomes byte-for-byte what the factory would have built at the
  top level (``LossVariableMapper(CRPS)``, ``MultiscaleSpreadLossWrapper``, …).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import BaseLossWrapper
from anemoi.training.losses.base import LossFactoryContextKey
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.losses.scaler_tensor import TENSOR_SPEC

LOGGER = logging.getLogger(__name__)

#: kwargs of the loss call that the (mapper-wrapped) diagnostic metrics understand
_DIAG_FORWARD_KWARGS = ("grid_shard_slice", "group", "pred_layout", "target_layout")

DEFAULT_DIAGNOSTICS: dict[str, str] = {
    "ensvar": "anemoi.training.spread.EnsembleVariance",
    "ensmse": "anemoi.training.spread.EnsembleMeanSquaredError",
}


class TrainingDiagnosticsLossWrapper(BaseLossWrapper):
    """Transparent loss wrapper that records per-variable ⟨s²⟩ and ⟨(x̄−y)²⟩ on every call.

    Parameters
    ----------
    loss : dict | DictConfig | BaseLoss
        Config of the actual training loss (built here with the anemoi loss factory) or a ready
        :class:`BaseLoss`.
    diagnostics : dict[str, str] | None
        ``{name: _target_}`` of the diagnostic metrics; default :data:`DEFAULT_DIAGNOSTICS`.
        Each is built as ``{_target_: ..., scalers: diagnostics_scalers}`` through the factory.
    diagnostics_scalers : list[str]
        Scalers for the diagnostic metrics; must not contain variable-dimension scalers
        (default ``['node_weights']`` — area weighting only, like the validation metrics).
    enabled : bool
        Master switch (``False`` → pure pass-through, nothing recorded).
    available_scalers, data_indices
        Loss-factory context (injected by ``get_loss_function``).
    **kwargs
        ``graph_data`` / ``data_node_name`` (injected because ``needs_graph_data = True``) and
        anything else meant for the wrapped loss' constructor is *not* accepted here — the
        wrapped loss is configured through its own config block.
    """

    needs_graph_data: bool = True
    #: opt-in marker for ``BaseTrainingModule._compute_loss``: forward ``rollout_step``
    accepts_rollout_step: bool = True
    factory_context_keys = frozenset(
        {LossFactoryContextKey.AVAILABLE_SCALERS, LossFactoryContextKey.DATA_INDICES},
    )

    def __init__(
        self,
        loss: dict[str, Any] | DictConfig | BaseLoss,
        *,
        diagnostics: dict[str, str] | None = None,
        diagnostics_scalers: list[str] | tuple[str, ...] = ("node_weights",),
        enabled: bool = True,
        available_scalers: dict[str, TENSOR_SPEC] | None = None,
        data_indices: IndexCollection | None = None,
        graph_data: object | None = None,
        data_node_name: str | None = None,
        **kwargs,
    ) -> None:
        if kwargs:
            msg = f"TrainingDiagnosticsLossWrapper got unexpected keyword argument(s) {sorted(kwargs)}"
            raise TypeError(msg)
        factory_kwargs = {
            "scalers": available_scalers,
            "data_indices": data_indices,
            "graph_data": graph_data,
            "data_node_name": data_node_name,
        }
        if isinstance(loss, (DictConfig, dict)):
            loss = get_loss_function(DictConfig(dict(loss)), **factory_kwargs)
        if not hasattr(loss, "scaler"):
            # container losses (CombinedLoss) delete their scaler — the children own theirs and
            # scaler traffic goes through the container's add_scaler; give BaseLossWrapper's
            # scaler-sharing an empty stand-in.
            from anemoi.training.losses.scaler_tensor import ScaleTensor

            loss.scaler = ScaleTensor()
        super().__init__(loss=loss)

        self.enabled = bool(enabled)
        self.diagnostics = torch.nn.ModuleDict()
        for name, target in (diagnostics or DEFAULT_DIAGNOSTICS).items():
            cfg = DictConfig({"_target_": target, "scalers": list(diagnostics_scalers)})
            self.diagnostics[name] = get_loss_function(cfg, **factory_kwargs)
        # never let a diagnostic metric's scaler alias the loss' scaler
        assert all(m.scaler is not self.scaler for m in self.diagnostics.values())

        self.variable_names: list[str] | None = None
        if data_indices is not None:
            n2i = data_indices.model.output.name_to_index
            self.variable_names = [name for name, _ in sorted(n2i.items(), key=lambda kv: kv[1])]
            # Single-scale SpreadScaledCRPS with per-variable overrides (bounded variables pinned to
            # a = 1) has no MultiscaleSpreadLossWrapper container to bind the names -> do it here. Idempotent
            # for leaves a container already resolved (same name_to_index; the loss sees model-output order).
            for leaf in self.spread_leaves:
                if leaf.has_variable_overrides:
                    leaf.resolve_variable_spread_adjustment(n2i)

        #: detached per-variable diagnostics recorded by forward; callbacks *peek* at batch end and
        #: clear at batch start (several callbacks may read the same entries). Capped so that a run
        #: without any consumer callback cannot grow it unboundedly. Entries hold tensor-valued
        #: metric vectors plus an optional non-tensor ``"_rollout_step"`` int.
        self.pending: list[dict[str, torch.Tensor]] = []
        self.max_pending: int = 64

        # Install the diagnostic metrics into a band-capable wrapped loss (MultiscaleSpread-
        # LossWrapper) so it records per-band spread/skill on the banded tensors it already forms.
        if self.enabled and hasattr(self.loss, "set_band_diagnostics"):
            self.loss.set_band_diagnostics(dict(self.diagnostics.items()))

    # ------------------------------------------------------------------ metadata
    @property
    def spread_leaves(self) -> list:
        """All :class:`~anemoi.training.spread.crps.SpreadScaledCRPS` leaves of the wrapped loss (may be empty)."""
        from anemoi.training.spread.crps import SpreadScaledCRPS  # local import: crps must not depend on this module

        leaves = self.loss.iter_leaf_losses() if hasattr(self.loss, "iter_leaf_losses") else [self.loss]
        return [leaf for leaf in leaves if isinstance(leaf, SpreadScaledCRPS)]

    @property
    def name(self) -> str:
        """Delegate to the wrapped loss (BaseLossWrapper does not)."""
        return getattr(self.loss, "name", self.loss.__class__.__name__.lower())

    def drain(self) -> list[dict[str, torch.Tensor]]:
        """Return and clear the pending diagnostics."""
        out, self.pending = self.pending, []
        return out

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        rollout_step: int | None = None,
        **kwargs,
    ) -> torch.Tensor:  # noqa: ANN003
        if rollout_step is not None and getattr(self.loss, "accepts_rollout_step", False):
            kwargs["rollout_step"] = rollout_step
        out = self.loss(pred, target, squash=squash, **kwargs)
        if self.enabled:
            diag_kwargs = {k: kwargs[k] for k in _DIAG_FORWARD_KWARGS if k in kwargs}
            with torch.no_grad():
                values = {
                    name: metric(pred.detach(), target.detach(), squash=False, **diag_kwargs).detach().float()
                    for name, metric in self.diagnostics.items()
                }
            if hasattr(self.loss, "consume_band_diagnostics"):
                values.update(self.loss.consume_band_diagnostics())  # {f"{name}_band{i}": vec}
            if rollout_step is not None:
                values["_rollout_step"] = int(rollout_step)  # authoritative step for the logger
            self.pending.append(values)
            if len(self.pending) > self.max_pending:
                del self.pending[: len(self.pending) - self.max_pending]
        return out
