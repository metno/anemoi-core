# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Multi-scale (band-pass) loss with a per-band spread adjustment.

The band-pass
multi-scale loss of Lang, Leutbecher & Maciel (arXiv:2506.10868), as provided by
:class:`anemoi.training.losses.MultiscaleLossWrapper`, but with a *different* widening factor
``a_i`` per band, applied by the wrapped :class:`anemoi.training.spread.SpreadScaledCRPS`::

    L = Σ_i  w_i · afCRPS_{a_i}( B_i(x) , B_i(y) ),      B_1 = D_1,  B_i = D_i − D_{i−1},  B_n = I − D_{n−1}

(D_i are the smoothing operators, coarsest first; the telescoping differences are computed by
the parent ``forward``).  Because the widening is linear in the members and the band-pass
operators are linear, widening each band around its own band-mean equals widening the members
with the scale-dependent filter Σ_i a_i B_i.

Implementation strategy (chosen so that the 80-line
sharding-aware ``forward`` of the parent is *not* duplicated):

* one shared per-scale loss instance (as upstream);
* the parent ``forward`` calls ``self._smooth_for_loss(x, y, i)`` exactly once per band ``i``,
  sequentially and immediately before the per-band loss call, so we override that hook to set
  ``leaf.spread_adjustment = a_i`` on every :class:`SpreadScaledCRPS` leaf.

Loss-factory integration: ``anemoi.training.losses.loss.get_loss_function``
only builds the inner ``per_scale_loss`` for the hard-coded ``_target_``
``anemoi.training.losses.MultiscaleLossWrapper`` (``NESTED_LOSSES``).  Instead of patching the
factory, this class follows the ``CombinedLoss`` pattern: it declares
``factory_context_keys = {AVAILABLE_SCALERS, DATA_INDICES}`` so the factory hands it the full
scaler dict and the data indices, and it builds the inner loss itself with
``get_loss_function`` — which yields exactly the same object as upstream's nested path
(``LossVariableMapper(SpreadScaledCRPS)`` with the scalers listed in the per-scale config).
``graph_data`` / ``data_node_name`` are injected by the factory because
``needs_graph_data = True`` is inherited.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import LossFactoryContextKey
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.losses.multiscale import MultiscaleLossWrapper
from anemoi.training.losses.scaler_tensor import TENSOR_SPEC

from anemoi.training.spread.crps import SpreadScaledCRPS

LOGGER = logging.getLogger(__name__)


class ChainedProjectionProvider:
    """A band smoother given as a PRODUCT of file-backed sparse matrices (experimental).

    ``multiscale_config.loss_matrices`` accepts, in place of one filename, a list ``[M_1, M_2, ..., M_k]`` meaning the
    operator ``M_1 · M_2 · ... · M_k`` applied RIGHT TO LEFT (``M_k`` first). The n320 coarse band is
    ``[o96_to_n320_up.npz, o96_gauss_8h.npz, n320_to_o96_down.npz]`` = R_up · G_8h · R_down (within 3 % of the
    exact n320 Gaussian on smooth fields, σ_eff = the o96 8h kernel's), never formed explicitly (R_up · G_8h · R_down
    would have ~10⁹ non-zeros at n320). Each factor is an upstream ``ProjectionGraphProvider`` ([out_nodes, in_nodes]
    CSR); shapes must chain. ``projection_matrix`` is None, so the nnz-based band-order guard skips the chained band.
    """

    def __init__(self, providers: list) -> None:
        if len(providers) < 1:
            raise ValueError("ChainedProjectionProvider needs at least one matrix")
        shapes = [tuple(p.get_edges().shape) for p in providers]
        for (o1, i1), (o2, i2) in zip(shapes, shapes[1:], strict=False):
            if i1 != o2:
                msg = f"chained loss matrices do not compose: shapes {shapes} (each factor's in_nodes must equal the next one's out_nodes)"
                raise ValueError(msg)
        if shapes[0][0] != shapes[-1][1]:
            msg = f"chained loss matrix product must be square (grid -> grid), got {shapes[0][0]} x {shapes[-1][1]} from {shapes}"
            raise ValueError(msg)
        self.providers = list(providers)
        self.shapes = shapes
        self.projection_matrix = None

    def get_edges(self, device=None) -> list:  # noqa: ANN001
        """The factors' CSR matrices on ``device``, in CONFIG order (left to right)."""
        return [p.get_edges(device=device) for p in self.providers]


class MultiscaleSpreadLossWrapper(MultiscaleLossWrapper):
    """:class:`MultiscaleLossWrapper` with a per-band spread adjustment ``a_i``.

    Parameters
    ----------
    per_scale_loss : BaseLoss | dict | DictConfig
        The per-band loss.  When built through the anemoi loss factory this is the *config*
        of the per-scale loss (``_target_: anemoi.training.spread.SpreadScaledCRPS`` plus its
        ``scalers`` list); it is instantiated here with ``get_loss_function``.  A ready
        :class:`BaseLoss` instance is accepted too (tests / manual use).  At least one leaf
        must be a :class:`SpreadScaledCRPS`.
    weights : list[float]
        Per-band loss weights ζ_i, coarsest band first (as upstream).
    per_scale_spread_adjustment : Sequence[float]
        Per-band widening factors ``a_i``, coarsest band first; length must equal the number
        of bands (= number of smoothers + 1 in on-the-fly/graph mode).
    available_scalers, data_indices
        Loss-factory context (see module docstring).  ``data_indices`` is also used to resolve
        per-variable overrides of the leaf (``SpreadScaledCRPS.per_variable_spread_adjustment``)
        against ``data_indices.model.output.name_to_index``.
    **kwargs
        Forwarded to :class:`MultiscaleLossWrapper` (``multiscale_config``, ``graph_data``,
        ``data_node_name``, ``autocast``, ``sparse_projector_num_chunks``, ``ignore_nans``).
    """

    name: str = "MultiscaleSpreadLossWrapper"
    needs_graph_data: bool = True
    #: opt-in marker for ``BaseTrainingModule._compute_loss``: forward ``rollout_step``
    accepts_rollout_step: bool = True
    factory_context_keys = frozenset(
        {LossFactoryContextKey.AVAILABLE_SCALERS, LossFactoryContextKey.DATA_INDICES},
    )

    def __init__(
        self,
        per_scale_loss: BaseLoss | dict[str, Any] | DictConfig,
        weights: list[float],
        per_scale_spread_adjustment: Sequence[float],
        *,
        per_variable_band_spread_adjustment: dict[str, Sequence[float]] | None = None,
        per_rollout_step_spread_adjustment: Sequence[float] | None = None,
        available_scalers: dict[str, TENSOR_SPEC] | None = None,
        data_indices: IndexCollection | None = None,
        **kwargs,
    ) -> None:
        if isinstance(per_scale_loss, (DictConfig, dict)):
            per_scale_loss = get_loss_function(
                DictConfig(dict(per_scale_loss)),
                scalers=available_scalers,
                data_indices=data_indices,
                graph_data=kwargs.get("graph_data"),
                data_node_name=kwargs.get("data_node_name"),
            )
        super().__init__(per_scale_loss=per_scale_loss, weights=weights, **kwargs)

        self._spread_leaves: list[SpreadScaledCRPS] = [
            leaf for leaf in self.loss.iter_leaf_losses() if isinstance(leaf, SpreadScaledCRPS)
        ]
        if not self._spread_leaves:
            msg = (
                "MultiscaleSpreadLossWrapper requires a SpreadScaledCRPS per-scale loss, got "
                f"{type(self.loss).__name__} with leaves {[type(l).__name__ for l in self.loss.iter_leaf_losses()]}"
            )
            raise TypeError(msg)

        a = [float(v) for v in per_scale_spread_adjustment]
        if len(a) != self.num_scales:
            msg = (
                f"per_scale_spread_adjustment has {len(a)} entries but the wrapper has "
                f"{self.num_scales} bands (weights: {list(self.weights)})"
            )
            raise ValueError(msg)
        for i, value in enumerate(a):
            SpreadScaledCRPS._validate_a(value, what=f"per_scale_spread_adjustment[{i}]")
        self.per_scale_spread_adjustment: list[float] = a
        self._band_cursor: int | None = None

        if data_indices is not None:
            name_to_index = data_indices.model.output.name_to_index
            for leaf in self._spread_leaves:
                if leaf.has_variable_overrides:
                    leaf.resolve_variable_spread_adjustment(name_to_index)
                elif per_variable_band_spread_adjustment:
                    # the per-band composition rewrites the leaf's overrides each band; bind the
                    # index now so the first rewrite can validate names
                    leaf.resolve_variable_spread_adjustment(name_to_index)

        # Controller-facing factors: per-(variable, band)
        # multiplier table and per-(band, rollout-step) factor, composing MULTIPLICATIVELY with the
        # per-band scalar:
        #   a_eff(v, b, r) = a_b · m_{v,b} · a_{b,r}          (m and a_{b,r} default to 1)
        # A flat per-rollout-step sequence is accepted and broadcast across bands (one shared
        # factor per step). The leaf's *static* per-variable overrides (bounded variables pinned to 1,
        # absolute tables) always win — they are captured here and re-applied on top per
        # band, so a pinned variable stays at its configured value in every band and rollout step.
        self._static_overrides: list[dict[str, float]] = [
            leaf.per_variable_spread_adjustment for leaf in self._spread_leaves
        ]
        self._vb_table: dict[str, list[float]] = {}
        self._rollout_factors: list[list[float]] | None = None  # [band][0-based rollout step]
        self._current_rollout_step: int | None = None
        self._warned_rollout_clamp = False
        if per_variable_band_spread_adjustment:
            self.set_per_variable_band_table(per_variable_band_spread_adjustment)
        if per_rollout_step_spread_adjustment is not None:
            self.set_per_rollout_step_adjustment(per_rollout_step_spread_adjustment)

        # Per-band spread/skill diagnostics — metrics injected by TrainingDiagnosticsLossWrapper
        # (plain dict on purpose: an nn.ModuleDict attribute would double-register the outer
        # wrapper's metric modules). Filled by _smooth_for_loss, consumed after each forward.
        self._band_diag_metrics: dict[str, Any] | None = None
        self._band_diag_kwargs: dict[str, Any] = {}
        self._band_diag_active: bool = False
        self._band_diag_prev: tuple[Any, Any] | None = None
        self._band_diag_values: dict[str, Any] = {}
        self._warned_sharded_band_diag = False

        self._check_smoother_order()

        LOGGER.info(
            "MultiscaleSpreadLossWrapper: %d bands, weights=%s, per_scale_spread_adjustment=%s, "
            "per-variable overrides=%s",
            self.num_scales,
            list(self.weights),
            self.per_scale_spread_adjustment,
            {k: v for leaf in self._spread_leaves for k, v in leaf.per_variable_spread_adjustment.items()},
        )

    def _check_smoother_order(self) -> None:
        """Soft guard: bands must be coarsest first (upstream reverses an on-the-fly ``smoothers`` mapping,
        file mode does not) — a wrong order silently corrupts the band decomposition. Uses the number of
        non-zeros per smoother as a proxy for its width (a wider Gaussian needs more neighbours)."""
        nnz = []
        for provider in self.smoothing_matrices:
            if provider is None:
                nnz.append(0)
                continue
            mat = getattr(provider, "projection_matrix", None)
            try:
                nnz.append(int(mat._nnz()) if mat is not None else -1)
            except (AttributeError, RuntimeError):  # pragma: no cover - layout without _nnz
                nnz.append(-1)
        if -1 in nnz:
            return
        if any(a < b for a, b in zip(nnz, nnz[1:], strict=False)):
            LOGGER.warning(
                "MultiscaleSpreadLossWrapper: smoother nnz per band = %s is not decreasing coarse -> fine; check "
                "the smoother ordering (on-the-fly 'smoothers:' mapping must be listed FINEST first, file-mode "
                "'loss_matrices' COARSEST first). per_scale_spread_adjustment=%s is applied in band order.",
                nnz,
                self.per_scale_spread_adjustment,
            )

    # ------------------------------------------------------------------ controller API
    @property
    def static_variable_overrides(self) -> dict[str, float]:
        """The leaves' static per-variable overrides (pins) — excluded from adaptive control."""
        out: dict[str, float] = {}
        for static in self._static_overrides:
            out.update(static)
        return out

    def set_per_variable_band_table(self, table: dict[str, Sequence[float]]) -> None:
        """Replace the per-(variable, band) multiplier table ``m_{v,b}`` (live-updatable).

        Values compose multiplicatively with the per-band scalar and the per-rollout-step factor;
        variables with a *static* leaf override (pins) are unaffected. Band order coarsest first.
        """
        new: dict[str, list[float]] = {}
        for name, values in (table or {}).items():
            vals = [float(v) for v in values]
            if len(vals) != self.num_scales:
                msg = (
                    f"per_variable_band_spread_adjustment[{name!r}] has {len(vals)} entries, "
                    f"expected {self.num_scales} bands"
                )
                raise ValueError(msg)
            for b, v in enumerate(vals):
                SpreadScaledCRPS._validate_a(v, what=f"per_variable_band_spread_adjustment[{name!r}][{b}]")
            new[str(name)] = vals
        self._vb_table = new

    def get_per_variable_band_table(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._vb_table.items()}

    def set_per_rollout_step_adjustment(
        self,
        factors: Sequence[float] | Sequence[Sequence[float]] | None,
    ) -> None:
        """Replace the per-(band, rollout-step) factors ``a_{b,r}`` (live-updatable).

        Accepts either one sequence per band (outer length = ``num_scales``, equal inner lengths,
        inner index = 0-based rollout step) or a single flat sequence, which is broadcast to every
        band (the r3 shared-``a_r`` behaviour).
        """
        if factors is None:
            self._rollout_factors = None
            return
        rows = list(factors)
        if not rows:
            msg = "per_rollout_step_spread_adjustment must be a non-empty sequence (or None)"
            raise ValueError(msg)
        if isinstance(rows[0], (int, float)):
            rows = [list(rows)] * self.num_scales  # shared across bands
        if len(rows) != self.num_scales:
            msg = (
                f"per_rollout_step_spread_adjustment has {len(rows)} band rows, expected "
                f"{self.num_scales} (or one flat sequence shared across bands)"
            )
            raise ValueError(msg)
        table: list[list[float]] = []
        for b, row in enumerate(rows):
            vals = [float(v) for v in row]
            if not vals or len(vals) != len([float(v) for v in rows[0]]):
                msg = "per_rollout_step_spread_adjustment band rows must be non-empty and equal-length"
                raise ValueError(msg)
            for r, v in enumerate(vals):
                SpreadScaledCRPS._validate_a(v, what=f"per_rollout_step_spread_adjustment[band {b}][{r}]")
            table.append(vals)
        self._rollout_factors = table

    def get_per_rollout_step_adjustment(self) -> list[list[float]] | None:
        """The per-band factor rows ``a_{b,r}`` (always nested, even when set from a flat sequence)."""
        return [list(row) for row in self._rollout_factors] if self._rollout_factors is not None else None

    def _rollout_factor(self, band: int) -> float:
        if self._rollout_factors is None:
            return 1.0
        step = self._current_rollout_step
        if step is None:
            msg = (
                "per_rollout_step_spread_adjustment is configured but the loss was called without "
                "rollout_step — the anemoi-core patch forwarding rollout_step into _compute_loss "
                "is required; refusing to silently apply the wrong factor."
            )
            raise RuntimeError(msg)
        row = self._rollout_factors[band]
        if step >= len(row):
            if not self._warned_rollout_clamp:
                LOGGER.warning(
                    "MultiscaleSpreadLossWrapper: rollout_step %d beyond the %d configured "
                    "per-rollout-step factors; clamping to the last entry (logged once).",
                    step,
                    len(row),
                )
                self._warned_rollout_clamp = True
            step = len(row) - 1
        return row[step]

    # ------------------------------------------------------------------ per-band hook
    def _set_band_spread_adjustment(self, i: int) -> None:
        a_i = self.per_scale_spread_adjustment[i]
        if not self._vb_table and self._rollout_factors is None:
            # plain path: scalar per band only, leaf overrides untouched.
            for leaf in self._spread_leaves:
                leaf.spread_adjustment = a_i
            return
        a_r = self._rollout_factor(i)
        scaled = {name: a_i * m[i] * a_r for name, m in self._vb_table.items()}
        for leaf, static in zip(self._spread_leaves, self._static_overrides, strict=True):
            leaf.spread_adjustment = a_i * a_r
            leaf.per_variable_spread_adjustment = {**scaled, **static}

    # ------------------------------------------------------------------ per-band diagnostics
    def set_band_diagnostics(self, metrics: dict[str, Any] | None) -> None:
        """Install (or clear) the diagnostic metric modules used for per-band spread/skill.

        Called by :class:`anemoi.training.spread.TrainingDiagnosticsLossWrapper`; a plain dict so the
        modules stay registered only on their owner.
        """
        self._band_diag_metrics = dict(metrics) if metrics else None

    def consume_band_diagnostics(self) -> dict[str, Any]:
        """Return and clear the per-band diagnostic vectors of the last forward."""
        out, self._band_diag_values = self._band_diag_values, {}
        return out

    def _record_band_diagnostics(self, sx, sy, i: int) -> None:  # noqa: ANN001
        """Compute per-band ⟨s²⟩/⟨(x̄−y)²⟩ from the smoothed levels (band = successive difference).

        Runs under ``no_grad`` on detached tensors; reconstructs the same telescoping bands the
        parent forward computes right after this hook (unsharded path only — one extra subtraction,
        no extra smoothing).
        """
        import torch

        with torch.no_grad():
            sx = sx.detach()
            sy = sy.detach()
            if i == 0:
                band_x, band_y = sx, sy
            else:
                prev_x, prev_y = self._band_diag_prev
                band_x, band_y = sx - prev_x, sy - prev_y
            self._band_diag_prev = (sx, sy) if i < self.num_scales - 1 else None
            for name, metric in self._band_diag_metrics.items():
                value = metric(band_x, band_y, squash=False, **self._band_diag_kwargs)
                self._band_diag_values[f"{name}_band{i}"] = value.detach().float()

    def _load_file_smoothing_matrices(self, loss_matrices_path, loss_matrices):  # noqa: ANN001, ANN201 - parent signature
        """Parent file mode, plus list entries = chained products (:class:`ChainedProjectionProvider`)."""
        from omegaconf import ListConfig

        if not loss_matrices or not any(isinstance(e, (list, tuple, ListConfig)) for e in loss_matrices):
            return super()._load_file_smoothing_matrices(loss_matrices_path, loss_matrices)
        out = []
        for entry in loss_matrices:
            if isinstance(entry, (list, tuple, ListConfig)):
                factors = [super(MultiscaleSpreadLossWrapper, self)._load_file_smoothing_matrices(loss_matrices_path, [f])[0] for f in entry]
                if any(f is None for f in factors):
                    raise ValueError(f"a chained loss-matrix entry may not contain null: {list(entry)}")
                prov = ChainedProjectionProvider(factors)
                LOGGER.info("Loss smoothing: chained %s (applied right to left), shapes %s", list(entry), prov.shapes)
                out.append(prov)
            else:
                out.extend(super()._load_file_smoothing_matrices(loss_matrices_path, [entry]))
        return out

    def _project_chain(self, x, prov: ChainedProjectionProvider):  # noqa: ANN001, ANN201
        for mat in reversed(prov.get_edges(device=x.device)):
            x = self.projector(x, mat, num_chunks=self.sparse_projector_num_chunks)
        return x

    def _smooth_for_loss(self, x, y, i: int):  # noqa: ANN001, ANN201 - signature fixed by parent
        """Parent hook, called once per band ``i`` right before the per-band loss call.

        Sets the band's effective widening on the shared per-scale loss, then
        delegates the smoothing to the parent; optionally records per-band diagnostics.
        (Setting a Python float attribute is not part of the autograd graph; the widening
        itself happens inside ``SpreadScaledCRPS._kernel_crps`` and is differentiated.)
        """
        expected = 0 if self._band_cursor is None else self._band_cursor + 1
        if i != expected:
            msg = (
                f"MultiscaleSpreadLossWrapper: band hook called for band {i}, expected {expected}; the upstream "
                "MultiscaleLossWrapper.forward no longer visits bands sequentially - per-band 'a' would be misapplied."
            )
            raise RuntimeError(msg)
        self._band_cursor = i
        self._set_band_spread_adjustment(i)
        prov = self.smoothing_matrices[i]
        if isinstance(prov, ChainedProjectionProvider):
            sx, sy = self._project_chain(x, prov), self._project_chain(y, prov)
        else:
            sx, sy = super()._smooth_for_loss(x, y, i)
        if self._band_diag_active:
            self._record_band_diagnostics(sx, sy, i)
        return sx, sy

    def forward(self, *args, rollout_step: int | None = None, **kwargs):  # noqa: ANN002, ANN003, ANN201
        """Parent forward, guarded: every band must have been visited exactly once, in order.

        ``rollout_step`` (forwarded by the training module or the diagnostics wrapper) selects
        the per-rollout-step factor; band diagnostics are recorded when metrics are installed
        and the model is not grid-sharded (the smoothed levels seen by the hook are then
        channel-sharded and per-variable metrics would be wrong — skipped with a one-time warning).
        """
        self._band_cursor = None
        self._current_rollout_step = rollout_step
        sharded = kwargs.get("grid_shard_sizes") is not None
        self._band_diag_active = self._band_diag_metrics is not None and not sharded
        if self._band_diag_metrics is not None and sharded and not self._warned_sharded_band_diag:
            LOGGER.warning(
                "MultiscaleSpreadLossWrapper: per-band diagnostics are skipped for grid-sharded "
                "models (logged once).",
            )
            self._warned_sharded_band_diag = True
        if self._band_diag_active:
            self._band_diag_kwargs = {
                k: kwargs[k]
                for k in ("grid_shard_slice", "group", "pred_layout", "target_layout")
                if k in kwargs
            }
            self._band_diag_prev = None
            self._band_diag_values = {}
        try:
            out = super().forward(*args, **kwargs)
        finally:
            self._band_diag_prev = None
            self._band_diag_active = False
        if self._band_cursor != self.num_scales - 1:
            msg = (
                f"MultiscaleSpreadLossWrapper: forward visited bands up to {self._band_cursor}, expected "
                f"{self.num_scales - 1}; per-band 'a' would be misapplied."
            )
            raise RuntimeError(msg)
        return out
