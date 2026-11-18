# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Target-noise training: the training loss is evaluated against perturbed targets.

During **training only**, the loss is evaluated against

    y ← y + η,      η_v = A_v(r) · w_v(x) · (D ξ_v) / ‖D‖_row,      ξ_v ~ N(0, I) white on the grid,

per model-output variable ``v``: ``D`` is a (row-normalised) smoothing matrix — e.g. the coarse
Gaussian of the multiscale loss, so the noise is spatially coherent at that scale — and the
division by the row L2 norm ``sqrt(Σ_j D_ij²)`` re-normalises the coloured field to **unit marginal
variance** at every node, so that ``A_v`` *is* the marginal standard deviation of the perturbation
(normalised units).  ``A_v(r)`` is read from one JSON table per rollout step ``r`` (the last table
is reused beyond the list); an optional dose map ``w_v(x)`` with area mean ``w² = 1``
redistributes the amplitude in space.  Optional linear annealing to 0 over ``anneal_steps``.

Purpose: an ensemble trained with a proper score calibrates its spread to the error it meets on
its training samples.  Where that error is smaller than the error on unseen data — the model has
partly memorised its training set — the ensemble is under-dispersed out of sample.  Target noise
of variance ``A_v² = c · MSE_out − MSE_in`` (squared error of the ensemble mean out of sample and
in sample, ``c`` the ratio of in-sample to out-of-sample ensemble variance) raises the error the
model sees to the level it will meet, and leaves the score untouched.

Why a training-*method* subclass: the task's ``get_targets`` knows neither the mode (train/val),
the rollout/global step, the data indices, nor the ensemble-comm group, all of which the noise
needs; ``EnsembleTraining.compute_dataset_loss_metrics`` has them.

Determinism / resampling: the white noise comes from a private generator seeded from
``(seed, global_step, rollout_step, ens_comm_group_id)``, so (i) the forward pass and the
``torch.utils.checkpoint`` recompute in backward see the same η, and (ii) all ranks of an ensemble
group (which hold the same batch, possibly different grid shards) draw the same full-grid ξ and
slice their own shard — the coloured field is consistent across shards.

**The noise is resampled at every visit.**  The seed depends only on *when* a batch is processed
(global step / rollout step / group), never on the sample's identity or content — a training
sample revisited in a later epoch necessarily lands on a different global_step and gets an
independent η.  Noise fixed per sample would itself be memorised and defeat the purpose.

Config::

    training:
      method:
        _target_: anemoi.training.spread.target_noise.TargetNoiseEnsembleTraining
      target_noise:
        enabled: True
        smoother: <path>/gauss_8x.npz              # scipy .npz CSR (n_grid, n_grid), rows = targets
        amplitude_table:                           # one {var: amplitude} JSON per rollout step
          - <path>/target_noise_step1.json
          - <path>/target_noise_step2.json
        amplitude_fraction: 1.0
        dose_map: <path>/dose_map.npz              # optional
        include_variables: [z_500, msl, sp]        # or exclude_variables
        anneal_steps: null
        seed: 42

Caveat: train-side ⟨(x̄−y)²⟩ diagnostics are computed against the *noised* targets (they include
``A_v²``); use the validation-side SSR (clean targets) as the reference.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.train.methods.ensemble import EnsembleTraining
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


def load_amplitude_table(source: str | Path | Mapping[str, float]) -> dict[str, float]:
    """``{variable: RMSE}`` from a mapping or a JSON file (flat, or with an ``"rmse"`` sub-dict)."""
    if isinstance(source, Mapping):
        table: Any = dict(source)
    else:
        table = json.loads(Path(source).read_text())
    if isinstance(table, dict) and "rmse" in table and isinstance(table["rmse"], dict):
        table = table["rmse"]
    out: dict[str, float] = {}
    for k, v in table.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    if not out:
        msg = f"amplitude table {source!r} contains no {{variable: value}} entries"
        raise ValueError(msg)
    return out


class TargetNoise:
    """Coloured, per-variable target perturbation generator (see module docstring).

    Parameters
    ----------
    smoother : str | Path | scipy.sparse.spmatrix
        Row-normalised smoothing matrix ``D`` of shape ``(n_grid, n_grid)``.
    amplitudes_full : torch.Tensor
        ``(n_full,)`` marginal std of the perturbation per DATA_FULL channel (0 = untouched), or
        ``(R, n_full)`` for PER-ROLLOUT-STEP amplitudes (row r = 0-based rollout step, clamp-to-last
        beyond).
    anneal_steps : int | None
        Linear decay of the amplitude to 0 over the first ``anneal_steps`` optimizer steps.
    seed : int
        Base seed of the private generator.
    dose_map : torch.Tensor | None
        ``(n_grid, n_full)`` static spatial dose map ``w_v(x)``: the perturbation of
        channel v at node x is multiplied by ``w_v(x)``; built from the control's local climatological ensemble-mean
        error normalised to an area-weighted mean of ``w²`` of 1, so the global-mean noise variance is unchanged and
        the amplitude table stays the amplitude authority.  ``None`` (or all ones) = today's spatially uniform dose.
    """

    def __init__(
        self,
        smoother: str | Path | sp.spmatrix,
        amplitudes_full: torch.Tensor,
        *,
        anneal_steps: int | None = None,
        seed: int = 0,
        dose_map: torch.Tensor | None = None,
    ) -> None:
        m = sp.load_npz(smoother) if isinstance(smoother, (str, Path)) else smoother
        m = sp.csr_matrix(m, dtype=np.float32)
        if m.shape[0] != m.shape[1]:
            msg = f"smoother must be square (n_grid, n_grid), got {m.shape}"
            raise ValueError(msg)
        self.n_grid = int(m.shape[0])
        row_l2 = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        if not np.all(row_l2 > 0):
            msg = "smoother has empty rows; every target node needs at least one weight"
            raise ValueError(msg)
        # int32 indices halve the index memory of the (~33M nnz) 8h matrix on the GPU; `_on` falls back to
        # int64 if the device's sparse.mm rejects them.
        self._crow = torch.from_numpy(m.indptr.astype(np.int32))
        self._col = torch.from_numpy(m.indices.astype(np.int32))
        self._vals = torch.from_numpy(m.data.astype(np.float32))
        self._csr_cpu = torch.sparse_csr_tensor(self._crow, self._col, self._vals, size=(self.n_grid, self.n_grid))
        self._inv_row_l2_cpu = torch.from_numpy((1.0 / row_l2).astype(np.float32))
        amps = amplitudes_full.detach().float().clone()
        self._amps_steps = amps.unsqueeze(0) if amps.dim() == 1 else amps  # (R, n_full)
        self.n_amp_steps = int(self._amps_steps.shape[0])
        self.amplitudes_full = self._amps_steps[0]  # step-0 row (back-compat accessor)
        self.anneal_steps = None if anneal_steps is None else int(anneal_steps)
        self.seed = int(seed)
        self._device_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # active = union over steps, so a variable dosed only at some steps still gets its noise there
        self.active_full_idx = torch.nonzero((self._amps_steps > 0).any(dim=0), as_tuple=False).reshape(-1)
        self._dose_map_cpu: torch.Tensor | None = None
        if dose_map is not None:
            dm = dose_map.detach().float().clone()
            if dm.shape != (self.n_grid, self._amps_steps.shape[1]):
                msg = f"dose_map must be (n_grid, n_full) = ({self.n_grid}, {self._amps_steps.shape[1]}), got {tuple(dm.shape)}"
                raise ValueError(msg)
            if not torch.all(dm >= 0):
                msg = "dose_map must be non-negative"
                raise ValueError(msg)
            if not torch.all(dm == 1.0):       # all-ones = no map (bitwise today's behaviour)
                self._dose_map_cpu = dm
        self._dose_map_dev: dict[torch.device, torch.Tensor] = {}

    # ------------------------------------------------------------------ helpers
    def _on(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if device not in self._device_cache:
            csr = self._csr_cpu.to(device)
            try:  # self-test the sparse kernel with int32 indices on this device
                torch.sparse.mm(csr, torch.zeros(self.n_grid, 1, device=device))
            except RuntimeError as err:  # pragma: no cover - device dependent
                LOGGER.warning("TargetNoise: int32 CSR sparse.mm unsupported on %s (%s); using int64 indices", device, err)
                csr = torch.sparse_csr_tensor(self._crow.long(), self._col.long(), self._vals, size=(self.n_grid, self.n_grid)).to(device)
            self._device_cache[device] = (csr, self._inv_row_l2_cpu.to(device), self._amps_steps.to(device))
        return self._device_cache[device]

    def amplitude_factor(self, global_step: int) -> float:
        if self.anneal_steps is None or self.anneal_steps <= 0:
            return 1.0
        return max(0.0, 1.0 - float(global_step) / float(self.anneal_steps))

    @staticmethod
    def make_seed(seed: int, global_step: int, rollout_step: int, group_id: int) -> int:
        return (int(seed) * 1_000_003 + int(global_step) * 10_007 + int(rollout_step) * 101 + int(group_id)) % (2**63 - 1)

    def coloured_unit_noise(self, n_fields: int, *, device: torch.device, seed: int) -> torch.Tensor:
        """``(n_grid, n_fields)`` fields with unit marginal variance at every node."""
        csr, inv_row_l2, _ = self._on(device)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        xi = torch.randn(self.n_grid, n_fields, generator=gen, device=device, dtype=torch.float32)
        return torch.sparse.mm(csr, xi) * inv_row_l2[:, None]

    # ------------------------------------------------------------------ main entry
    def perturb(
        self,
        y: torch.Tensor,
        *,
        global_step: int,
        rollout_step: int,
        group_id: int,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        """Return ``y + η`` for a DATA_FULL-layout target ``(bs, t, e, grid|shard, n_full)``."""
        factor = self.amplitude_factor(global_step)
        if factor <= 0.0 or self.active_full_idx.numel() == 0:
            return y
        device = y.device
        _, _, amp = self._on(device)
        active = self.active_full_idx.to(device)
        n_grid_local = y.shape[TensorDim.GRID]
        if grid_shard_slice is None:
            if n_grid_local != self.n_grid:
                msg = f"target has {n_grid_local} grid points but the smoother is ({self.n_grid}, {self.n_grid})"
                raise ValueError(msg)
            rows = slice(0, self.n_grid)
        else:
            rows = grid_shard_slice
            if (rows.stop - rows.start) != n_grid_local or rows.stop > self.n_grid:
                msg = f"grid_shard_slice {rows} inconsistent with target grid {n_grid_local} / smoother {self.n_grid}"
                raise ValueError(msg)
        bs, t, e = y.shape[0], y.shape[1], y.shape[2]
        n_out = int(active.numel())
        seed = self.make_seed(self.seed, global_step, rollout_step, group_id)
        field = self.coloured_unit_noise(bs * t * n_out, device=device, seed=seed)  # (n_grid, bs*t*n_out)
        field = field[rows]  # (n_shard, bs*t*n_out)
        eta = field.reshape(n_grid_local, bs, t, n_out).permute(1, 2, 0, 3)  # (bs, t, n_shard, n_out)
        eta = eta.unsqueeze(2).expand(bs, t, e, n_grid_local, n_out)  # same perturbation for every target "member"
        amp_row = amp[min(int(rollout_step), self.n_amp_steps - 1)]  # per-step amplitudes, clamp-to-last
        eta = eta * (amp_row[active] * factor)
        if self._dose_map_cpu is not None:
            if device not in self._dose_map_dev:
                self._dose_map_dev[device] = self._dose_map_cpu.to(device)
            eta = eta * self._dose_map_dev[device][rows][:, active]  # (n_shard, n_out) broadcast over (bs, t, e)
        y = y.clone()
        y[..., active] = y[..., active] + eta.to(y.dtype)
        return y


def build_target_noise(
    cfg: Mapping[str, Any] | DictConfig,
    *,
    data_indices: Any,
) -> TargetNoise:
    """Build a :class:`TargetNoise` from the ``training.target_noise`` block and the dataset's data indices."""
    cfg = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    known = {"enabled", "smoother", "amplitude_table", "amplitude_fraction", "exclude_variables", "include_variables", "anneal_steps", "seed", "dose_map"}
    unknown = sorted(set(cfg) - known)
    if unknown:
        msg = f"training.target_noise: unknown keys {unknown}; known: {sorted(known)}"
        raise TypeError(msg)
    fraction = float(cfg.get("amplitude_fraction", 0.25))
    if not fraction >= 0.0:
        msg = f"amplitude_fraction must be >= 0, got {fraction}"
        raise ValueError(msg)
    raw_tables = cfg["amplitude_table"]
    step_tables = raw_tables if isinstance(raw_tables, (list, tuple)) else [raw_tables]
    tables = [load_amplitude_table(t) for t in step_tables]  # one per 0-based rollout step (clamp-to-last)
    table = tables[0]
    exclude = set(cfg.get("exclude_variables") or [])
    include = cfg.get("include_variables")
    out_index = data_indices.data.output  # DATA_FULL index space; .ordered_names = the model-output variables
    name_to_index = dict(out_index.name_to_index)  # ALL variables (forcings included) -> DATA_FULL channel
    out_names = list(out_index.ordered_names)
    if include is not None:
        # perturb ONLY these variables
        # (amplitude 0 elsewhere). Mutually exclusive with exclude_variables; unknown names are an error.
        if exclude:
            msg = "training.target_noise: give include_variables OR exclude_variables, not both"
            raise ValueError(msg)
        include = set(include)
        unknown_inc = sorted(include - set(out_names))
        if unknown_inc:
            msg = f"training.target_noise.include_variables not model outputs: {unknown_inc}"
            raise KeyError(msg)
        exclude = set(out_names) - include
    n_full = max(name_to_index.values()) + 1
    for r, table in enumerate(tables):
        missing = sorted(v for v in out_names if v not in table and v not in exclude)
        if missing:
            msg = f"amplitude table [step {r}] lacks {len(missing)} model-output variables: {missing[:12]}{'…' if len(missing) > 12 else ''}"
            raise KeyError(msg)
    amps = torch.zeros(len(tables), n_full, dtype=torch.float32)
    for r, table in enumerate(tables):
        for name in out_names:
            if name in exclude:
                continue
            amps[r, name_to_index[name]] = fraction * table[name]
    if amps.shape[0] == 1:
        amps = amps[0]
    dose_map = None
    if cfg.get("dose_map"):
        dose_map = load_dose_map(cfg["dose_map"], name_to_index=name_to_index, active_names=[v for v in out_names if v not in exclude], n_full=n_full)
    return TargetNoise(cfg["smoother"], amps, anneal_steps=cfg.get("anneal_steps"), seed=int(cfg.get("seed", 0)), dose_map=dose_map)


def load_dose_map(source: str | Path | Mapping[str, Any], *, name_to_index: Mapping[str, int], active_names: list[str], n_full: int) -> torch.Tensor:
    """``(n_grid, n_full)`` dose map from a ``jobs/dose_map.py`` .npz (``names`` (nvar), ``w`` (nvar, n_grid), ``lat``) or an
    in-memory mapping ``{"names": [...], "w": array, "lat": array}``; every perturbed variable must have a map (KeyError
    otherwise), inactive channels get 1.  Logs the area-weighted mean of ``w²`` per variable (should be 1 ± 2 %)."""
    z = dict(source) if isinstance(source, Mapping) else dict(np.load(source, allow_pickle=False))
    names = [str(n) for n in np.asarray(z["names"]).tolist()]
    w = np.asarray(z["w"], dtype=np.float32)
    if w.ndim != 2 or w.shape[0] != len(names):
        msg = f"dose_map: w must be (nvar, n_grid) with nvar = len(names) = {len(names)}, got {w.shape}"
        raise ValueError(msg)
    missing = sorted(v for v in active_names if v not in names)
    if missing:
        msg = f"dose_map lacks {len(missing)} perturbed variables: {missing[:12]}{'…' if len(missing) > 12 else ''}"
        raise KeyError(msg)
    n_grid = w.shape[1]
    out = torch.ones(n_grid, n_full, dtype=torch.float32)
    area = None
    if "lat" in z:
        area = np.cos(np.radians(np.asarray(z["lat"], dtype=np.float64)))
        area /= area.sum()
    worst = 0.0
    for v in active_names:
        col = w[names.index(v)]
        out[:, name_to_index[v]] = torch.from_numpy(col)
        if area is not None:
            worst = max(worst, abs(float((area * col.astype(np.float64) ** 2).sum()) - 1.0))
    if worst > 0.02:
        LOGGER.warning("dose_map: area-weighted mean of w^2 deviates from 1 by up to %.3f — the global-mean dose is not preserved", worst)
    LOGGER.info("dose_map: %d perturbed variables mapped on %d nodes (w range %.2f–%.2f)", len(active_names), n_grid, float(w.min()), float(w.max()))
    return out


class TargetNoiseEnsembleTraining(EnsembleTraining):
    """:class:`EnsembleTraining` whose *training* loss sees ``y + η``. Validation is untouched."""

    def __init__(self, *, config: DictConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        tn_cfg = getattr(config.training, "target_noise", None)
        if tn_cfg is None:
            msg = "TargetNoiseEnsembleTraining requires a training.target_noise config block"
            raise ValueError(msg)
        self.target_noise_enabled = bool(OmegaConf.select(config, "training.target_noise.enabled", default=True))
        self._target_noise_cfg = tn_cfg
        self.target_noise: dict[str, TargetNoise] = {}
        if self.target_noise_enabled:
            for dataset_name, indices in self.data_indices.items():
                self.target_noise[dataset_name] = build_target_noise(tn_cfg, data_indices=indices)
                tn = self.target_noise[dataset_name]
                LOGGER.info(
                    "TargetNoise[%s]: %d/%d output variables perturbed, mean amplitude %.4f (normalised units), "
                    "anneal_steps=%s, seed=%d, smoother n=%d",
                    dataset_name,
                    int(tn.active_full_idx.numel()),
                    len(indices.data.output.ordered_names),
                    float(tn.amplitudes_full[tn.active_full_idx].mean()) if tn.active_full_idx.numel() else 0.0,
                    tn.anneal_steps,
                    tn.seed,
                    tn.n_grid,
                )
        else:
            LOGGER.warning("TargetNoiseEnsembleTraining: target noise DISABLED by config (enabled: False)")

    def compute_dataset_loss_metrics(  # noqa: D102 - see EnsembleTraining
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        dataset_name: str,
        rollout_step: int | None = None,
        validation_mode: bool = False,
        **kwargs,
    ):
        if self.target_noise_enabled and not validation_mode and self.training and dataset_name in self.target_noise:
            y = self.target_noise[dataset_name].perturb(
                y,
                global_step=int(self.global_step),
                rollout_step=int(rollout_step or 0),
                group_id=int(getattr(self, "ens_comm_group_id", 0) or 0),
                grid_shard_slice=self.grid_shard_slice.get(dataset_name) if isinstance(self.grid_shard_slice, dict) else None,
            )
        return super().compute_dataset_loss_metrics(
            y_pred,
            y,
            dataset_name,
            rollout_step=rollout_step,
            validation_mode=validation_mode,
            **kwargs,
        )
