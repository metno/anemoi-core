# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Lightning callbacks: train-side spread/skill logging, epoch-level SSR, adaptive spread factors.

Register under ``diagnostics.callbacks`` (an untyped list in the anemoi schema — custom
``_target_``\\s pass ``config_validation``), e.g.::

    diagnostics:
      callbacks:
        - _target_: anemoi.training.spread.callbacks.SpreadSkillLogger
          variables: ${training.metrics.datasets.data}
          dump_path: ${system.output.logs.root}/spread_skill_epochs.jsonl   # per-variable epoch dump (all vars)
        - _target_: anemoi.training.spread.callbacks.GpuMemoryLogger
        - _target_: anemoi.training.spread.callbacks.CallbackMetricsDump
          path: ${system.output.logs.root}/callback_metrics.jsonl

Metric-name conventions (mirroring anemoi's validation metrics
``val_{ds}_{metric}_metric/{ds}/{var}/{step}``):

* ``train_{ds}_ensvar_metric/{ds}/{var}/{step}`` and ``train_{ds}_ensmse_metric/{ds}/{var}/{step}``
  — per-batch means of ⟨s²⟩ and ⟨(x̄−y)²⟩ on training batches (normalised space), logged
  ``on_epoch=True`` (epoch mean; ``sync_dist``) and optionally ``on_step``;
* ``train_{ds}_ssr/{ds}/{var}/{step}`` and ``val_{ds}_ssr/{ds}/{var}/{step}`` — the corrected
  spread–skill ratio ``sqrt((M+1)/M · ⟨s²⟩ / ⟨(x̄−y)²⟩)`` computed **from the epoch means**
  (sqrt of means, never mean of sqrts) at ``on_train_epoch_end`` /
  ``on_validation_epoch_end``.  ``M`` = ``pl_module.nens_per_group``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from anemoi.training.spread.crps import SpreadScaledCRPS
from anemoi.training.spread.diagnostics import TrainingDiagnosticsLossWrapper
from anemoi.training.spread.metrics import spread_skill_ratio
from anemoi.training.spread.multiscale import MultiscaleSpreadLossWrapper

LOGGER = logging.getLogger(__name__)


def _diagnostic_wrappers(pl_module: pl.LightningModule) -> dict[str, TrainingDiagnosticsLossWrapper]:
    """{dataset_name: TrainingDiagnosticsLossWrapper} found at the top of ``pl_module.loss``."""
    losses = getattr(pl_module, "loss", None)
    if losses is None:
        return {}
    items = losses.items() if hasattr(losses, "items") else [("data", losses)]
    return {ds: loss for ds, loss in items if isinstance(loss, TrainingDiagnosticsLossWrapper)}


def _spread_leaves(pl_module: pl.LightningModule) -> list[SpreadScaledCRPS]:
    """All SpreadScaledCRPS leaves under ``pl_module.loss`` (walks wrappers via ``.loss``)."""
    found: list[SpreadScaledCRPS] = []
    seen: set[int] = set()

    def walk(module: object) -> None:
        if id(module) in seen or module is None:
            return
        seen.add(id(module))
        if isinstance(module, SpreadScaledCRPS):
            found.append(module)
            return
        for child in (getattr(module, "loss", None), *getattr(module, "losses", [])):
            walk(child)

    losses = getattr(pl_module, "loss", None)
    for loss in losses.values() if hasattr(losses, "values") else [losses]:
        walk(loss)
    return found


def _multiscale_spread_wrappers(pl_module: pl.LightningModule) -> list[MultiscaleSpreadLossWrapper]:
    found: list[MultiscaleSpreadLossWrapper] = []

    def walk(module: object) -> None:
        if module is None:
            return
        if isinstance(module, MultiscaleSpreadLossWrapper):
            found.append(module)
            return
        for child in (getattr(module, "loss", None), *getattr(module, "losses", [])):
            walk(child)

    losses = getattr(pl_module, "loss", None)
    for loss in losses.values() if hasattr(losses, "values") else [losses]:
        walk(loss)
    return found


def _ens_size(pl_module: pl.LightningModule) -> int | None:
    m = getattr(pl_module, "nens_per_group", None)
    return int(m) if m else None


def _per_run_path(path: str, trainer: pl.Trainer) -> str:
    """Insert the (MLflow) run id into a dump filename: runs sharing an output root would otherwise
    append into the same file. Falls back to the plain path when no run id is resolvable."""
    run_id = None
    for lg in getattr(trainer, "loggers", None) or ([trainer.logger] if getattr(trainer, "logger", None) else []):
        run_id = getattr(lg, "run_id", None) or getattr(getattr(lg, "experiment", None), "run_id", None)
        if isinstance(run_id, str) and run_id:
            break
        run_id = None
    if not run_id:
        return path
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{str(run_id)[:12]}{p.suffix}"))


def _n_task_steps(pl_module: pl.LightningModule, mode: str) -> int | None:
    """Number of rollout steps of the task in ``mode`` (``None`` if unknown)."""
    task = getattr(pl_module, "task", None)
    if task is not None and hasattr(task, "steps"):
        try:
            return len(task.steps(mode))
        except Exception:  # noqa: BLE001 - defensive: never break training for a diagnostic
            return None
    return None


def _entry_step_and_tensors(order_step: int, values: dict[str, Any]) -> tuple[int, dict[str, torch.Tensor]]:
    """Resolve a pending entry to its (1-based) rollout step and its tensor-valued metrics.

    Entries recorded with the forwarded ``rollout_step`` carry an authoritative non-tensor
    ``"_rollout_step"`` (0-based); older entries fall back to the call-order inference.
    """
    recorded = values.get("_rollout_step")
    step = recorded + 1 if isinstance(recorded, int) else order_step
    return step, {k: v for k, v in values.items() if isinstance(v, torch.Tensor)}


def _peek_forward_entries(wrapper: TrainingDiagnosticsLossWrapper, n_steps: int | None) -> list[dict[str, torch.Tensor]]:
    """The forward-pass entries of the current batch, in rollout order.

    ``pending`` holds ``fwd_0 … fwd_{n-1}`` followed (training only, checkpoint recompute) by the
    identical values in reverse; keep the first ``n_steps``.
    """
    pending = list(wrapper.pending)
    if n_steps is not None and len(pending) >= n_steps:
        pending = pending[:n_steps]
    return pending


def _all_reduce_sum_(t: torch.Tensor) -> torch.Tensor:
    """In-place sum over all ranks (no-op without an initialised process group)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t


class EpochAccumulator:
    """Per-(dataset, rollout step, metric) sums of the wrapper's per-variable vectors over an epoch.

    ``add(ds, step, values)`` accumulates one batch; :meth:`reduce` sums across ranks and returns
    ``{(ds, step): {"count": n, metric: mean_vector, ...}}`` — epoch **means** per variable
    (normalised loss space), i.e. exactly the ⟨s²⟩ / ⟨(x̄−y)²⟩ the SSR needs (sqrt of means).
    Ranks of the same model-comm group hold identical (already group-reduced) vectors, so the
    global sum double-counts them consistently in numerator and count — means are unaffected.
    """

    def __init__(self) -> None:
        self.sums: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self.counts: dict[tuple[str, int], int] = {}

    def clear(self) -> None:
        self.sums.clear()
        self.counts.clear()

    def add(self, ds: str, step: int, values: dict[str, torch.Tensor]) -> None:
        key = (ds, step)
        slot = self.sums.setdefault(key, {})
        for name, vec in values.items():
            v = vec.detach().float().reshape(-1)
            slot[name] = slot[name] + v if name in slot else v.clone()
        self.counts[key] = self.counts.get(key, 0) + 1

    def reduce(self) -> dict[tuple[str, int], dict[str, Any]]:
        out: dict[tuple[str, int], dict[str, Any]] = {}
        for key, slot in self.sums.items():
            names = sorted(slot)
            device = slot[names[0]].device
            count = _all_reduce_sum_(torch.tensor([float(self.counts[key])], device=device)).item()
            rec: dict[str, Any] = {"count": int(round(count))}
            for name in names:
                total = _all_reduce_sum_(slot[name].clone())
                rec[name] = (total / max(count, 1.0)).cpu()
            out[key] = rec
        return out


class SpreadSkillLogger(Callback):
    """Log train-side ⟨s²⟩ / ⟨(x̄−y)²⟩ per variable and the epoch-level SSR (train and val).

    Parameters
    ----------
    variables : list[str] | None
        Variables to log individually on the train side and — because anemoi averages
        validation metrics over the variables of a metric range in *physical* units — the only
        keys for which a validation-side SSR is derived.  Use ``${training.metrics.datasets.data}``
        in the config to mirror the validation list.
    log_all : bool
        Also log the plain mean over all model-output variables under ``all``.
    every_n_batches : int
        Record only every n-th training batch (values are always *computed* by the wrapper; this
        limits logging traffic).  ``1`` = every batch.
    log_on_step : bool
        Additionally log the per-batch values ``on_step=True`` (adds a per-step ``sync_dist``
        reduction; off by default — the epoch means are what the SSR needs).
    ensvar_name, ensmse_name : str
        Names of the diagnostic metrics inside :class:`TrainingDiagnosticsLossWrapper` **and** of
        the corresponding ``validation_metrics`` entries (defaults ``ensvar`` / ``ensmse``).
    dump_path : str | None
        If set, rank 0 appends one JSON record per train / validation epoch (and rollout step)
        with the epoch-mean ⟨s²⟩, ⟨(x̄−y)²⟩ and the corrected SSR for **all** model-output
        variables (plus ``all`` = mean over variables), in normalised (loss) space, accumulated
        with :class:`EpochAccumulator` (all-reduced over ranks).  Source of the target-noise amplitude
        tables and of per-variable figures.
    """

    def __init__(
        self,
        variables: Iterable[str] | None = None,
        *,
        log_all: bool = True,
        every_n_batches: int = 1,
        log_on_step: bool = False,
        ensvar_name: str = "ensvar",
        ensmse_name: str = "ensmse",
        dump_path: str | None = None,
    ) -> None:
        super().__init__()
        self.variables = list(variables) if variables is not None else []
        self.log_all = bool(log_all)
        self.every_n_batches = max(1, int(every_n_batches))
        self.log_on_step = bool(log_on_step)
        self.ensvar_name = ensvar_name
        self.ensmse_name = ensmse_name
        self.dump_path = str(dump_path) if dump_path else None
        self.last_train_ssr: dict[str, float] = {}
        self.last_val_ssr: dict[str, float] = {}
        self._acc = {"train": EpochAccumulator(), "val": EpochAccumulator()}
        self.last_dump: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}

    # ------------------------------------------------------------------ per-batch train-side logging
    def _clear(self, pl_module: pl.LightningModule) -> None:
        for wrapper in _diagnostic_wrappers(pl_module).values():
            wrapper.drain()

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:  # noqa: ARG002
        self._clear(pl_module)

    def on_validation_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:  # noqa: ARG002
        self._clear(pl_module)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:  # noqa: ARG002
        # Validation-side *logged* numbers come from the proper validation_metrics (physical space);
        # the loss-space vectors are only accumulated for the per-variable epoch dump.
        if getattr(trainer, "sanity_checking", False):
            return
        n_steps = _n_task_steps(pl_module, "validation")
        for ds, wrapper in _diagnostic_wrappers(pl_module).items():
            for order, values in enumerate(_peek_forward_entries(wrapper, n_steps), start=1):
                step, tensors = _entry_step_and_tensors(order, values)
                self._acc["val"].add(ds, step, tensors)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:  # noqa: ARG002
        wrappers = _diagnostic_wrappers(pl_module)
        if not wrappers:
            return
        n_steps = _n_task_steps(pl_module, "training")
        batch_size = next(iter(batch.values())).shape[0] if isinstance(batch, dict) else None

        for ds, wrapper in wrappers.items():
            pending = _peek_forward_entries(wrapper, n_steps)  # forward-pass entries in rollout order (recomputes follow)
            for order, values in enumerate(pending, start=1):
                step, tensors = _entry_step_and_tensors(order, values)
                self._acc["train"].add(ds, step, tensors)
            if not pending or batch_idx % self.every_n_batches:
                continue
            names = wrapper.variable_names
            for step, values in enumerate(pending, start=1):
                for metric_name, vec in values.items():
                    if metric_name not in (self.ensvar_name, self.ensmse_name):
                        continue
                    key = f"train_{ds}_{metric_name}_metric/{ds}"
                    entries: dict[str, torch.Tensor] = {}
                    if self.log_all:
                        entries["all"] = vec.mean()
                    if names is not None:
                        idx = {n: i for i, n in enumerate(names)}
                        for var in self.variables:
                            if var in idx:
                                entries[var] = vec[idx[var]]
                    for mkey, value in entries.items():
                        pl_module.log(
                            f"{key}/{mkey}/{step}",
                            value,
                            on_step=self.log_on_step,
                            on_epoch=True,
                            prog_bar=False,
                            logger=getattr(pl_module, "logger_enabled", True),
                            batch_size=batch_size,
                            sync_dist=True,
                        )

    # ------------------------------------------------------------------ epoch-level SSR
    def _log_ssr(self, trainer: pl.Trainer, pl_module: pl.LightningModule, split: str) -> dict[str, float]:
        """Derive SSR from the epoch means already reduced into ``trainer.callback_metrics``."""
        ens_size = _ens_size(pl_module)
        if ens_size is None:
            LOGGER.warning("SpreadSkillLogger: pl_module has no nens_per_group; SSR not computed.")
            return {}
        metrics = trainer.callback_metrics
        out: dict[str, float] = {}
        var_prefix = f"{split}_"
        for key, value in list(metrics.items()):
            # key = f"{split}_{ds}_{ensvar}_metric/{ds}/{mkey}/{step}"  (no _epoch suffix: on_step=False)
            if not key.startswith(var_prefix) or f"_{self.ensvar_name}_metric/" not in key or key.endswith("_step") or key.endswith("_epoch"):
                continue
            mse_key = key.replace(f"_{self.ensvar_name}_metric/", f"_{self.ensmse_name}_metric/", 1)
            if mse_key not in metrics:
                continue
            # metric-range keys: .../{mkey}/{step}. Validation metrics live in physical units and anemoi
            # averages them over the variables of a range, so 'all'/group ranges give a dimensionally
            # incoherent SSR -> keep only single-variable keys (those listed in training.metrics).
            # Train-side keys are ours (normalised space): 'all' is a coherent summary there.
            mkey = key.rsplit("/", 2)[-2] if key.count("/") >= 2 else ""
            if split == "val" and mkey not in self.variables:
                continue
            ssr = spread_skill_ratio(float(value), float(metrics[mse_key]), ens_size)
            if math.isnan(ssr):
                continue
            ssr_key = key.replace(f"_{self.ensvar_name}_metric/", "_ssr/", 1)
            out[ssr_key] = ssr
            pl_module.log(ssr_key, ssr, on_step=False, on_epoch=True, prog_bar=False, logger=getattr(pl_module, "logger_enabled", True), sync_dist=False, rank_zero_only=True)
        return out

    # ------------------------------------------------------------------ per-variable epoch dump (all variables)
    def _dump_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule, split: str) -> list[dict[str, Any]]:
        """Reduce the split's accumulator (collective!), build the records, write them on rank 0."""
        acc = self._acc[split]
        if not acc.sums:
            return []
        LOGGER.info(
            "SpreadSkillLogger: %s epoch reduce start (%d slots, %d metric keys)",
            split,
            len(acc.sums),
            sum(len(s) for s in acc.sums.values()),
        )
        reduced = acc.reduce()
        LOGGER.info("SpreadSkillLogger: %s epoch reduce done", split)
        acc.clear()
        ens_size = _ens_size(pl_module)
        wrappers = _diagnostic_wrappers(pl_module)
        records: list[dict[str, Any]] = []
        for (ds, step), rec in sorted(reduced.items()):
            names = wrappers[ds].variable_names if ds in wrappers else None
            var = rec.get(self.ensvar_name)
            mse = rec.get(self.ensmse_name)
            if var is None or mse is None:
                continue
            n = int(var.numel())
            names = list(names) if names is not None and len(names) == n else [str(i) for i in range(n)]
            ssr = {}
            if ens_size is not None:
                for i, name in enumerate(names):
                    r = spread_skill_ratio(float(var[i]), float(mse[i]), ens_size)
                    ssr[name] = r if not math.isnan(r) else None
                r_all = spread_skill_ratio(float(var.mean()), float(mse.mean()), ens_size)
                ssr["all"] = r_all if not math.isnan(r_all) else None
            record = {
                "split": split,
                "dataset": ds,
                "rollout_step": step,
                "epoch": int(getattr(trainer, "current_epoch", -1)),
                "global_step": int(getattr(trainer, "global_step", -1)),
                "n_batches": rec["count"],
                "ens_size": ens_size,
                "space": "normalised",
                self.ensvar_name: {**{name: float(var[i]) for i, name in enumerate(names)}, "all": float(var.mean())},
                self.ensmse_name: {**{name: float(mse[i]) for i, name in enumerate(names)}, "all": float(mse.mean())},
                "ssr": ssr,
            }
            # per-band blocks (keys "{ensvar}_band{k}" recorded by the multiscale wrapper)
            bands: dict[str, Any] = {}
            for k in range(16):
                bvar = rec.get(f"{self.ensvar_name}_band{k}")
                bmse = rec.get(f"{self.ensmse_name}_band{k}")
                if bvar is None or bmse is None:
                    break
                bssr = {}
                if ens_size is not None:
                    for i, name in enumerate(names):
                        r = spread_skill_ratio(float(bvar[i]), float(bmse[i]), ens_size)
                        bssr[name] = r if not math.isnan(r) else None
                    r_all = spread_skill_ratio(float(bvar.mean()), float(bmse.mean()), ens_size)
                    bssr["all"] = r_all if not math.isnan(r_all) else None
                bands[str(k)] = {
                    self.ensvar_name: {**{name: float(bvar[i]) for i, name in enumerate(names)}, "all": float(bvar.mean())},
                    self.ensmse_name: {**{name: float(bmse[i]) for i, name in enumerate(names)}, "all": float(bmse.mean())},
                    "ssr": bssr,
                }
            if bands:
                record["bands"] = bands
            records.append(record)
        if self.dump_path and getattr(trainer, "is_global_zero", True) and records:
            p = Path(_per_run_path(self.dump_path, trainer))
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
        self.last_dump[split] = records
        return records

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.last_train_ssr = self._log_ssr(trainer, pl_module, "train")
        self._dump_epoch(trainer, pl_module, "train")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(trainer, "sanity_checking", False):
            self._acc["val"].clear()
            return
        self.last_val_ssr = self._log_ssr(trainer, pl_module, "val")
        self._dump_epoch(trainer, pl_module, "val")


class CombinedAdaptiveSpreadController(Callback):
    """Adaptive controller of the spread factors of a multiscale spread loss.

    Drives val SSR → 1 per (variable × band × rollout step) by feedback on the widening tensor of a
    :class:`MultiscaleSpreadLossWrapper`:

    * **Signal**: live *train*-side per-(v, b, r) ⟨s²⟩ / ⟨(x̄−y)²⟩ from the
      band diagnostics, smoothed by per-batch EMAs, corrected by the last-known val/train SSR
      ratio per (v, b, r) — the ratio refreshes at every true validation and is logged/dumped
      (it is the live ``g`` estimate).
    * **Structure**: full per-(variable, band) table shared across rollout steps, times a
      per-(band, rollout-step) factor ``a_{b,r}``;
      ``a_{b,0} ≡ 1`` (identifiability — the table owns the first lead). Variables with static leaf overrides (bounded pins) are never controlled.
    * **Update** every ``update_interval`` global steps: multiplicative
      ``a ← a · (1/SSR_corr)^κ`` with the per-update change clamped to ``±max_update_frac``;
      table entries driven by ``SSR_corr(v, b, r=0)`` and clamped to ``table_clamp`` — either one
      global ``(lo, hi)`` or one pair per band; ``a_{b,r}`` (r ≥ 1)
      driven by the **median** over v of ``SSR_corr(·, b, r)`` (the median, so that a few variables do not dominate) and clamped to ``rollout_clamp`` — one global ``(lo, hi)`` or one pair per band
      (clamping the coarse/mid ``a_{b,r}`` at 1.0 guards against a spread runaway at long leads).
    * **Seeding**: ``seed_dump_path`` initialises the table and ``a_{b,r}`` from the last
      record of a previous run's controller dump . A flat legacy ``a_r`` record is broadcast across bands;
      ``seed_extend_rollout`` extends each band row to the new rollout depth by repeating the last
      value. Seeded values are clamped into the (possibly new) band
      clamps and the number of clipped entries is logged. A checkpoint resume wins over the seed.
    * **Warmup/phases**: no updates before ``warmup_steps`` and until the median corrected
      SSR at r = 0 exceeds ``min_ssr``; when the number of rollout steps changes mid-run (rollout
      phase start), updates freeze for ``freeze_steps`` and the transition is logged with a table
      snapshot.
    * EMAs of ⟨s²⟩ and ⟨(x̄−y)²⟩ are kept separately (ratio of smoothed quantities) and
      all-reduced across ranks at update/validation time, so every rank applies identical factors
      (the widening is part of the loss — ranks must not diverge).

    State (table, ``a_{b,r}``, EMAs, ratios, counters) is checkpointed with the callback.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        update_interval: int = 250,
        kappa: float = 0.3,
        max_update_frac: float = 0.015,
        table_clamp: Sequence[float] | Sequence[Sequence[float]] = (0.9, 1.3),
        rollout_clamp: tuple[float, float] = (0.9, 1.2),
        ema_horizon: int = 250,
        warmup_steps: int = 500,
        min_ssr: float = 0.3,
        freeze_steps: int = 1000,
        dataset_name: str = "data",
        ensvar_name: str = "ensvar",
        ensmse_name: str = "ensmse",
        dump_path: str | None = None,
        seed_dump_path: str | None = None,
        seed_extend_rollout: int | None = None,
        ssr_target: float | Sequence[float] = 1.0,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)  # master switch (False -> fully inert)
        # the SSR the controller drives towards, one value or one per 0-based rollout
        # step (clamp-to-last; e.g. [1.0, 1.0, 1.0, 1.03] = 24 h target 1.03). 1.0 = the default law.
        self.ssr_target: list[float] = [float(x) for x in ssr_target] if isinstance(ssr_target, (list, tuple)) else [float(ssr_target)]
        if not self.ssr_target or any(not (x > 0) for x in self.ssr_target):
            msg = f"ssr_target must be positive (one value or one per rollout step), got {ssr_target!r}"
            raise ValueError(msg)
        self.update_interval = max(1, int(update_interval))
        self.kappa = float(kappa)
        self.max_update_frac = float(max_update_frac)
        # global (lo, hi) or one (lo, hi) per band; normalised to per-band in on_train_start
        clamp = list(table_clamp)
        if isinstance(clamp[0], (int, float)):
            self.table_clamp: list[tuple[float, float]] | tuple[float, float] = (float(clamp[0]), float(clamp[1]))
        else:
            self.table_clamp = [(float(lo), float(hi)) for lo, hi in clamp]
        rclamp = list(rollout_clamp)
        if isinstance(rclamp[0], (int, float)):
            self.rollout_clamp: list[tuple[float, float]] | tuple[float, float] = (float(rclamp[0]), float(rclamp[1]))
        else:  # one (lo, hi) per band
            self.rollout_clamp = [(float(lo), float(hi)) for lo, hi in rclamp]
        self.ema_alpha = math.exp(-1.0 / max(1, int(ema_horizon)))
        self.warmup_steps = int(warmup_steps)
        self.min_ssr = float(min_ssr)
        self.freeze_steps = int(freeze_steps)
        self.dataset_name = dataset_name
        self.ensvar_name = ensvar_name
        self.ensmse_name = ensmse_name
        self.dump_path = str(dump_path) if dump_path else None
        self.seed_dump_path = str(seed_dump_path) if seed_dump_path else None
        self.seed_extend_rollout = int(seed_extend_rollout) if seed_extend_rollout else None

        # live state (initialised on_train_start / grown on first sight of a (b, r) slot)
        self.names: list[str] = []
        self.controlled: torch.Tensor | None = None  # (v,) bool — False for pinned variables
        self.table: torch.Tensor | None = None  # (v, B)
        self.a_br: list[list[float]] = []  # [band][r-1] (a_{b,0} fixed at 1.0)
        self._clamp_lo: torch.Tensor | None = None  # (1, B) — built in on_train_start
        self._clamp_hi: torch.Tensor | None = None
        self._ema_var: dict[tuple[int, int], torch.Tensor] = {}  # (b, r) -> (v,)
        self._ema_mse: dict[tuple[int, int], torch.Tensor] = {}
        self._ratio: dict[tuple[int, int], torch.Tensor] = {}  # val/train SSR ratio (live g)
        self._val_var: dict[tuple[int, int], torch.Tensor] = {}
        self._val_mse: dict[tuple[int, int], torch.Tensor] = {}
        self._val_count = 0
        self._pending_val: tuple[dict, dict] | None = None  # stashed val sums awaiting the next update
        self._last_update_step = -1
        self._freeze_until = -1
        self._last_n_steps: int | None = None
        self._last_factor_sign: torch.Tensor | None = None
        self._sign_flips = 0
        self._n_updates = 0
        self._pending_state: dict[str, Any] | None = None
        self._num_bands = 0

    # ------------------------------------------------------------------ wiring
    def _wrapper(self, pl_module: pl.LightningModule) -> MultiscaleSpreadLossWrapper | None:
        wrappers = _multiscale_spread_wrappers(pl_module)
        return wrappers[0] if wrappers else None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        if not self.enabled:
            LOGGER.info("CombinedAdaptiveSpreadController: enabled=False; inert.")
            return
        wrapper = self._wrapper(pl_module)
        if wrapper is None:
            LOGGER.warning("CombinedAdaptiveSpreadController: no MultiscaleSpreadLossWrapper; inactive.")
            return
        leaf = wrapper._spread_leaves[0]
        index = leaf.variable_index
        if index is None:
            LOGGER.warning(
                "CombinedAdaptiveSpreadController: leaf variable names unresolved; inactive.",
            )
            return
        self.names = [name for name, _ in sorted(index.items(), key=lambda kv: kv[1])]
        pinned = set(wrapper.static_variable_overrides)
        self.controlled = torch.tensor([name not in pinned for name in self.names])
        self._num_bands = wrapper.num_scales

        # normalise the clamp config to per-band (1, B) bound tensors
        clamp = self.table_clamp
        if isinstance(clamp, tuple):
            clamp = [clamp] * self._num_bands
        if len(clamp) != self._num_bands:
            msg = f"table_clamp has {len(clamp)} band pairs, expected {self._num_bands}"
            raise ValueError(msg)
        if isinstance(self.rollout_clamp, list) and len(self.rollout_clamp) != self._num_bands:
            msg = f"rollout_clamp has {len(self.rollout_clamp)} band pairs, expected {self._num_bands}"
            raise ValueError(msg)
        self._clamp_lo = torch.tensor([lo for lo, _ in clamp], dtype=torch.float64).unsqueeze(0)
        self._clamp_hi = torch.tensor([hi for _, hi in clamp], dtype=torch.float64).unsqueeze(0)

        if self._pending_state is not None:
            self._load_live_state(self._pending_state)
            self._pending_state = None
        if self.table is None and self.seed_dump_path:
            self._load_seed()
        if self.table is None:
            configured = wrapper.get_per_variable_band_table()
            self.table = torch.ones(len(self.names), self._num_bands, dtype=torch.float64)
            for name, values in configured.items():
                if name in index:
                    self.table[self.names.index(name)] = torch.tensor(values, dtype=torch.float64)
        if not self.a_br:
            self.a_br = [[] for _ in range(self._num_bands)]
        self._apply(wrapper)
        LOGGER.info(
            "CombinedAdaptiveSpreadController active: %d variables (%d controlled), %d bands, "
            "update every %d steps, kappa=%.2f, clamps table=%s rollout=%s, warmup=%d, min_ssr=%.2f",
            len(self.names),
            int(self.controlled.sum()),
            self._num_bands,
            self.update_interval,
            self.kappa,
            self.table_clamp,
            self.rollout_clamp,
            self.warmup_steps,
            self.min_ssr,
        )

    def _load_seed(self) -> None:
        """Initialise table and ``a_{b,r}`` from the last record of a previous run's dump."""
        lines = Path(self.seed_dump_path).read_text().strip().splitlines()
        rec = json.loads(lines[-1])
        self.table = torch.ones(len(self.names), self._num_bands, dtype=torch.float64)
        matched = 0
        for name, values in rec.get("table", {}).items():
            if name in self.names:
                self.table[self.names.index(name)] = torch.tensor(values, dtype=torch.float64)
                matched += 1
        raw = self.table.clone()
        self.table = torch.maximum(torch.minimum(self.table, self._clamp_hi), self._clamp_lo)
        clipped = int((self.table != raw).sum())
        a_br = rec.get("a_br")
        if a_br is None and rec.get("a_r"):  # legacy shared a_r -> broadcast across bands
            a_br = [list(rec["a_r"])] * self._num_bands
        self.a_br = [[float(v) for v in row] for row in (a_br or [[] for _ in range(self._num_bands)])]
        if self.seed_extend_rollout:  # extend to the new rollout depth by repeating the last value
            target = self.seed_extend_rollout - 1
            for row in self.a_br:
                while row and len(row) < target:
                    row.append(row[-1])
        self.a_br = [
            [min(max(v, self._rollout_bounds(b)[0]), self._rollout_bounds(b)[1]) for v in row] for b, row in enumerate(self.a_br)
        ]
        LOGGER.info(
            "CombinedAdaptiveSpreadController: seeded from %s @ step %s (%d/%d table rows matched, "
            "%d entries clipped into the band clamps), a_br = %s",
            self.seed_dump_path,
            rec.get("global_step"),
            matched,
            int(self.controlled.sum()),
            clipped,
            [[round(v, 4) for v in row] for row in self.a_br],
        )

    def _apply(self, wrapper: MultiscaleSpreadLossWrapper) -> None:
        """Push the controller state into the wrapper (controlled variables only)."""
        table = {
            name: [float(v) for v in self.table[i]]
            for i, name in enumerate(self.names)
            if bool(self.controlled[i])
        }
        wrapper.set_per_variable_band_table(table)
        if any(self.a_br):
            depth = max(len(row) for row in self.a_br)
            rows = [[1.0, *row, *([row[-1]] * (depth - len(row)))] if row else [1.0] * (depth + 1) for row in self.a_br]
            wrapper.set_per_rollout_step_adjustment(rows)
        else:
            wrapper.set_per_rollout_step_adjustment(None)

    # ------------------------------------------------------------------ signal accumulation
    def _rollout_bounds(self, band: int) -> tuple[float, float]:
        """``(lo, hi)`` of ``a_{b,r}`` for ``band``: the global pair, or the band's own pair."""
        if isinstance(self.rollout_clamp, tuple):
            return self.rollout_clamp
        return self.rollout_clamp[band]

    def _band_vectors(self, values: dict[str, Any]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        # kept on the metrics' device: the NCCL all-reduce at update time needs CUDA tensors
        out = {}
        for b in range(self._num_bands):
            var = values.get(f"{self.ensvar_name}_band{b}")
            mse = values.get(f"{self.ensmse_name}_band{b}")
            if var is not None and mse is not None:
                out[b] = (var.detach().double(), mse.detach().double())
        return out

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:  # noqa: ARG002
        if self.table is None:
            return
        wrapper = self._wrapper(pl_module)
        wrappers = _diagnostic_wrappers(pl_module)
        if wrapper is None or self.dataset_name not in wrappers:
            return
        n_steps = _n_task_steps(pl_module, "training")
        if n_steps is not None and self._last_n_steps is not None and n_steps != self._last_n_steps:
            self._freeze_until = int(getattr(trainer, "global_step", 0)) + self.freeze_steps
            LOGGER.info(
                "CombinedAdaptiveSpreadController: rollout steps %d -> %d at step %d; updates frozen "
                "for %d steps. table min/mean/max = %.4f/%.4f/%.4f, a_br = %s",
                self._last_n_steps,
                n_steps,
                int(getattr(trainer, "global_step", 0)),
                self.freeze_steps,
                float(self.table.min()),
                float(self.table.mean()),
                float(self.table.max()),
                [[round(v, 4) for v in row] for row in self.a_br],
            )
        self._last_n_steps = n_steps

        for order, values in enumerate(_peek_forward_entries(wrappers[self.dataset_name], n_steps), start=1):
            step, _tensors = _entry_step_and_tensors(order, values)
            r = step - 1
            for b, (var, mse) in self._band_vectors(values).items():
                key = (b, r)
                if key not in self._ema_var:
                    self._ema_var[key] = var.clone()
                    self._ema_mse[key] = mse.clone()
                else:
                    a = self.ema_alpha
                    # a checkpoint-resumed EMA may live on CPU — rehome it to the incoming device
                    if self._ema_var[key].device != var.device:
                        self._ema_var[key] = self._ema_var[key].to(var.device)
                        self._ema_mse[key] = self._ema_mse[key].to(var.device)
                    self._ema_var[key].mul_(a).add_(var, alpha=1.0 - a)
                    self._ema_mse[key].mul_(a).add_(mse, alpha=1.0 - a)
            for row in self.a_br:
                if r > len(row):
                    row.extend([1.0] * (r - len(row)))

        global_step = int(getattr(trainer, "global_step", 0))
        if (
            global_step >= self.warmup_steps
            and global_step >= self._freeze_until
            and global_step - self._last_update_step >= self.update_interval
            and self._ema_var
        ):
            self._update(trainer, pl_module, wrapper, global_step)

    # ------------------------------------------------------------------ the update
    def _corrected_ssr(self, key: tuple[int, int], ens_size: int) -> torch.Tensor | None:
        var = self._ema_var.get(key)
        mse = self._ema_mse.get(key)
        if var is None or mse is None:
            return None
        var = _all_reduce_sum_(var.clone())
        mse = _all_reduce_sum_(mse.clone())
        # summing var and mse over ranks leaves their ratio ≈ the rank-mean ratio (R cancels)
        ssr = ((ens_size + 1) / ens_size * var / mse.clamp(min=1e-30)) ** 0.5
        ssr[mse <= 0] = float("nan")
        ssr = ssr.cpu()
        ratio = self._ratio.get(key)
        if ratio is not None:
            ssr = ssr * ratio
        return ssr

    def _target(self, step: int) -> float:
        """SSR target of 0-based rollout step ``step`` (clamp-to-last; 1.0 unless ``ssr_target`` is set)."""
        return self.ssr_target[min(int(step), len(self.ssr_target) - 1)]

    def _update(self, trainer: pl.Trainer, pl_module: pl.LightningModule, wrapper: MultiscaleSpreadLossWrapper, global_step: int) -> None:
        ens_size = _ens_size(pl_module)
        if ens_size is None:
            return
        LOGGER.debug("CombinedAdaptiveSpreadController: update reduce start (step %d)", global_step)
        self._refresh_ratio(ens_size)  # folds any stashed validation sums into the ratios first
        ssr0 = [self._corrected_ssr((b, 0), ens_size) for b in range(self._num_bands)]
        if any(s is None for s in ssr0):
            return
        ssr0 = torch.stack(ssr0, dim=1)  # (v, B)
        med0 = float(torch.nanmedian(ssr0))
        if not math.isfinite(med0) or med0 <= self.min_ssr:
            LOGGER.info(
                "CombinedAdaptiveSpreadController: median corrected SSR %.3f <= min_ssr %.2f at step %d; holding.",
                med0,
                self.min_ssr,
                global_step,
            )
            self._last_update_step = global_step
            return

        lo, hi = 1.0 - self.max_update_frac, 1.0 + self.max_update_frac
        factor = (self._target(0) / ssr0.clamp(min=1e-6)) ** self.kappa
        factor = factor.clamp(min=lo, max=hi)
        factor[~torch.isfinite(ssr0)] = 1.0
        factor[~self.controlled] = 1.0
        self.table = torch.maximum(torch.minimum(self.table * factor, self._clamp_hi), self._clamp_lo)

        sign = torch.sign(torch.log(factor))
        if self._last_factor_sign is not None and self._last_factor_sign.shape == sign.shape:
            self._sign_flips = int(((sign * self._last_factor_sign) < 0).sum())
        self._last_factor_sign = sign

        # a_{b,r} (r >= 1): each band row driven by the median over v of its own band's SSR
        r_factors: list[list[float]] = [[] for _ in range(self._num_bands)]
        depth = max((len(row) for row in self.a_br), default=0)
        ssr_cache: dict[tuple[int, int], torch.Tensor | None] = {}
        for r in range(1, depth + 1):
            for b in range(self._num_bands):
                ssr_cache[(b, r)] = self._corrected_ssr((b, r), ens_size)
        for b, row in enumerate(self.a_br):
            for r in range(1, len(row) + 1):
                ssr_r = ssr_cache.get((b, r))
                if ssr_r is None:
                    r_factors[b].append(1.0)
                    continue
                med = float(torch.nanmedian(ssr_r))
                if not math.isfinite(med) or med <= 0:
                    r_factors[b].append(1.0)
                    continue
                f = min(max((self._target(r) / med) ** self.kappa, lo), hi)
                rlo, rhi = self._rollout_bounds(b)
                row[r - 1] = min(max(row[r - 1] * f, rlo), rhi)
                r_factors[b].append(f)

        self._apply(wrapper)
        self._last_update_step = global_step
        self._n_updates += 1

        log = getattr(pl_module, "log", None)
        if callable(log):
            log("adaptive/ssr_corr_med_r0", med0, on_step=True, on_epoch=False, logger=getattr(pl_module, "logger_enabled", True), rank_zero_only=True)
            log("adaptive/table_mean", float(self.table[self.controlled].mean()), on_step=True, on_epoch=False, logger=getattr(pl_module, "logger_enabled", True), rank_zero_only=True)
            log("adaptive/table_max", float(self.table[self.controlled].max()), on_step=True, on_epoch=False, logger=getattr(pl_module, "logger_enabled", True), rank_zero_only=True)
            log("adaptive/sign_flips", float(self._sign_flips), on_step=True, on_epoch=False, logger=getattr(pl_module, "logger_enabled", True), rank_zero_only=True)
            for b, row in enumerate(self.a_br):
                for r, v in enumerate(row, start=1):
                    log(f"adaptive/a_b{b}r{r}", v, on_step=True, on_epoch=False, logger=getattr(pl_module, "logger_enabled", True), rank_zero_only=True)
            r0 = self._ratio.get((0, 0))
            if r0 is not None:  # the live g estimate (val/train SSR, band 0, training lead)
                log("adaptive/val_train_ratio_med_b0r0", float(r0.median()), on_step=True, on_epoch=False, logger=getattr(pl_module, "logger_enabled", True), rank_zero_only=True)
        self._dump(trainer, global_step, med0, r_factors)

    # ------------------------------------------------------------------ validation ratio (the live g)
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:  # noqa: ARG002
        if self.table is None or getattr(trainer, "sanity_checking", False):
            return
        wrappers = _diagnostic_wrappers(pl_module)
        if self.dataset_name not in wrappers:
            return
        n_steps = _n_task_steps(pl_module, "validation")
        for order, values in enumerate(_peek_forward_entries(wrappers[self.dataset_name], n_steps), start=1):
            step, _tensors = _entry_step_and_tensors(order, values)
            r = step - 1
            for b, (var, mse) in self._band_vectors(values).items():
                key = (b, r)
                if key not in self._val_var:
                    self._val_var[key] = var.clone()
                    self._val_mse[key] = mse.clone()
                else:
                    self._val_var[key] += var.to(self._val_var[key].device)
                    self._val_mse[key] += mse.to(self._val_mse[key].device)
        self._val_count += 1

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Stash the epoch's val sums; NO collectives here.

        The ratio is computed inside the next training-side ``_update`` (a proven-working
        collective site). Running all-reduces from this hook deadlocks on NCCL — the training-loss collectives interleave with
        Lightning's own epoch-end reductions here. The ~one-update-interval lag this adds to the
        ratio's train-side denominator is well within the EMA horizon.
        """
        if self.table is None or getattr(trainer, "sanity_checking", False) or not self._val_var:
            self._val_var, self._val_mse, self._val_count = {}, {}, 0
            return
        self._pending_val = ({k: t.clone() for k, t in self._val_var.items()},
                             {k: t.clone() for k, t in self._val_mse.items()})
        self._val_var, self._val_mse, self._val_count = {}, {}, 0
        LOGGER.info(
            "CombinedAdaptiveSpreadController: val sums stashed (%d keys); ratio refresh at next update",
            len(self._pending_val[0]),
        )

    def _refresh_ratio(self, ens_size: int) -> None:
        """Fold the stashed validation sums into val/train SSR ratios (collectives; update-time only).

        RANK-SYMMETRIC BY CONSTRUCTION (the 2026-08-25 deadlock, job 54148969/54153752, was a rank
        whose local val sums came up empty skipping the loop while the others entered it): every
        rank always executes the same collective sequence — one flag all-reduce, then, if ANY rank
        stashed validation sums, one pass over the STRUCTURAL key set (the train-EMA keys, identical
        on all ranks), zero-filling keys a rank has no local data for (zeros are the correct
        contribution to a sum; the reduced totals cover exactly the ranks that saw data).
        """
        if not self._ema_var:
            return
        ref = next(iter(self._ema_var.values()))
        have = torch.tensor([0.0 if self._pending_val is None else 1.0, 0.0], device=ref.device)
        have = _all_reduce_sum_(have)
        if have[0] <= 0:
            return
        val_var, val_mse = self._pending_val if self._pending_val is not None else ({}, {})
        self._pending_val = None
        keys = sorted(self._ema_var)
        LOGGER.info(
            "CombinedAdaptiveSpreadController: ratio reduce start (%d structural keys, local val keys %d)",
            len(keys),
            len(val_var),
        )
        for key in keys:
            tvar = self._ema_var[key]
            tmse = self._ema_mse[key]
            zero = torch.zeros_like(tvar)
            vvar = _all_reduce_sum_(val_var.get(key, zero).clone().to(tvar.device))
            vmse = _all_reduce_sum_(val_mse.get(key, zero).clone().to(tvar.device))
            tvar = _all_reduce_sum_(tvar.clone())
            tmse = _all_reduce_sum_(tmse.clone())
            if float(vmse.abs().max()) <= 0.0:
                continue  # no rank saw validation data for this key
            ssr_val = ((ens_size + 1) / ens_size * vvar / vmse.clamp(min=1e-30)) ** 0.5
            ssr_train = ((ens_size + 1) / ens_size * tvar / tmse.clamp(min=1e-30)) ** 0.5
            ratio = ssr_val / ssr_train.clamp(min=1e-6)
            ratio[~torch.isfinite(ratio)] = 1.0
            self._ratio[key] = ratio.cpu()
        LOGGER.info("CombinedAdaptiveSpreadController: ratio reduce done")

    # ------------------------------------------------------------------ dump & state
    def _dump(self, trainer: pl.Trainer, global_step: int, med0: float, r_factors: list[list[float]]) -> None:
        if not self.dump_path or not getattr(trainer, "is_global_zero", True):
            return
        rec = {
            "global_step": global_step,
            "n_updates": self._n_updates,
            "ssr_corr_med_r0": med0,
            "table": {name: [round(float(v), 5) for v in self.table[i]] for i, name in enumerate(self.names) if bool(self.controlled[i])},
            "a_br": [[round(v, 5) for v in row] for row in self.a_br],
            "r_factors": [[round(f, 5) for f in row] for row in r_factors],
            "sign_flips": self._sign_flips,
            "ratio_med": {f"b{b}r{r}": round(float(t.median()), 5) for (b, r), t in sorted(self._ratio.items())},
        }
        p = Path(_per_run_path(self.dump_path, trainer))
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def _live_state(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "controlled": self.controlled.tolist() if self.controlled is not None else None,
            "table": self.table.tolist() if self.table is not None else None,
            "a_br": [list(row) for row in self.a_br],
            "ema_var": {f"{b},{r}": t.tolist() for (b, r), t in self._ema_var.items()},
            "ema_mse": {f"{b},{r}": t.tolist() for (b, r), t in self._ema_mse.items()},
            "ratio": {f"{b},{r}": t.tolist() for (b, r), t in self._ratio.items()},
            "last_update_step": self._last_update_step,
            "freeze_until": self._freeze_until,
            "last_n_steps": self._last_n_steps,
            "n_updates": self._n_updates,
        }

    def _load_live_state(self, state: dict[str, Any]) -> None:
        def _keyed(d: dict[str, list[float]] | None) -> dict[tuple[int, int], torch.Tensor]:
            out = {}
            for k, v in (d or {}).items():
                b, r = k.split(",")
                out[(int(b), int(r))] = torch.tensor(v, dtype=torch.float64)
            return out

        if state.get("table") is not None:
            self.table = torch.tensor(state["table"], dtype=torch.float64)
        if state.get("controlled") is not None:
            self.controlled = torch.tensor(state["controlled"], dtype=torch.bool)
        if state.get("a_br") is not None:
            self.a_br = [[float(v) for v in row] for row in state["a_br"]]
        elif state.get("a_r"):  # legacy shared-a_r checkpoint -> broadcast across bands
            self.a_br = [[float(v) for v in state["a_r"]] for _ in range(max(1, self._num_bands))]
        self._ema_var = _keyed(state.get("ema_var"))
        self._ema_mse = _keyed(state.get("ema_mse"))
        self._ratio = _keyed(state.get("ratio"))
        self._last_update_step = int(state.get("last_update_step", -1))
        self._freeze_until = int(state.get("freeze_until", -1))
        self._last_n_steps = state.get("last_n_steps")
        self._n_updates = int(state.get("n_updates", 0))

    def state_dict(self) -> dict[str, Any]:
        return self._live_state()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # pl_module is not available here; re-applied in on_train_start
        self._pending_state = dict(state_dict) if state_dict else None


class CallbackMetricsDump(Callback):
    """Append ``trainer.callback_metrics`` (all logged scalars, epoch-reduced) as one JSON line per
    train/validation epoch end — rank 0 only. A logger-independent record of the metric history
    (dev runs have MLflow disabled; production runs get a plain-text copy next to the MLflow store).

    Parameters
    ----------
    path : str
        Output file (parents are created); e.g. ``${system.output.logs.root}/callback_metrics.jsonl``.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = str(path)

    def _dump(self, trainer: pl.Trainer, tag: str) -> None:
        # Read callback_metrics on ALL ranks before the rank gate: the access can trigger
        # Lightning's deferred sync_dist reductions, and a rank-0-only read desynchronises the
        # collective order across ranks (a validation-end NCCL deadlock otherwise).
        import json
        from pathlib import Path

        rec: dict[str, Any] = {"tag": tag, "epoch": int(getattr(trainer, "current_epoch", -1)), "step": int(getattr(trainer, "global_step", -1))}
        for k, v in trainer.callback_metrics.items():
            try:
                rec[k] = float(v)
            except (TypeError, ValueError):  # pragma: no cover
                continue
        if not getattr(trainer, "is_global_zero", True):
            return
        p = Path(_per_run_path(self.path, trainer))
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        self._dump(trainer, "train_epoch_end")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        if getattr(trainer, "sanity_checking", False):
            return
        self._dump(trainer, "validation_epoch_end")


class GpuMemoryLogger(Callback):
    """Log the peak GPU memory (``torch.cuda.max_memory_allocated`` / ``max_memory_reserved``, GiB) at
    every train/validation epoch end (Lightning: ``gpu_mem/max_allocated_gib`` reduced with ``max`` over
    ranks) and print it every ``print_every`` training batches . Harmless (no-op) without CUDA.
    """

    def __init__(self, print_every: int = 50) -> None:
        super().__init__()
        self.print_every = max(1, int(print_every))

    @staticmethod
    def _peak() -> tuple[float, float] | None:
        if not torch.cuda.is_available():
            return None
        gib = float(2**30)
        return torch.cuda.max_memory_allocated() / gib, torch.cuda.max_memory_reserved() / gib

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:  # noqa: ARG002
        peak = self._peak()
        if peak is not None and batch_idx % self.print_every == 0:
            LOGGER.info("GPU peak memory (rank %d): allocated %.2f GiB, reserved %.2f GiB", int(getattr(trainer, "global_rank", 0)), *peak)

    def _log(self, pl_module: pl.LightningModule) -> None:
        peak = self._peak()
        if peak is None:
            return
        pl_module.log("gpu_mem/max_allocated_gib", peak[0], on_step=False, on_epoch=True, sync_dist=True, reduce_fx="max", logger=getattr(pl_module, "logger_enabled", True))
        pl_module.log("gpu_mem/max_reserved_gib", peak[1], on_step=False, on_epoch=True, sync_dist=True, reduce_fx="max", logger=getattr(pl_module, "logger_enabled", True))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        self._log(pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        self._log(pl_module)
