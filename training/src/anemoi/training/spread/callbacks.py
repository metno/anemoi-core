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
