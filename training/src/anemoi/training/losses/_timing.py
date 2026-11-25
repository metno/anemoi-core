# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
"""Opt-in wall-clock timing of the loss terms.

Off by default (one attribute check per section). ``ANEMOI_LOSS_TIMING=1`` synchronises the GPU around every section and
prints, on rank 0, the mean seconds per training step: the whole step (loss start to loss start), every loss term and the
sections inside the multiscale wrapper. The first ``ANEMOI_LOSS_TIMING_WARMUP`` steps (default 5: compilation, allocator
warm-up) are left out. The synchronisation itself slows the step a little; compare variants measured the same way.
"""
import os
import time

import torch

ENABLED = os.environ.get("ANEMOI_LOSS_TIMING", "0") == "1"
WARMUP = int(os.environ.get("ANEMOI_LOSS_TIMING_WARMUP", "5"))
EVERY = int(os.environ.get("ANEMOI_LOSS_TIMING_EVERY", "5"))
_RANK0 = os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", "0")) == "0"
_acc: dict[str, float] = {}
_state = {"step": 0, "t_prev": None, "n": 0}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class section:  # noqa: N801
    """``with section("name"):`` accumulates the wall time of the block under ``name``."""

    __slots__ = ("name", "t0")

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "section":
        if ENABLED:
            _sync()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if ENABLED:
            _sync()
            _acc[self.name] = _acc.get(self.name, 0.0) + time.perf_counter() - self.t0


def step() -> None:
    """Mark the start of the loss evaluation of one training step."""
    if not ENABLED:
        return
    _sync()
    now = time.perf_counter()
    k = _state["step"]
    if k == WARMUP:
        _acc.clear()
        _state["n"] = 0
    elif k > WARMUP:
        _acc["step"] = _acc.get("step", 0.0) + now - _state["t_prev"]
        _state["n"] += 1
        n = _state["n"]
        if _RANK0 and n % EVERY == 0:
            parts = [f"{name} {value / n:.3f}" for name, value in _acc.items() if name != "step"]
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            print(
                f"LOSS_TIMING steps {WARMUP + 1}-{k}: step {_acc['step'] / n:.3f} s | peak {peak:.1f} GiB | " + " | ".join(parts),
                flush=True,
            )
    _state["t_prev"] = now
    _state["step"] = k + 1


# ---------------------------------------------------------------- per-term loss values (calibration)
VALUES = os.environ.get("ANEMOI_LOSS_TERM_VALUES", "0") == "1"
_values: dict[str, list] = {}


def record_value(label: str, value: torch.Tensor, weight: float) -> None:
    """Accumulate the unweighted value of one loss term over validation batches (``ANEMOI_LOSS_TERM_VALUES=1``).

    Prints, on rank 0, the running mean of every term and of ``weight * term`` every 10 batches.
    """
    entry = _values.setdefault(label, [0.0, 0, float(weight)])
    entry[0] += float(value.detach().float().sum())
    entry[1] += 1
    last = list(_values)[-1]  # the terms are recorded in order: print once the last one has seen the batch
    if _RANK0 and label == last and entry[1] % 10 == 0:
        for name, (total, n, w) in _values.items():
            print(f"LOSS_TERM_VALUE batches {n}: {name} | value {total / n:.6g} | weight {w:g} | weighted {w * total / n:.6g}", flush=True)
