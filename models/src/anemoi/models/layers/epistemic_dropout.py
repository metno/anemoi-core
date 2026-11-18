# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Epistemic dropout for the graph-transformer processor (WeatherNext 3, Rasp et al. 2026, §2.2).

Semantics
---------
* Training: ONE inverted-dropout mask per (global_step, rollout_step, model-shard rank, layer, site),
  SHARED across the ensemble members of that step.  The seed deliberately EXCLUDES any ensemble-member or
  ensemble-group index, so members living on different GPUs of one ensemble group draw identical masks for
  identical shard shapes; the mask has shape (n_local_nodes, channels) and broadcasts over the
  (batch x ensemble) row fold of the processor latent, so members sharing a device share it too.
* Inference: the same machinery with ``armed=True`` and a per-member ``member_key`` (plus the forecast step as
  ``rollout_step``), i.e. masks independent per member and per lead.
* Sites: 0 = attention-output residual branch, 1 = MLP-output residual branch (one mask each, per layer).

Engineering constraints (see the mspread implementation notes, 2026-09-04)
* The mask is drawn from a PRIVATE ``torch.Generator`` that is re-seeded on EVERY draw: the recompute inside
  ``torch.utils.checkpoint`` sees the same mask, and the global RNG (which drives the latent-noise injector) is
  never touched.
* The draw and every read of the mutable step state happen inside a ``torch.compiler.disable`` function, so
  ``torch.compile`` never guards on ``global_step`` (no per-step recompiles); at ``dropout_p == 0`` the caller
  never enters this module and the compiled graph is unchanged (bitwise-identical forward, test t-a).
* State lives in a plain Python object attached as ``_epistemic_dropout`` (NOT an ``nn.Module``): checkpoints
  pickled before this feature simply lack the attribute and behave as ``dropout_p = 0``; the helpers below
  attach it post-load for inference.
"""

from __future__ import annotations

import numpy as np
import torch

SITE_ATTENTION = 0
SITE_MLP = 1

try:  # torch >= 2.1
    _compiler_disable = torch.compiler.disable
except AttributeError:  # pragma: no cover - very old torch
    _compiler_disable = torch._dynamo.disable


class EpistemicDropoutState:
    """Mutable per-forward step state shared by every processor block (plain object, picklable)."""

    __slots__ = ("armed", "global_step", "rollout_step", "member_key", "base_seed")

    def __init__(self, base_seed: int = 0) -> None:
        self.armed = False
        self.global_step = 0
        self.rollout_step = 0
        self.member_key = 0
        self.base_seed = int(base_seed)

    def set(self, *, global_step: int, rollout_step: int, armed: bool, member_key: int = 0) -> None:
        self.global_step = int(global_step)
        self.rollout_step = int(rollout_step)
        self.armed = bool(armed)
        self.member_key = int(member_key)


def dropout_seed(
    base_seed: int, global_step: int, rollout_step: int, shard_rank: int, layer_index: int, site: int, member_key: int = 0
) -> int:
    """Deterministic 32-bit seed; same arithmetic as ``anemoi.training.utils.seeding.derive_seed``.

    ``member_key`` is 0 in training (mask shared across members) and the member index at inference.
    """
    seq = np.random.SeedSequence(
        entropy=int(base_seed), spawn_key=(7, int(global_step), int(rollout_step), int(shard_rank), int(layer_index), int(site), int(member_key))
    )
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def draw_mask(p: float, shape: tuple[int, ...], seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Inverted-dropout mask (keep prob 1-p, scaled by 1/(1-p)) from a private, freshly seeded generator."""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    keep = torch.rand(shape, generator=gen, device=device) >= float(p)
    return keep.to(dtype) / (1.0 - float(p))


@_compiler_disable
def apply_site(module: torch.nn.Module, x: torch.Tensor, site: int, model_comm_group=None, batch_size: int = 1) -> torch.Tensor:
    """Apply the (step, shard, layer, site) mask to ``x`` of shape (rows, channels) if the state is armed.

    Reads ALL mutable state here (compiler-disabled) so dynamo never guards on it.  ``x`` rows are the
    (batch x ensemble x local-node) fold with ``batch_size`` = batch x ensemble row-groups; the mask is
    (local-node, channels) and repeats over the row-groups.  ``model_comm_group`` may be a process group
    (shard rank = its rank), an int (used as the shard rank directly) or None (rank 0).
    """
    state = getattr(module, "_epistemic_dropout", None)
    p = float(getattr(module, "dropout_p", 0.0))
    if state is None or not state.armed or p <= 0.0:
        return x
    if model_comm_group is None:
        shard_rank = 0
    elif isinstance(model_comm_group, int):
        shard_rank = model_comm_group
    else:
        shard_rank = int(model_comm_group.rank())
    n_rows, channels = x.shape[0], x.shape[-1]
    bs = int(batch_size) if batch_size else 1
    n_local = n_rows // bs if (bs > 0 and n_rows % bs == 0) else n_rows
    seed = dropout_seed(
        state.base_seed, state.global_step, state.rollout_step, shard_rank, int(getattr(module, "_layer_index", 0)), site, state.member_key
    )
    mask = draw_mask(p, (n_local, channels), seed, x.device, x.dtype)
    if n_local != n_rows:
        mask = mask.repeat(n_rows // n_local, 1)
    return x * mask


def attach(processor: torch.nn.Module, dropout_p: float | None = None, base_seed: int = 0) -> EpistemicDropoutState:
    """Give ``processor`` (a GraphTransformerProcessor, possibly unpickled from an old checkpoint) a shared state.

    Sets ``_layer_index`` on every block, optionally overrides ``dropout_p`` on every block, and links one
    :class:`EpistemicDropoutState` to the processor and all blocks.  Idempotent.
    """
    state = getattr(processor, "_epistemic_dropout", None)
    if state is None:
        state = EpistemicDropoutState(base_seed)
        processor._epistemic_dropout = state
    blocks = list(getattr(processor, "proc", []))
    for i, blk in enumerate(blocks):
        blk._layer_index = i
        blk._epistemic_dropout = state
        if dropout_p is not None:
            blk.dropout_p = float(dropout_p)
    return state


def set_step(processor: torch.nn.Module, *, global_step: int, rollout_step: int, armed: bool, member_key: int = 0) -> None:
    """Stamp the current (step, rollout_step, member) on the processor's shared state (attaching if needed)."""
    state = getattr(processor, "_epistemic_dropout", None)
    if state is None:
        state = attach(processor)
    state.set(global_step=global_step, rollout_step=rollout_step, armed=armed, member_key=member_key)
