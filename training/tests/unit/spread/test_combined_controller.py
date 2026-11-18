# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for anemoi.training.spread.callbacks.CombinedAdaptiveSpreadController.

Uses the FakeModule/FakeTrainer stand-ins of test_diagnostics_callbacks and the toy smoothers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.training.losses.loss import get_loss_function
from anemoi.training.spread import TrainingDiagnosticsLossWrapper
from anemoi.training.spread.callbacks import CombinedAdaptiveSpreadController
from .test_diagnostics_callbacks import FakeModule
from .test_diagnostics_callbacks import FakeTrainer

NAMES = {"v0": 0, "v1": 1, "v2": 2}


def _stack(toy_smoother_dir: Path, n_grid: int, pins: dict | None = None) -> TrainingDiagnosticsLossWrapper:
    inner = {
        "_target_": "anemoi.training.spread.MultiscaleSpreadLossWrapper",
        "weights": [1.0, 1.0, 1.0],
        "per_scale_spread_adjustment": [1.0, 1.0, 1.0],
        "multiscale_config": {
            "loss_matrices_path": str(toy_smoother_dir),
            "loss_matrices": ["coarse.npz", "fine.npz", None],
        },
        "per_scale_loss": {
            "_target_": "anemoi.training.spread.SpreadScaledCRPS",
            "alpha": 0.95,
            "scalers": ["node_weights"],
            **({"per_variable_spread_adjustment": pins} if pins else {}),
        },
    }
    cfg = DictConfig({"_target_": "anemoi.training.spread.TrainingDiagnosticsLossWrapper", "loss": inner})
    scalers = {"node_weights": (3, torch.full((n_grid,), 1.0 / n_grid))}
    wrapped = get_loss_function(cfg, scalers=scalers, data_indices=None)
    wrapped.loss._spread_leaves[0].resolve_variable_spread_adjustment(NAMES)
    return wrapped


def _controller(**kwargs) -> CombinedAdaptiveSpreadController:
    defaults = {"update_interval": 1, "warmup_steps": 0, "min_ssr": 0.0, "ema_horizon": 1, "freeze_steps": 5}
    defaults.update(kwargs)
    return CombinedAdaptiveSpreadController(**defaults)


def _drive_batch(wrapped, module, trainer, ctrl, y, members, n_steps=2) -> None:  # noqa: ANN001
    """One fake training batch: forward per rollout step with the given member fields, then batch end."""
    for r in range(n_steps):
        x = np.stack(members, axis=2).astype(np.float32)
        wrapped(torch.from_numpy(x), torch.from_numpy(y), rollout_step=r)
    ctrl.on_train_batch_end(trainer, module, None, {"data": torch.zeros(2, 1)}, 0)
    wrapped.drain()


@pytest.fixture
def target(n_grid: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((2, 1, 1, n_grid, 3)).astype(np.float32)


def _underdispersed(y: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Members hugging each other far from the target — white fields, so every band sees SSR << 1."""
    base = 0.5 * rng.standard_normal(y[:, :, 0].shape)
    return [y[:, :, 0] + base + 0.001 * rng.standard_normal(base.shape) for _ in range(2)]


def _overdispersed(y: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Members far apart around the target — SSR > 1 in every band."""
    return [y[:, :, 0] + 3.0 * rng.standard_normal(y[:, :, 0].shape) for _ in range(2)]


def test_underdispersed_raises_table_and_ar(toy_smoother_dir: Path, n_grid: int, target: np.ndarray) -> None:
    wrapped = _stack(toy_smoother_dir, n_grid)
    module = FakeModule(wrapped, n_steps=2)
    trainer = FakeTrainer()
    trainer.global_step = 0
    ctrl = _controller(max_update_frac=0.015)
    ctrl.on_train_start(trainer, module)
    assert ctrl.table is not None and bool(ctrl.controlled.all())

    # tiny spread, O(1) error -> SSR << 1 -> every factor at the +1.5% clamp
    rng = np.random.default_rng(7)
    trainer.global_step = 1
    _drive_batch(wrapped, module, trainer, ctrl, target, _underdispersed(target, rng))
    assert ctrl._n_updates == 1
    np.testing.assert_allclose(ctrl.table.numpy(), 1.015, rtol=1e-6)
    assert [len(row) for row in ctrl.a_br] == [1, 1, 1]  # step r=1 seen, per band (r4)
    np.testing.assert_allclose(ctrl.a_br, [[1.015]] * 3, rtol=1e-6)
    # and the wrapper received the updated state
    applied = wrapped.loss.get_per_variable_band_table()
    np.testing.assert_allclose(applied["v0"], [1.015] * 3, rtol=1e-6)
    rows = wrapped.loss.get_per_rollout_step_adjustment()
    assert len(rows) == 3
    for row in rows:
        assert row == pytest.approx([1.0, 1.015])


def test_overdispersed_lowers_table_and_clamps(toy_smoother_dir: Path, n_grid: int, target: np.ndarray) -> None:
    wrapped = _stack(toy_smoother_dir, n_grid)
    module = FakeModule(wrapped, n_steps=1)
    trainer = FakeTrainer()
    ctrl = _controller(table_clamp=(0.99, 1.3))
    ctrl.on_train_start(trainer, module)
    # huge spread around the target -> SSR >> 1 -> factors at the lower per-update clamp,
    # then the table floor
    rng = np.random.default_rng(7)
    for i in range(1, 4):
        trainer.global_step = i
        _drive_batch(wrapped, module, trainer, ctrl, target, _overdispersed(target, rng), n_steps=1)
    assert ctrl._n_updates == 3
    assert float(ctrl.table.max()) == pytest.approx(0.99)  # clamped at the configured floor
    assert ctrl.a_br == [[], [], []]  # single rollout step: no a_{b,r} controlled


def test_warmup_and_freeze_gate_updates(toy_smoother_dir: Path, n_grid: int, target: np.ndarray) -> None:
    wrapped = _stack(toy_smoother_dir, n_grid)
    module = FakeModule(wrapped, n_steps=1)
    trainer = FakeTrainer()
    ctrl = _controller(warmup_steps=100)
    ctrl.on_train_start(trainer, module)
    rng = np.random.default_rng(7)
    trainer.global_step = 5  # below warmup
    _drive_batch(wrapped, module, trainer, ctrl, target, _underdispersed(target, rng), n_steps=1)
    assert ctrl._n_updates == 0
    np.testing.assert_allclose(ctrl.table.numpy(), 1.0)

    # rollout-phase change freezes updates
    ctrl2 = _controller(freeze_steps=50)
    module2 = FakeModule(wrapped, n_steps=1)
    ctrl2.on_train_start(trainer, module2)
    trainer.global_step = 1
    _drive_batch(wrapped, module2, trainer, ctrl2, target, _underdispersed(target, rng), n_steps=1)
    assert ctrl2._n_updates == 1
    module2.task.steps = lambda mode="training": ({"rollout_step": 0}, {"rollout_step": 1})  # rollout 1 -> 2
    trainer.global_step = 2
    _drive_batch(wrapped, module2, trainer, ctrl2, target, _underdispersed(target, rng), n_steps=2)
    assert ctrl2._n_updates == 1  # frozen
    assert ctrl2._freeze_until == 2 + 50


def test_pinned_variable_never_controlled(toy_smoother_dir: Path, n_grid: int, target: np.ndarray) -> None:
    wrapped = _stack(toy_smoother_dir, n_grid, pins={"v2": 1.0})
    module = FakeModule(wrapped, n_steps=1)
    trainer = FakeTrainer()
    ctrl = _controller()
    ctrl.on_train_start(trainer, module)
    assert ctrl.controlled.tolist() == [True, True, False]
    rng = np.random.default_rng(7)
    trainer.global_step = 1
    _drive_batch(wrapped, module, trainer, ctrl, target, _underdispersed(target, rng), n_steps=1)
    table = ctrl.table.numpy()
    np.testing.assert_allclose(table[:2], 1.015, rtol=1e-6)
    np.testing.assert_allclose(table[2], 1.0)  # pinned row untouched
    assert "v2" not in wrapped.loss.get_per_variable_band_table()
    # and the leaf still reports the pin as a static override
    assert wrapped.loss.static_variable_overrides == {"v2": 1.0}


def test_validation_ratio_corrects_signal(toy_smoother_dir: Path, n_grid: int, target: np.ndarray) -> None:
    """A val/train SSR ratio > 1 (val more dispersed) scales the corrected signal accordingly."""
    wrapped = _stack(toy_smoother_dir, n_grid)
    module = FakeModule(wrapped, n_steps=1)
    trainer = FakeTrainer()
    ctrl = _controller()
    ctrl.on_train_start(trainer, module)
    # train side: underdispersed
    rng = np.random.default_rng(7)
    members = _underdispersed(target, rng)
    trainer.global_step = 1
    _drive_batch(wrapped, module, trainer, ctrl, target, members, n_steps=1)
    # validation pass: same fields -> ratio == 1
    x = np.stack(members, axis=2).astype(np.float32)
    wrapped(torch.from_numpy(x), torch.from_numpy(target), rollout_step=0)
    ctrl.on_validation_batch_end(trainer, module, None, None, 0)
    wrapped.drain()
    ctrl.on_validation_epoch_end(trainer, module)
    # val-end only STASHES (no collectives there — NCCL deadlock 2026-08-25); the ratio
    # materialises at the next training-side update
    assert ctrl._pending_val is not None and (0, 0) not in ctrl._ratio
    trainer.global_step = 2
    _drive_batch(wrapped, module, trainer, ctrl, target, members, n_steps=1)
    assert ctrl._pending_val is None
    assert (0, 0) in ctrl._ratio
    np.testing.assert_allclose(ctrl._ratio[(0, 0)].numpy(), 1.0, rtol=1e-5)


def test_state_roundtrip(toy_smoother_dir: Path, n_grid: int, target: np.ndarray) -> None:
    wrapped = _stack(toy_smoother_dir, n_grid)
    module = FakeModule(wrapped, n_steps=2)
    trainer = FakeTrainer()
    ctrl = _controller()
    ctrl.on_train_start(trainer, module)
    rng = np.random.default_rng(7)
    trainer.global_step = 1
    _drive_batch(wrapped, module, trainer, ctrl, target, _underdispersed(target, rng))
    state = ctrl.state_dict()

    ctrl2 = _controller()
    ctrl2.load_state_dict(state)
    ctrl2.on_train_start(trainer, module)
    np.testing.assert_allclose(ctrl2.table.numpy(), ctrl.table.numpy())
    assert ctrl2.a_br == ctrl.a_br
    assert set(ctrl2._ema_var) == set(ctrl._ema_var)
    # the restored state is re-applied to the wrapper
    np.testing.assert_allclose(
        wrapped.loss.get_per_variable_band_table()["v0"], [float(v) for v in ctrl.table[0]], rtol=1e-6
    )


# --------------------------------------------------------------------------- r4: band clamps + seeding
def test_band_scoped_table_clamps(toy_smoother_dir: Path, n_grid: int, target: np.ndarray) -> None:
    """Per-band (lo, hi) pairs bound each band's column independently."""
    wrapped = _stack(toy_smoother_dir, n_grid)
    module = FakeModule(wrapped, n_steps=1)
    trainer = FakeTrainer()
    ctrl = _controller(table_clamp=[[0.9, 1.02], [0.9, 1.6], [0.9, 1.05]])
    ctrl.on_train_start(trainer, module)
    rng = np.random.default_rng(7)
    for i in range(1, 6):  # underdispersed -> +1.5%/update everywhere until a band cap binds
        trainer.global_step = i
        _drive_batch(wrapped, module, trainer, ctrl, target, _underdispersed(target, rng), n_steps=1)
    table = ctrl.table.numpy()
    np.testing.assert_allclose(table[:, 0], 1.02, rtol=1e-6)  # coarse capped
    np.testing.assert_allclose(table[:, 1], 1.015**5, rtol=1e-6)  # mid still free
    np.testing.assert_allclose(table[:, 2], 1.05, rtol=1e-6)  # fine capped


def test_seed_from_dump(toy_smoother_dir: Path, n_grid: int, tmp_path: Path) -> None:
    """Seeding from a previous run's dump: legacy flat a_r broadcasts, rollout extension repeats the
    last value, and seeded table entries are clipped into the (new) band clamps."""
    import json

    seed = tmp_path / "seed_controller.jsonl"
    recs = [
        {"global_step": 250, "table": {"v0": [1.0, 1.0, 1.0]}, "a_r": [1.0]},
        {
            "global_step": 20000,
            "table": {"v0": [1.2, 1.26, 1.28], "v1": [1.0, 1.1, 1.05]},
            "a_r": [1.06, 0.95, 0.93],
        },
    ]
    seed.write_text("\n".join(json.dumps(r) for r in recs) + "\n")

    wrapped = _stack(toy_smoother_dir, n_grid)
    module = FakeModule(wrapped, n_steps=1)
    trainer = FakeTrainer()
    ctrl = _controller(
        table_clamp=[[0.9, 2.0], [0.9, 1.6], [0.9, 1.1]],
        seed_dump_path=str(seed),
        seed_extend_rollout=8,
    )
    ctrl.on_train_start(trainer, module)
    table = ctrl.table.numpy()
    np.testing.assert_allclose(table[0], [1.2, 1.26, 1.1], rtol=1e-6)  # fine 1.28 clipped to 1.1
    np.testing.assert_allclose(table[1], [1.0, 1.1, 1.05], rtol=1e-6)
    np.testing.assert_allclose(table[2], 1.0)  # absent from the seed -> identity
    assert ctrl.a_br == [[1.06, 0.95, 0.93, 0.93, 0.93, 0.93, 0.93]] * 3  # broadcast + extended
    rows = wrapped.loss.get_per_rollout_step_adjustment()
    assert len(rows) == 3
    for row in rows:
        assert row == pytest.approx([1.0, 1.06, 0.95, 0.93, 0.93, 0.93, 0.93, 0.93])


def test_legacy_ar_checkpoint_broadcasts(toy_smoother_dir: Path, n_grid: int) -> None:
    """A legacy checkpoint state with a flat shared a_r loads as band-broadcast a_br."""
    wrapped = _stack(toy_smoother_dir, n_grid)
    module = FakeModule(wrapped, n_steps=1)
    trainer = FakeTrainer()
    ctrl = _controller()
    ctrl.load_state_dict({"table": [[1.1] * 3] * 3, "a_r": [1.05, 0.97]})
    ctrl.on_train_start(trainer, module)
    assert ctrl.a_br == [[1.05, 0.97]] * 3
    np.testing.assert_allclose(ctrl.table.numpy(), 1.1)


def test_band_scoped_rollout_clamps(toy_smoother_dir: Path, n_grid: int, target: np.ndarray) -> None:
    """Per-band (lo, hi) pairs for a_{b,r}: the coarse and mid
    rows are capped at 1.0 while the fine row keeps the usual room; a global pair still applies to every band."""
    wrapped = _stack(toy_smoother_dir, n_grid)
    module = FakeModule(wrapped, n_steps=2)
    trainer = FakeTrainer()
    ctrl = _controller(rollout_clamp=[[0.9, 1.0], [0.9, 1.0], [0.9, 1.2]])
    ctrl.on_train_start(trainer, module)
    rng = np.random.default_rng(11)
    for i in range(1, 6):  # underdispersed at both steps -> a_{b,1} wants +1.5 %/update in every band
        trainer.global_step = i
        _drive_batch(wrapped, module, trainer, ctrl, target, _underdispersed(target, rng), n_steps=2)
    a_br = ctrl.a_br
    assert a_br[0][0] == pytest.approx(1.0) and a_br[1][0] == pytest.approx(1.0)  # coarse/mid capped at 1.0
    assert a_br[2][0] == pytest.approx(1.015**5, rel=1e-6)  # fine still free below 1.2
    # a global pair behaves as before
    ctrl2 = _controller(rollout_clamp=[0.9, 1.2])
    ctrl2.on_train_start(trainer, FakeModule(_stack(toy_smoother_dir, n_grid), n_steps=2))
    assert ctrl2._rollout_bounds(0) == (0.9, 1.2) and ctrl2._rollout_bounds(2) == (0.9, 1.2)
    with pytest.raises(ValueError, match="rollout_clamp"):
        c3 = _controller(rollout_clamp=[[0.9, 1.0], [0.9, 1.2]])
        c3.on_train_start(trainer, FakeModule(_stack(toy_smoother_dir, n_grid), n_steps=2))


def test_controller_ssr_target_knob() -> None:
    c = CombinedAdaptiveSpreadController()
    assert c.ssr_target == [1.0] and c._target(0) == 1.0 and c._target(3) == 1.0
    c = CombinedAdaptiveSpreadController(ssr_target=[1.0, 1.0, 1.0, 1.03])
    assert c._target(0) == 1.0 and c._target(3) == 1.03 and c._target(7) == 1.03
    with pytest.raises(ValueError):
        CombinedAdaptiveSpreadController(ssr_target=[1.0, 0.0])
