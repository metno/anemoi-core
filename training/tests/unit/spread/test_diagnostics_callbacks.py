# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for anemoi.training.spread.diagnostics.TrainingDiagnosticsLossWrapper and anemoi.training.spread.callbacks.

The Lightning trainer/module are replaced by tiny fakes exposing exactly the attributes the code
touches (``pl_module.loss``, ``.task.steps``, ``.nens_per_group``, ``.log``; ``trainer.callback_metrics``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from .conftest import add_uniform_node_weights
from anemoi.training.spread import EnsembleMeanSquaredError
from anemoi.training.spread import EnsembleVariance
from anemoi.training.spread import MultiscaleSpreadLossWrapper
from anemoi.training.spread import SpreadScaledCRPS
from anemoi.training.spread import TrainingDiagnosticsLossWrapper
from anemoi.training.spread import spread_skill_ratio
from anemoi.training.spread.callbacks import SpreadSkillLogger


class FakeModule:
    """Minimal stand-in for the anemoi EnsembleTraining LightningModule."""

    def __init__(self, loss, n_steps: int = 1, nens: int = 2) -> None:  # noqa: ANN001
        self.loss = torch.nn.ModuleDict({"data": loss})
        self.task = SimpleNamespace(steps=lambda mode="training": tuple({"rollout_step": i} for i in range(n_steps)))
        self.nens_per_group = nens
        self.logger_enabled = True
        self.logged: dict[str, float] = {}

    def log(self, name, value, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.logged[name] = float(value)


class FakeTrainer:
    def __init__(self) -> None:
        self.callback_metrics: dict[str, torch.Tensor] = {}
        self.sanity_checking = False


def _wrapper_config(smoother_dir: Path | None, a: list[float]) -> DictConfig:
    inner = {
        "_target_": "anemoi.training.spread.MultiscaleSpreadLossWrapper",
        "weights": [1.0] * len(a),
        "per_scale_spread_adjustment": a,
        "multiscale_config": (
            None
            if smoother_dir is None
            else {"loss_matrices_path": str(smoother_dir), "loss_matrices": ["coarse.npz", "fine.npz", None]}
        ),
        "per_scale_loss": {"_target_": "anemoi.training.spread.SpreadScaledCRPS", "alpha": 0.95, "scalers": ["node_weights"]},
    }
    return DictConfig({"_target_": "anemoi.training.spread.TrainingDiagnosticsLossWrapper", "loss": inner})


@pytest.fixture
def scalers(n_grid: int) -> dict:
    return {"node_weights": (3, torch.full((n_grid,), 1.0 / n_grid)), "general_variable": (4, torch.tensor([1.0, 2.0, 3.0]))}


def test_wrapper_is_transparent_and_records(toy_smoother_dir: Path, scalers: dict, n_grid: int) -> None:
    from anemoi.training.losses.loss import get_loss_function

    wrapped = get_loss_function(_wrapper_config(toy_smoother_dir, [1.41, 1.15, 1.0]), scalers=scalers, data_indices=None)
    assert isinstance(wrapped, TrainingDiagnosticsLossWrapper)
    assert isinstance(wrapped.loss, MultiscaleSpreadLossWrapper)
    assert wrapped.name == "MultiscaleSpreadLossWrapper"
    assert set(wrapped.diagnostics) == {"ensvar", "ensmse"}
    # the diagnostics carry only node_weights, never the loss' variable scalers
    for m in wrapped.diagnostics.values():
        assert "node_weights" in m.scaler and "general_variable" not in m.scaler

    ref = get_loss_function(_wrapper_config(toy_smoother_dir, [1.41, 1.15, 1.0]).loss, scalers=scalers, data_indices=None)
    x = torch.randn(2, 1, 2, n_grid, 3)
    y = torch.randn(2, 1, 1, n_grid, 3)
    assert torch.equal(wrapped(x, y), ref(x, y))  # transparent
    assert len(wrapped.pending) == 1
    ev, em = EnsembleVariance(), EnsembleMeanSquaredError()
    add_uniform_node_weights(ev, n_grid)
    add_uniform_node_weights(em, n_grid)
    torch.testing.assert_close(wrapped.pending[0]["ensvar"], ev(x, y, squash=False))
    torch.testing.assert_close(wrapped.pending[0]["ensmse"], em(x, y, squash=False))
    assert wrapped.pending[0]["ensvar"].requires_grad is False
    # gradient flows through the transparent path
    xg = x.clone().requires_grad_(True)
    wrapped(xg, y).backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    # drain empties
    assert len(wrapped.drain()) == 2 and wrapped.pending == []
    # disabled → pass-through only
    wrapped.enabled = False
    wrapped(x, y)
    assert wrapped.pending == []


def test_wrapper_rejects_unknown_kwargs(scalers: dict) -> None:
    from anemoi.training.losses.loss import get_loss_function

    from hydra.errors import InstantiationException

    cfg = _wrapper_config(None, [1.2])
    cfg["typo"] = 1
    with pytest.raises((TypeError, InstantiationException), match="typo"):  # hydra wraps the TypeError
        get_loss_function(cfg, scalers=scalers, data_indices=None)


def test_logger_logs_first_n_rollout_entries_and_ssr(scalers: dict, n_grid: int) -> None:
    from anemoi.training.losses.loss import get_loss_function

    wrapped = get_loss_function(_wrapper_config(None, [1.3]), scalers=scalers, data_indices=None)
    wrapped.variable_names = ["z_500", "tp", "t_850"]  # what data_indices would have provided
    module = FakeModule(wrapped, n_steps=2)
    trainer = FakeTrainer()
    cb = SpreadSkillLogger(variables=["z_500", "t_850", "not_a_var"])
    batch = {"data": torch.zeros(3, 4, 1, n_grid, 5)}

    cb.on_train_batch_start(trainer, module, batch, 0)
    gen = torch.Generator().manual_seed(3)
    xs = [torch.randn(3, 1, 2, n_grid, 3, generator=gen) for _ in range(2)]
    ys = [torch.randn(3, 1, 1, n_grid, 3, generator=gen) for _ in range(2)]
    for x, y in zip(xs, ys, strict=True):  # forward: rollout steps 0, 1
        wrapped(x, y)
    for x, y in zip(reversed(xs), reversed(ys), strict=True):  # checkpoint recompute (reverse order)
        wrapped(x, y)
    assert len(wrapped.pending) == 4
    cb.on_train_batch_end(trainer, module, None, batch, 0)
    assert len(wrapped.pending) == 4  # peek semantics: cleared at the next batch start (several readers)
    cb.on_train_batch_start(trainer, module, batch, 1)
    assert wrapped.pending == []

    ev, em = EnsembleVariance(), EnsembleMeanSquaredError()
    add_uniform_node_weights(ev, n_grid)
    add_uniform_node_weights(em, n_grid)
    for step, (x, y) in enumerate(zip(xs, ys, strict=True), start=1):
        v, e = ev(x, y, squash=False), em(x, y, squash=False)
        assert module.logged[f"train_data_ensvar_metric/data/z_500/{step}"] == pytest.approx(v[0].item(), rel=1e-6)
        assert module.logged[f"train_data_ensmse_metric/data/t_850/{step}"] == pytest.approx(e[2].item(), rel=1e-6)
        assert module.logged[f"train_data_ensvar_metric/data/all/{step}"] == pytest.approx(v.mean().item(), rel=1e-6)
    assert not any("not_a_var" in k or "/tp/" in k for k in module.logged)
    assert not any(k.endswith("/3") for k in module.logged)  # recompute entries were dropped

    # epoch end: SSR from the (fake) epoch means, sqrt of means
    trainer.callback_metrics = {k: torch.tensor(v) for k, v in module.logged.items()}
    trainer.callback_metrics["val_data_ensvar_metric/data/z_500/1"] = torch.tensor(0.5)
    trainer.callback_metrics["val_data_ensmse_metric/data/z_500/1"] = torch.tensor(1.0)
    trainer.callback_metrics["val_data_ensvar_metric/data/orphan/1"] = torch.tensor(0.5)  # no mse partner
    trainer.callback_metrics["val_data_ensvar_metric/data/all/1"] = torch.tensor(0.5)  # multi-variable range: skipped
    trainer.callback_metrics["val_data_ensmse_metric/data/all/1"] = torch.tensor(1.0)
    cb.on_train_epoch_end(trainer, module)
    cb.on_validation_epoch_end(trainer, module)
    v1 = module.logged["train_data_ensvar_metric/data/z_500/1"]
    e1 = module.logged["train_data_ensmse_metric/data/z_500/1"]
    assert module.logged["train_data_ssr/data/z_500/1"] == pytest.approx(spread_skill_ratio(v1, e1, 2))
    assert module.logged["val_data_ssr/data/z_500/1"] == pytest.approx(np.sqrt(1.5 * 0.5 / 1.0))
    assert "val_data_ssr/data/orphan/1" not in module.logged
    assert "val_data_ssr/data/all/1" not in module.logged  # physical-unit average over variables is not an SSR
    assert "train_data_ssr/data/all/1" in module.logged  # normalised-space summary is kept on the train side
    assert cb.last_val_ssr == {"val_data_ssr/data/z_500/1": pytest.approx(np.sqrt(0.75))}


def test_logger_validation_side_values_only_feed_the_epoch_dump(scalers: dict, n_grid: int) -> None:
    from anemoi.training.losses.loss import get_loss_function

    wrapped = get_loss_function(_wrapper_config(None, [1.0]), scalers=scalers, data_indices=None)
    module, trainer, cb = FakeModule(wrapped), FakeTrainer(), SpreadSkillLogger()
    cb.on_validation_batch_start(trainer, module, {}, 0)
    wrapped(torch.randn(1, 1, 2, n_grid, 2), torch.randn(1, 1, 1, n_grid, 2))
    cb.on_validation_batch_end(trainer, module, None, {}, 0)
    assert module.logged == {}  # nothing logged per batch on the validation side
    assert cb._acc["val"].counts == {("data", 1): 1}  # ... but accumulated for the per-variable epoch dump
    cb.on_train_batch_start(trainer, module, {}, 0)  # next batch: pending cleared
    assert wrapped.pending == []
    cb.on_train_batch_end(trainer, module, None, {"data": torch.zeros(1, 1, 1, 1, 1)}, 0)
    assert module.logged == {}


def test_callback_metrics_dump(tmp_path: Path) -> None:
    import json

    from anemoi.training.spread.callbacks import CallbackMetricsDump

    trainer = FakeTrainer()
    trainer.is_global_zero = True
    trainer.current_epoch, trainer.global_step = 3, 120
    trainer.callback_metrics = {"val_data_ssr/data/z_500/1": torch.tensor(0.9), "junk": "x"}
    cb = CallbackMetricsDump(str(tmp_path / "sub" / "m.jsonl"))
    cb.on_validation_epoch_end(trainer, None)
    trainer.sanity_checking = True
    cb.on_validation_epoch_end(trainer, None)  # skipped
    cb.on_train_epoch_end(trainer, None)
    lines = [json.loads(l) for l in (tmp_path / "sub" / "m.jsonl").read_text().splitlines()]
    assert [l["tag"] for l in lines] == ["validation_epoch_end", "train_epoch_end"]
    assert lines[0]["val_data_ssr/data/z_500/1"] == pytest.approx(0.9) and lines[0]["epoch"] == 3 and "junk" not in lines[0]
