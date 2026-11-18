# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for anemoi.training.spread.MultiscaleSpreadLossWrapper.

Toy geometry from conftest: 1-D periodic ring, two row-normalised moving-average smoothers
(coarse width 15, fine width 5) stored as scipy .npz → file mode of MultiscaleLossWrapper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.losses import CRPS
from anemoi.training.losses import MultiscaleLossWrapper
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_loss_function
from .conftest import add_uniform_node_weights
from anemoi.training.spread import MultiscaleSpreadLossWrapper
from anemoi.training.spread import SpreadScaledCRPS
from .test_spread_scaled_crps import m2_reference


class _RecordingLoss(BaseLoss):
    """Test helper: records the (pred, target) tensors it is called with and returns 0."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, pred, target, squash=True, **kwargs):  # noqa: ANN001, ANN003, ANN201, ARG002
        self.calls.append((pred.clone(), target.clone()))
        return torch.zeros((), dtype=pred.dtype)


def _file_multiscale_config(smoother_dir: Path) -> dict:
    return {"loss_matrices_path": str(smoother_dir), "loss_matrices": ["coarse.npz", "fine.npz", None]}


def _bands_numpy(field: np.ndarray, smoothers: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Telescoping bands D1 f, D2 f − D1 f, f − D2 f for a field of shape (..., grid, v).

    Smoothers act on the grid axis (second to last).
    """
    d1 = np.einsum("ij,...jv->...iv", smoothers["coarse"], field)
    d2 = np.einsum("ij,...jv->...iv", smoothers["fine"], field)
    return [d1, d2 - d1, field - d2]


@pytest.fixture
def batch(n_grid: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.standard_normal((2, 1, 2, n_grid, 3)).astype(np.float32)
    y = rng.standard_normal((2, 1, 1, n_grid, 3)).astype(np.float32)
    return x, y


# --------------------------------------------------------------------------- (iii) band identity
def test_band_decomposition_sums_to_identity(toy_smoother_dir: Path, toy_smoothers: dict, batch: tuple, n_grid: int) -> None:
    x, y = batch
    rec = _RecordingLoss()
    wrapper = MultiscaleLossWrapper(
        per_scale_loss=rec,
        weights=[1.0, 1.0, 1.0],
        multiscale_config=_file_multiscale_config(toy_smoother_dir),
    )
    assert wrapper.num_scales == 3
    wrapper(torch.from_numpy(x), torch.from_numpy(y))
    assert len(rec.calls) == 3
    preds = [c[0].numpy() for c in rec.calls]
    targs = [c[1].numpy() for c in rec.calls]
    np.testing.assert_allclose(sum(preds), x, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sum(targs), y, rtol=1e-5, atol=1e-6)
    # and the bands are exactly the telescoping differences of the toy smoothers
    for got, ref in zip(preds, _bands_numpy(x, toy_smoothers), strict=True):
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)
    # row-normalised smoothers: a constant field lives entirely in the coarsest band
    rec.calls.clear()
    const = np.full_like(x, 3.0)
    wrapper(torch.from_numpy(const), torch.from_numpy(const[:, :, :1]))
    np.testing.assert_allclose(rec.calls[0][0].numpy(), const, rtol=1e-5, atol=1e-5)
    for c in rec.calls[1:]:
        np.testing.assert_allclose(c[0].numpy(), 0.0, atol=1e-5)


# --------------------------------------------------------------------------- per-band a
def test_per_band_a_matches_numpy_reference(toy_smoother_dir: Path, toy_smoothers: dict, batch: tuple, n_grid: int) -> None:
    """wrapper loss == Σ_i w_i · SpreadScaledCRPS(a_i)(band_i(x), band_i(y)), bands & CRPS recomputed in numpy."""
    x, y = batch
    alpha = 0.95
    weights = [1.0, 0.7, 0.5]
    a_per_band = [1.41, 1.15, 1.0]
    leaf = SpreadScaledCRPS(alpha=alpha, spread_adjustment=1.0)
    wrapper = MultiscaleSpreadLossWrapper(
        per_scale_loss=leaf,
        weights=weights,
        per_scale_spread_adjustment=a_per_band,
        multiscale_config=_file_multiscale_config(toy_smoother_dir),
    )
    add_uniform_node_weights(wrapper, n_grid)  # delegated to the leaf
    out = wrapper(torch.from_numpy(x), torch.from_numpy(y), squash=False).numpy()

    ref = np.zeros(3)
    for w, a, bx, by in zip(weights, a_per_band, _bands_numpy(x, toy_smoothers), _bands_numpy(y, toy_smoothers), strict=True):
        pt = m2_reference(bx[:, :, 0], bx[:, :, 1], by[:, :, 0], alpha, a)  # (bs, t, g, v)
        ref += w * pt.mean(axis=2).sum(axis=1).mean(axis=0)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)
    # scalar (squash=True, CRPS squash_mode="sum") is the sum over variables
    np.testing.assert_allclose(wrapper(torch.from_numpy(x), torch.from_numpy(y)).item(), ref.sum(), rtol=1e-4, atol=1e-5)
    # the wrapper leaves the leaf at the last band's a (finest) — documented behaviour, not load-bearing
    assert leaf.spread_adjustment == a_per_band[-1]


def test_uniform_a_equals_upstream_wrapper(toy_smoother_dir: Path, batch: tuple, n_grid: int) -> None:
    """a_i = 1 for all bands ⇒ identical to the upstream MultiscaleLossWrapper(CRPS)."""
    x, y = batch
    up = MultiscaleLossWrapper(
        per_scale_loss=CRPS(alpha=0.95),
        weights=[1.0, 1.0, 1.0],
        multiscale_config=_file_multiscale_config(toy_smoother_dir),
    )
    ours = MultiscaleSpreadLossWrapper(
        per_scale_loss=SpreadScaledCRPS(alpha=0.95),
        weights=[1.0, 1.0, 1.0],
        per_scale_spread_adjustment=[1.0, 1.0, 1.0],
        multiscale_config=_file_multiscale_config(toy_smoother_dir),
    )
    for w in (up, ours):
        add_uniform_node_weights(w, n_grid)
    assert torch.equal(up(torch.from_numpy(x), torch.from_numpy(y)), ours(torch.from_numpy(x), torch.from_numpy(y)))


def test_single_band_equals_leaf(batch: tuple, n_grid: int) -> None:
    x, y = batch
    leaf = SpreadScaledCRPS(alpha=0.95, spread_adjustment=1.0)
    wrapper = MultiscaleSpreadLossWrapper(per_scale_loss=leaf, weights=[1.0], per_scale_spread_adjustment=[1.3], multiscale_config=None)
    ref = SpreadScaledCRPS(alpha=0.95, spread_adjustment=1.3)
    add_uniform_node_weights(wrapper, n_grid)
    add_uniform_node_weights(ref, n_grid)
    assert wrapper.num_scales == 1
    torch.testing.assert_close(wrapper(torch.from_numpy(x), torch.from_numpy(y)), ref(torch.from_numpy(x), torch.from_numpy(y)))


def test_wrapper_validation_errors(toy_smoother_dir: Path) -> None:
    cfg = _file_multiscale_config(toy_smoother_dir)
    with pytest.raises(ValueError, match="3 bands"):
        MultiscaleSpreadLossWrapper(SpreadScaledCRPS(), weights=[1, 1, 1], per_scale_spread_adjustment=[1.4, 1.0], multiscale_config=cfg)
    with pytest.raises(TypeError, match="requires a SpreadScaledCRPS"):
        MultiscaleSpreadLossWrapper(CRPS(), weights=[1, 1, 1], per_scale_spread_adjustment=[1.4, 1.2, 1.0], multiscale_config=cfg)
    with pytest.raises(ValueError, match="per_scale_spread_adjustment\\[1\\]"):
        MultiscaleSpreadLossWrapper(SpreadScaledCRPS(), weights=[1, 1, 1], per_scale_spread_adjustment=[1.4, 0.0, 1.0], multiscale_config=cfg)


def test_gradient_flows_through_widening(toy_smoother_dir: Path, batch: tuple, n_grid: int) -> None:
    x, y = batch
    xt = torch.from_numpy(x).requires_grad_(True)
    wrapper = MultiscaleSpreadLossWrapper(
        per_scale_loss=SpreadScaledCRPS(alpha=0.95),
        weights=[1.0, 1.0, 1.0],
        per_scale_spread_adjustment=[1.41, 1.15, 1.0],
        multiscale_config=_file_multiscale_config(toy_smoother_dir),
    )
    add_uniform_node_weights(wrapper, n_grid)
    wrapper(xt, torch.from_numpy(y)).backward()
    assert xt.grad is not None and torch.isfinite(xt.grad).all() and xt.grad.abs().sum() > 0


# --------------------------------------------------------------------------- loss-factory integration
def test_factory_builds_wrapper_from_config(toy_smoother_dir: Path, batch: tuple, n_grid: int) -> None:
    """get_loss_function on our _target_ (not in NESTED_LOSSES) must still yield a working wrapper with scalers."""
    x, y = batch
    cfg = OmegaConf.create(
        {
            "_target_": "anemoi.training.spread.MultiscaleSpreadLossWrapper",
            "multiscale_config": _file_multiscale_config(toy_smoother_dir),
            "weights": [1.0, 1.0, 1.0],
            "per_scale_spread_adjustment": [1.41, 1.15, 1.0],
            "per_scale_loss": {
                "_target_": "anemoi.training.spread.SpreadScaledCRPS",
                "alpha": 0.95,
                "scalers": ["node_weights"],
                "spread_adjustment": 1.0,
            },
        },
    )
    scalers = {"node_weights": (3, torch.full((n_grid,), 1.0 / n_grid)), "unused": (4, torch.ones(3))}
    loss = get_loss_function(cfg, scalers=scalers, data_indices=None)
    assert isinstance(loss, MultiscaleSpreadLossWrapper)
    assert isinstance(loss.loss, SpreadScaledCRPS)  # no data_indices → no LossVariableMapper
    assert "node_weights" in loss.loss.scaler and "unused" not in loss.loss.scaler
    # equals the manually built wrapper
    ref = MultiscaleSpreadLossWrapper(
        per_scale_loss=SpreadScaledCRPS(alpha=0.95),
        weights=[1.0, 1.0, 1.0],
        per_scale_spread_adjustment=[1.41, 1.15, 1.0],
        multiscale_config=_file_multiscale_config(toy_smoother_dir),
    )
    add_uniform_node_weights(ref, n_grid)
    torch.testing.assert_close(loss(torch.from_numpy(x), torch.from_numpy(y)), ref(torch.from_numpy(x), torch.from_numpy(y)))


def test_factory_passes_per_variable_overrides(toy_smoother_dir: Path, n_grid: int) -> None:
    cfg = DictConfig(
        {
            "_target_": "anemoi.training.spread.MultiscaleSpreadLossWrapper",
            "multiscale_config": _file_multiscale_config(toy_smoother_dir),
            "weights": [1.0, 1.0, 1.0],
            "per_scale_spread_adjustment": [1.41, 1.15, 1.0],
            "per_scale_loss": {
                "_target_": "anemoi.training.spread.SpreadScaledCRPS",
                "alpha": 0.95,
                "scalers": ["node_weights"],
                "per_variable_spread_adjustment": {"tp": 1.0},
            },
        },
    )
    scalers = {"node_weights": (3, torch.full((n_grid,), 1.0 / n_grid))}
    loss = get_loss_function(cfg, scalers=scalers, data_indices=None)
    leaf = loss.loss
    assert leaf.per_variable_spread_adjustment == {"tp": 1.0}
    # without data_indices nobody resolved the names → forward must fail loudly, not silently ignore tp
    with pytest.raises(RuntimeError, match="never resolved"):
        loss(torch.randn(1, 1, 2, n_grid, 2), torch.randn(1, 1, 1, n_grid, 2))
    leaf.resolve_variable_spread_adjustment({"z_500": 0, "tp": 1})
    loss(torch.randn(1, 1, 2, n_grid, 2), torch.randn(1, 1, 1, n_grid, 2))


def test_smoother_order_guard_warns(toy_smoother_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    """File mode with the smoothers listed fine-first must trigger the ordering warning (soft guard)."""
    import logging

    cfg_bad = {"loss_matrices_path": str(toy_smoother_dir), "loss_matrices": ["fine.npz", "coarse.npz", None]}
    with caplog.at_level(logging.WARNING, logger="anemoi.training.spread.multiscale"):
        MultiscaleSpreadLossWrapper(SpreadScaledCRPS(), weights=[1, 1, 1], per_scale_spread_adjustment=[1.4, 1.2, 1.0], multiscale_config=cfg_bad)
    assert any("not decreasing" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="anemoi.training.spread.multiscale"):
        MultiscaleSpreadLossWrapper(SpreadScaledCRPS(), weights=[1, 1, 1], per_scale_spread_adjustment=[1.4, 1.2, 1.0], multiscale_config=_file_multiscale_config(toy_smoother_dir))
    assert not any("not decreasing" in r.message for r in caplog.records)
