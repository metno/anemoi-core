# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests of the controller-facing features of the multiscale spread wrapper:

- per-band spread/skill diagnostics recorded by the multiscale wrapper,
- per-(variable, band) adjustment table composing multiplicatively with the band scalar,
- per-rollout-step factor applied via the forwarded ``rollout_step``.

Toy geometry from conftest (1-D periodic ring, two moving-average smoothers, file mode);
CPU only, seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.training.losses.loss import get_loss_function
from .conftest import add_uniform_node_weights
from anemoi.training.spread import MultiscaleSpreadLossWrapper
from anemoi.training.spread import SpreadScaledCRPS
from anemoi.training.spread import TrainingDiagnosticsLossWrapper
from .test_multiscale_spread_wrapper import _bands_numpy
from .test_multiscale_spread_wrapper import _file_multiscale_config
from .test_spread_scaled_crps import m2_reference

NAMES = {"v0": 0, "v1": 1, "v2": 2}


@pytest.fixture
def batch(n_grid: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.standard_normal((2, 1, 2, n_grid, 3)).astype(np.float32)
    y = rng.standard_normal((2, 1, 1, n_grid, 3)).astype(np.float32)
    return x, y


def _reference_loss(x, y, smoothers, weights, alpha, a_eff):  # noqa: ANN001, ANN201
    """Σ_i w_i · afCRPS(band_i) with per-(variable, band) widening ``a_eff[v][i]``, in numpy."""
    ref = np.zeros(x.shape[-1])
    for i, (w, bx, by) in enumerate(zip(weights, _bands_numpy(x, smoothers), _bands_numpy(y, smoothers), strict=True)):
        for v in range(x.shape[-1]):
            pt = m2_reference(
                bx[:, :, 0, :, v : v + 1],
                bx[:, :, 1, :, v : v + 1],
                by[:, :, 0, :, v : v + 1],
                alpha,
                a_eff[v][i],
            )
            ref[v] += w * pt.mean(axis=2).sum(axis=1).mean(axis=0)[0]
    return ref


def _make_wrapper(toy_smoother_dir: Path, n_grid: int, **kwargs):  # noqa: ANN003, ANN201
    leaf = SpreadScaledCRPS(alpha=0.95, spread_adjustment=1.0, per_variable_spread_adjustment=kwargs.pop("pins", None))
    leaf.resolve_variable_spread_adjustment(NAMES)
    wrapper = MultiscaleSpreadLossWrapper(
        per_scale_loss=leaf,
        weights=kwargs.pop("weights", [1.0, 0.7, 0.5]),
        per_scale_spread_adjustment=kwargs.pop("per_scale_spread_adjustment", [1.4, 1.15, 1.0]),
        multiscale_config=_file_multiscale_config(toy_smoother_dir),
        **kwargs,
    )
    add_uniform_node_weights(wrapper, n_grid)
    return wrapper


# --------------------------------------------------------------------------- per-(variable, band) table
def test_vb_table_matches_numpy_reference(toy_smoother_dir: Path, toy_smoothers: dict, batch: tuple, n_grid: int) -> None:
    """Effective widening: table variable a_i·m_i, plain variable a_i, pinned variable 1.0 — every band."""
    x, y = batch
    weights, a_band = [1.0, 0.7, 0.5], [1.4, 1.15, 1.0]
    table = {"v0": [1.1, 0.95, 1.05]}
    wrapper = _make_wrapper(
        toy_smoother_dir,
        n_grid,
        weights=weights,
        per_scale_spread_adjustment=a_band,
        per_variable_band_spread_adjustment=table,
        pins={"v2": 1.0},
    )
    out = wrapper(torch.from_numpy(x), torch.from_numpy(y), squash=False).numpy()
    a_eff = {
        0: [a * m for a, m in zip(a_band, table["v0"], strict=True)],  # table variable
        1: list(a_band),  # plain variable: band scalar
        2: [1.0, 1.0, 1.0],  # pinned: static override wins in every band
    }
    ref = _reference_loss(x, y, toy_smoothers, weights, 0.95, a_eff)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_vb_table_of_ones_matches_legacy(toy_smoother_dir: Path, batch: tuple, n_grid: int) -> None:
    x, y = batch
    legacy = _make_wrapper(toy_smoother_dir, n_grid)
    ours = _make_wrapper(toy_smoother_dir, n_grid, per_variable_band_spread_adjustment={"v0": [1.0, 1.0, 1.0]})
    np.testing.assert_allclose(
        ours(torch.from_numpy(x), torch.from_numpy(y), squash=False).numpy(),
        legacy(torch.from_numpy(x), torch.from_numpy(y), squash=False).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )


def test_legacy_path_leaves_leaf_overrides_untouched(toy_smoother_dir: Path, batch: tuple, n_grid: int) -> None:
    """Without a table / rollout factors the wrapper must behave exactly as the plain per-band path."""
    x, y = batch
    wrapper = _make_wrapper(toy_smoother_dir, n_grid, pins={"v2": 1.0})
    leaf = wrapper._spread_leaves[0]
    before = leaf.per_variable_spread_adjustment
    wrapper(torch.from_numpy(x), torch.from_numpy(y))
    assert leaf.per_variable_spread_adjustment == before == {"v2": 1.0}


def test_vb_table_validation(toy_smoother_dir: Path, n_grid: int) -> None:
    with pytest.raises(ValueError, match="expected 3 bands"):
        _make_wrapper(toy_smoother_dir, n_grid, per_variable_band_spread_adjustment={"v0": [1.0, 1.0]})
    with pytest.raises(ValueError, match="must be > 0"):
        _make_wrapper(toy_smoother_dir, n_grid, per_variable_band_spread_adjustment={"v0": [1.0, -0.5, 1.0]})
    wrapper = _make_wrapper(toy_smoother_dir, n_grid)
    wrapper.set_per_variable_band_table({"v0": [1.05, 1.0, 0.95]})
    assert wrapper.get_per_variable_band_table() == {"v0": [1.05, 1.0, 0.95]}


# --------------------------------------------------------------------------- per-rollout-step factor
def test_rollout_factor_applied(toy_smoother_dir: Path, batch: tuple, n_grid: int) -> None:
    """a_r composes multiplicatively with the band scalar; step beyond the list clamps to the last."""
    x, y = batch
    a_band = [1.4, 1.15, 1.0]
    with_ar = _make_wrapper(
        toy_smoother_dir,
        n_grid,
        per_scale_spread_adjustment=a_band,
        per_rollout_step_spread_adjustment=[1.0, 1.25],
    )
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)

    # step 0: a_r = 1 -> identical to the legacy wrapper
    legacy = _make_wrapper(toy_smoother_dir, n_grid, per_scale_spread_adjustment=a_band)
    np.testing.assert_allclose(
        with_ar(xt, yt, squash=False, rollout_step=0).numpy(),
        legacy(xt, yt, squash=False).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    # step 1: a_r = 1.25 -> equals a wrapper with the scaled per-band a
    scaled = _make_wrapper(toy_smoother_dir, n_grid, per_scale_spread_adjustment=[a * 1.25 for a in a_band])
    np.testing.assert_allclose(
        with_ar(xt, yt, squash=False, rollout_step=1).numpy(),
        scaled(xt, yt, squash=False).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    # step beyond the configured factors clamps to the last entry
    np.testing.assert_allclose(
        with_ar(xt, yt, squash=False, rollout_step=7).numpy(),
        scaled(xt, yt, squash=False).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )


def test_rollout_factor_requires_step(toy_smoother_dir: Path, batch: tuple, n_grid: int) -> None:
    x, y = batch
    wrapper = _make_wrapper(toy_smoother_dir, n_grid, per_rollout_step_spread_adjustment=[1.0, 1.25])
    with pytest.raises(RuntimeError, match="rollout_step"):
        wrapper(torch.from_numpy(x), torch.from_numpy(y))


# --------------------------------------------------------------------------- per-band diagnostics
def test_band_diagnostics_recorded(toy_smoother_dir: Path, toy_smoothers: dict, batch: tuple, n_grid: int) -> None:
    """The diagnostics wrapper's pending entries carry per-band ensvar/ensmse matching the
    metrics evaluated directly on the numpy telescoping bands, plus the recorded rollout step."""
    x, y = batch
    scalers = {"node_weights": (3, torch.full((n_grid,), 1.0 / n_grid))}
    inner = {
        "_target_": "anemoi.training.spread.MultiscaleSpreadLossWrapper",
        "weights": [1.0, 1.0, 1.0],
        "per_scale_spread_adjustment": [1.0, 1.0, 1.0],
        "multiscale_config": _file_multiscale_config(toy_smoother_dir),
        "per_scale_loss": {"_target_": "anemoi.training.spread.SpreadScaledCRPS", "alpha": 0.95, "scalers": ["node_weights"]},
    }
    cfg = DictConfig({"_target_": "anemoi.training.spread.TrainingDiagnosticsLossWrapper", "loss": inner})
    wrapped = get_loss_function(cfg, scalers=scalers, data_indices=None)
    assert isinstance(wrapped, TrainingDiagnosticsLossWrapper)
    assert wrapped.loss._band_diag_metrics is not None

    wrapped(torch.from_numpy(x), torch.from_numpy(y), rollout_step=0)
    assert len(wrapped.pending) == 1
    values = wrapped.pending[0]
    assert values["_rollout_step"] == 0
    for name in ("ensvar", "ensmse"):
        for i in range(3):
            assert f"{name}_band{i}" in values, sorted(values)

    bx = _bands_numpy(x, toy_smoothers)
    by = _bands_numpy(y, toy_smoothers)
    for i in range(3):
        for name in ("ensvar", "ensmse"):
            direct = wrapped.diagnostics[name](
                torch.from_numpy(bx[i]), torch.from_numpy(by[i]), squash=False
            ).float()
            np.testing.assert_allclose(values[f"{name}_band{i}"].numpy(), direct.numpy(), rtol=1e-5, atol=1e-6)

    # whole-field diagnostics are still recorded and unchanged by the band machinery
    direct_full = wrapped.diagnostics["ensvar"](torch.from_numpy(x), torch.from_numpy(y), squash=False).float()
    np.testing.assert_allclose(values["ensvar"].numpy(), direct_full.numpy(), rtol=1e-6, atol=1e-7)


def test_band_diagnostics_cleared_between_forwards(toy_smoother_dir: Path, batch: tuple, n_grid: int) -> None:
    x, y = batch
    scalers = {"node_weights": (3, torch.full((n_grid,), 1.0 / n_grid))}
    inner = {
        "_target_": "anemoi.training.spread.MultiscaleSpreadLossWrapper",
        "weights": [1.0, 1.0, 1.0],
        "per_scale_spread_adjustment": [1.0, 1.0, 1.0],
        "multiscale_config": _file_multiscale_config(toy_smoother_dir),
        "per_scale_loss": {"_target_": "anemoi.training.spread.SpreadScaledCRPS", "alpha": 0.95, "scalers": ["node_weights"]},
    }
    cfg = DictConfig({"_target_": "anemoi.training.spread.TrainingDiagnosticsLossWrapper", "loss": inner})
    wrapped = get_loss_function(cfg, scalers=scalers, data_indices=None)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    wrapped(xt, yt, rollout_step=0)
    wrapped(xt, yt, rollout_step=1)
    assert [v["_rollout_step"] for v in wrapped.pending] == [0, 1]
    assert wrapped.loss._band_diag_values == {}  # consumed by the outer wrapper each call


def test_per_band_rollout_factors(toy_smoother_dir: Path, batch: tuple, n_grid: int) -> None:
    """Nested a_{b,r} rows apply per band: at a given step, each band uses its own factor —
    equivalent to scaling the per-band scalars by that step's column."""
    x, y = batch
    a_band = [1.4, 1.15, 1.0]
    nested = _make_wrapper(
        toy_smoother_dir,
        n_grid,
        per_scale_spread_adjustment=a_band,
        per_rollout_step_spread_adjustment=[[1.0, 1.3], [1.0, 1.2], [1.0, 1.1]],
    )
    assert nested.get_per_rollout_step_adjustment() == [[1.0, 1.3], [1.0, 1.2], [1.0, 1.1]]
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    # step 0: all bands at 1 -> legacy
    legacy = _make_wrapper(toy_smoother_dir, n_grid, per_scale_spread_adjustment=a_band)
    np.testing.assert_allclose(
        nested(xt, yt, squash=False, rollout_step=0).numpy(),
        legacy(xt, yt, squash=False).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    # step 1: band-specific factors == scaling the band scalars by the step-1 column
    scaled = _make_wrapper(
        toy_smoother_dir,
        n_grid,
        per_scale_spread_adjustment=[a * f for a, f in zip(a_band, [1.3, 1.2, 1.1], strict=True)],
    )
    np.testing.assert_allclose(
        nested(xt, yt, squash=False, rollout_step=1).numpy(),
        scaled(xt, yt, squash=False).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    # flat input broadcasts to identical rows
    flat = _make_wrapper(toy_smoother_dir, n_grid, per_rollout_step_spread_adjustment=[1.0, 1.25])
    assert flat.get_per_rollout_step_adjustment() == [[1.0, 1.25]] * 3
