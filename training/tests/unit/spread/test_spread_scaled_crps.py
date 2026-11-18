# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for anemoi.training.spread.SpreadScaledCRPS.

All references are self-contained (numpy formulas inline); nothing is imported from the old project.
Tensor layout everywhere: ``y_pred (bs, t, e, latlon, v)``, ``y_target (bs, t, 1, latlon, v)``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anemoi.training.losses import CRPS
from .conftest import add_uniform_node_weights
from anemoi.training.spread import SpreadScaledCRPS

torch.manual_seed(0)


def m2_reference(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, alpha: float, a: float) -> np.ndarray:
    """Point-wise spread-scaled afCRPS for M = 2:

    d = |x1−x2|/2, x̄ = (x1+x2)/2, ε = (1−α)/2,
    L = ½(|x̄ + a·d − y| + |x̄ − a·d − y|) − (1−ε)·a·|x1−x2|/2.
    """
    eps = (1.0 - alpha) / 2.0
    xbar = 0.5 * (x1 + x2)
    d = 0.5 * np.abs(x1 - x2)
    return 0.5 * (np.abs(xbar + a * d - y) + np.abs(xbar - a * d - y)) - (1.0 - eps) * a * np.abs(x1 - x2) / 2.0


def gaussian_optimal_sigma(a: float, eps: float) -> float:
    """s*(a, ε)² = 2(1−ε)²a² / [(1+a²)(1+a²−(1−ε)²a²)] ."""
    num = 2.0 * (1.0 - eps) ** 2 * a**2
    den = (1.0 + a**2) * (1.0 + a**2 - (1.0 - eps) ** 2 * a**2)
    return float(np.sqrt(num / den))


def _random_batch(bs=2, t=2, e=2, g=96, v=3, seed=1):  # noqa: ANN001, ANN202
    gen = torch.Generator().manual_seed(seed)
    y_pred = torch.randn(bs, t, e, g, v, generator=gen)
    y_true = torch.randn(bs, t, 1, g, v, generator=gen)
    return y_pred, y_true


# --------------------------------------------------------------------------- (i) a = 1 ⇒ CRPS
@pytest.mark.parametrize("backend", ["naive", "stable"])
@pytest.mark.parametrize("ens_size", [2, 4])
@pytest.mark.parametrize("alpha", [1.0, 0.95])
def test_a1_is_crps_bitwise(backend: str, ens_size: int, alpha: float, n_grid: int) -> None:
    y_pred, y_true = _random_batch(e=ens_size, g=n_grid)
    ref = CRPS(alpha=alpha, backend=backend)
    ours = SpreadScaledCRPS(alpha=alpha, backend=backend, spread_adjustment=1.0)
    for loss in (ref, ours):
        add_uniform_node_weights(loss, n_grid)
    for squash in (True, False):
        assert torch.equal(ref(y_pred, y_true, squash=squash), ours(y_pred, y_true, squash=squash))


def test_a_not_1_differs_and_name() -> None:
    y_pred, y_true = _random_batch()
    ref = CRPS(alpha=0.95)
    ours = SpreadScaledCRPS(alpha=0.95, spread_adjustment=1.414)
    assert not torch.equal(ref(y_pred, y_true), ours(y_pred, y_true))
    assert ours.name == "sscrps0.95a1.41"


# --------------------------------------------------------------------------- (ii) M = 2 numpy reference
@pytest.mark.parametrize("backend", ["naive", "stable"])
@pytest.mark.parametrize("alpha", [1.0, 0.95, 0.5])
@pytest.mark.parametrize("a", [1.0, 1.125, 1.414, 0.8])
def test_m2_matches_numpy_reference(backend: str, alpha: float, a: float, n_grid: int, rng: np.random.Generator) -> None:
    bs, t, v = 2, 2, 3
    x = rng.standard_normal((bs, t, 2, n_grid, v)).astype(np.float32)
    y = rng.standard_normal((bs, t, 1, n_grid, v)).astype(np.float32)
    ref = m2_reference(x[:, :, 0], x[:, :, 1], y[:, :, 0], alpha, a)  # (bs, t, g, v)
    # BaseLoss.reduce with uniform node weights: mean over grid, sum over t, mean over batch, then
    # squash=False keeps one value per variable, squash=True (CRPS default squash_mode="sum") sums variables.
    ref_per_var = ref.mean(axis=2).sum(axis=1).mean(axis=0)

    loss = SpreadScaledCRPS(alpha=alpha, backend=backend, spread_adjustment=a)
    add_uniform_node_weights(loss, n_grid)
    out_per_var = loss(torch.from_numpy(x), torch.from_numpy(y), squash=False).numpy()
    out_total = loss(torch.from_numpy(x), torch.from_numpy(y)).item()
    np.testing.assert_allclose(out_per_var, ref_per_var, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out_total, ref_per_var.sum(), rtol=1e-5, atol=1e-6)


def test_widen_is_affine_around_mean() -> None:
    x = torch.randn(3, 1, 4, 10, 2)
    a = 1.5
    w = SpreadScaledCRPS.widen(x, a, dim=2)
    torch.testing.assert_close(w.mean(dim=2), x.mean(dim=2))  # mean preserved
    torch.testing.assert_close(w - w.mean(dim=2, keepdim=True), a * (x - x.mean(dim=2, keepdim=True)))


# --------------------------------------------------------------------------- per-variable overrides
def test_per_variable_override_matches_reference(n_grid: int, rng: np.random.Generator) -> None:
    alpha, a_default = 0.95, 1.4
    names = ["z_500", "tp", "t_850"]
    overrides = {"tp": 1.0}
    x = rng.standard_normal((1, 1, 2, n_grid, 3)).astype(np.float32)
    y = rng.standard_normal((1, 1, 1, n_grid, 3)).astype(np.float32)
    ref = np.stack(
        [
            m2_reference(x[0, 0, 0, :, j], x[0, 0, 1, :, j], y[0, 0, 0, :, j], alpha, overrides.get(n, a_default)).mean()
            for j, n in enumerate(names)
        ],
    )
    loss = SpreadScaledCRPS(alpha=alpha, spread_adjustment=a_default, per_variable_spread_adjustment=overrides)
    add_uniform_node_weights(loss, n_grid)
    with pytest.raises(RuntimeError, match="never resolved"):
        loss(torch.from_numpy(x), torch.from_numpy(y))
    loss.resolve_variable_spread_adjustment({n: i for i, n in enumerate(names)})
    out = loss(torch.from_numpy(x), torch.from_numpy(y), squash=False).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    # changing the default (what the multiscale wrapper does per band) must invalidate the cache
    loss.spread_adjustment = 1.0
    ref1 = np.stack(
        [
            m2_reference(x[0, 0, 0, :, j], x[0, 0, 1, :, j], y[0, 0, 0, :, j], alpha, overrides.get(n, 1.0)).mean()
            for j, n in enumerate(names)
        ],
    )
    out1 = loss(torch.from_numpy(x), torch.from_numpy(y), squash=False).numpy()
    np.testing.assert_allclose(out1, ref1, rtol=1e-5, atol=1e-6)


def test_per_variable_override_unknown_name_raises() -> None:
    loss = SpreadScaledCRPS(spread_adjustment=1.2, per_variable_spread_adjustment={"nope": 1.0})
    with pytest.raises(KeyError, match="nope"):
        loss.resolve_variable_spread_adjustment({"z_500": 0})


def test_per_variable_override_wrong_num_vars_raises(n_grid: int) -> None:
    loss = SpreadScaledCRPS(spread_adjustment=1.2, per_variable_spread_adjustment={"tp": 1.0})
    loss.resolve_variable_spread_adjustment({"z_500": 0, "tp": 1})
    y_pred, y_true = _random_batch(g=n_grid, v=3)
    with pytest.raises(RuntimeError, match="received 3"):
        loss(y_pred, y_true)


@pytest.mark.parametrize("bad", [0.0, -1.0, True, "1.2"])
def test_invalid_spread_adjustment_rejected(bad) -> None:  # noqa: ANN001
    with pytest.raises((ValueError, TypeError)):
        SpreadScaledCRPS(spread_adjustment=bad)


# --------------------------------------------------------------------------- (iii) Gaussian optimum
@pytest.mark.parametrize(
    ("a", "alpha", "expected"),
    [
        (1.0, 1.0, 1.0),  # fair CRPS: unbiased even at M = 2
        (1.0, 0.95, gaussian_optimal_sigma(1.0, 0.025)),  # 0.9518
        (1.414, 0.95, gaussian_optimal_sigma(1.414, 0.025)),  # 1.0740
        (1.125, 0.95, gaussian_optimal_sigma(1.125, 0.025)),  # 0.9998
    ],
)
def test_gaussian_optimum_matches_formula(a: float, alpha: float, expected: float) -> None:
    """Members ~ N(0, s²), truth ~ N(0, 1): the empirical argmin over s must match s*(a, ε), ε=(1−α)/2.

    Common random numbers (one standard-normal draw rescaled by s) make the empirical curve smooth
    in s, so a fine grid + parabolic refinement pins the optimum to ~1e-3.
    """
    n = 400_000
    gen = torch.Generator().manual_seed(1234)
    z = torch.randn(1, 1, 2, n, 1, generator=gen)  # member noise
    y = torch.randn(1, 1, 1, n, 1, generator=gen)  # truth
    loss = SpreadScaledCRPS(alpha=alpha, spread_adjustment=a, backend="naive")
    add_uniform_node_weights(loss, n)
    s_grid = np.round(np.arange(0.85, 1.20, 0.005), 4)
    with torch.no_grad():
        vals = np.array([loss(float(s) * z, y).item() for s in s_grid])
    i = int(vals.argmin())
    assert 0 < i < len(s_grid) - 1, "argmin at grid boundary; widen the grid"
    # parabolic refinement through the three points around the minimum
    (s0, s1, s2), (v0, v1, v2) = s_grid[i - 1 : i + 2], vals[i - 1 : i + 2]
    denom = (s0 - s1) * (s0 - s2) * (s1 - s2)
    A = (s2 * (v1 - v0) + s1 * (v0 - v2) + s0 * (v2 - v1)) / denom
    B = (s2**2 * (v0 - v1) + s1**2 * (v2 - v0) + s0**2 * (v1 - v2)) / denom
    s_star = -B / (2 * A)
    assert abs(s_star - expected) < 0.01, f"MC optimum {s_star:.4f} vs formula {expected:.4f}"


def test_unknown_kwargs_rejected() -> None:
    """CRPS.__init__ swallows unknown kwargs; a typo like `spread_adjustmen` must not silently give a=1."""
    with pytest.raises(TypeError, match="spread_adjustmen"):
        SpreadScaledCRPS(alpha=0.95, spread_adjustmen=1.4)
    SpreadScaledCRPS(alpha=0.95, backend="naive", no_autocast=True, ignore_nans=False, spread_adjustment=1.2)
