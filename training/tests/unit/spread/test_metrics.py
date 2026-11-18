# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for anemoi.training.spread.metrics: EnsembleVariance, EnsembleMeanSquaredError, SSR."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from .conftest import add_uniform_node_weights
from anemoi.training.spread import EnsembleMeanSquaredError
from anemoi.training.spread import EnsembleVariance
from anemoi.training.spread import spread_skill_ratio


def test_ensemble_variance_m2_matches_formula(n_grid: int, rng: np.random.Generator) -> None:
    x = rng.standard_normal((3, 2, 2, n_grid, 4)).astype(np.float32)
    y = rng.standard_normal((3, 2, 1, n_grid, 4)).astype(np.float32)
    ref = ((x[:, :, 0] - x[:, :, 1]) ** 2 / 2.0).mean(axis=2).sum(axis=1).mean(axis=0)  # (v,)
    metric = EnsembleVariance()
    add_uniform_node_weights(metric, n_grid)
    out = metric(torch.from_numpy(x), torch.from_numpy(y), squash=False).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(metric(torch.from_numpy(x), torch.from_numpy(y)).item(), ref.mean(), rtol=1e-5)
    assert metric.name == "ensvar"


@pytest.mark.parametrize("ens_size", [2, 5])
def test_ensemble_variance_is_unbiased(ens_size: int, n_grid: int, rng: np.random.Generator) -> None:
    x = rng.standard_normal((4, 1, ens_size, n_grid, 2)).astype(np.float32)
    y = np.zeros((4, 1, 1, n_grid, 2), dtype=np.float32)
    ref = x.var(axis=2, ddof=1).mean(axis=2).sum(axis=1).mean(axis=0)
    metric = EnsembleVariance()
    add_uniform_node_weights(metric, n_grid)
    np.testing.assert_allclose(metric(torch.from_numpy(x), torch.from_numpy(y), squash=False).numpy(), ref, rtol=1e-5, atol=1e-6)


def test_ensemble_mean_squared_error(n_grid: int, rng: np.random.Generator) -> None:
    x = rng.standard_normal((3, 2, 3, n_grid, 4)).astype(np.float32)
    y = rng.standard_normal((3, 2, 1, n_grid, 4)).astype(np.float32)
    ref = ((x.mean(axis=2, keepdims=True) - y) ** 2)[:, :, 0].mean(axis=2).sum(axis=1).mean(axis=0)
    metric = EnsembleMeanSquaredError()
    add_uniform_node_weights(metric, n_grid)
    np.testing.assert_allclose(metric(torch.from_numpy(x), torch.from_numpy(y), squash=False).numpy(), ref, rtol=1e-5, atol=1e-6)
    assert metric.name == "ensmse"


def test_metrics_upcast_and_ignore_nans(n_grid: int) -> None:
    x = torch.randn(1, 1, 2, n_grid, 1, dtype=torch.bfloat16)
    y = torch.randn(1, 1, 1, n_grid, 1, dtype=torch.bfloat16)
    y[0, 0, 0, 3, 0] = float("nan")
    m = EnsembleVariance(ignore_nans=True)
    add_uniform_node_weights(m, n_grid)
    out = m(x, y)
    assert out.dtype == torch.float32 and torch.isfinite(out)


def test_ssr_reliable_ensemble_needs_correction() -> None:
    """Members and truth iid N(0,1): corrected SSR → 1, uncorrected → sqrt(M/(M+1)) (=0.816 at M=2)."""
    gen = torch.Generator().manual_seed(7)
    n = 200_000
    for ens_size, expect_uncorrected in [(2, np.sqrt(2 / 3)), (8, np.sqrt(8 / 9))]:
        x = torch.randn(1, 1, ens_size, n, 1, generator=gen)
        y = torch.randn(1, 1, 1, n, 1, generator=gen)
        ev, em = EnsembleVariance(), EnsembleMeanSquaredError()
        for m in (ev, em):
            add_uniform_node_weights(m, n)
        v, e = ev(x, y).item(), em(x, y).item()
        assert abs(spread_skill_ratio(v, e, ens_size) - 1.0) < 0.01
        assert abs(np.sqrt(v / e) - expect_uncorrected) < 0.01


def test_ssr_edge_cases() -> None:
    with pytest.raises(ValueError):
        spread_skill_ratio(1.0, 1.0, 1)
    assert np.isnan(spread_skill_ratio(1.0, 0.0, 2))
