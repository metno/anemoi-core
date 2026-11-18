# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for chained smoothing operators (a coarse-grid smoother used on a finer grid).

Toy geometry: the conftest ring of n_grid = 96 nodes is the COARSE grid ("o96"); a fine ring of 2 × 96 nodes plays n320.
R_down (coarse × fine) averages each coarse node's two fine nodes; R_up (fine × coarse) interpolates linearly between the
two nearest coarse nodes (rows sum to 1, as the real 4-NN barycentric R_up). The chained coarse band is R_up · G · R_down
with G = the toy coarse smoother; band 2 is a fine-grid moving average; band 3 the identity remainder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from .conftest import add_uniform_node_weights
from .conftest import ring_moving_average
from anemoi.training.spread import MultiscaleSpreadLossWrapper
from anemoi.training.spread import SpreadScaledCRPS
from anemoi.training.spread.multiscale import ChainedProjectionProvider
from anemoi.training.spread.target_noise import ChainedTargetNoise
from anemoi.training.spread.target_noise import chain_row_variance
from .test_spread_scaled_crps import m2_reference


def _down(nc: int) -> np.ndarray:
    d = np.zeros((nc, 2 * nc))
    for i in range(nc):
        d[i, 2 * i] = d[i, 2 * i + 1] = 0.5
    return d


def _up(nc: int) -> np.ndarray:
    """fine node j sits at coarse coordinate j/2: linear interpolation on the ring."""
    u = np.zeros((2 * nc, nc))
    for j in range(2 * nc):
        x = j / 2.0
        i0 = int(np.floor(x)) % nc
        w = x - np.floor(x)
        u[j, i0] += 1.0 - w
        u[j, (i0 + 1) % nc] += w
    return u


@pytest.fixture
def chain_dir(tmp_path: Path, toy_smoothers: dict, n_grid: int) -> Path:
    sp.save_npz(tmp_path / "up.npz", sp.csr_matrix(_up(n_grid)))
    sp.save_npz(tmp_path / "down.npz", sp.csr_matrix(_down(n_grid)))
    sp.save_npz(tmp_path / "g_coarse.npz", sp.csr_matrix(toy_smoothers["coarse"]))
    sp.save_npz(tmp_path / "fine5.npz", sp.csr_matrix(ring_moving_average(2 * n_grid, 5)))
    return tmp_path


def _cfg(d: Path) -> dict:
    return {"loss_matrices_path": str(d), "loss_matrices": [["up.npz", "g_coarse.npz", "down.npz"], "fine5.npz", None]}


def _dense(d: Path, name: str) -> np.ndarray:
    return sp.load_npz(d / name).toarray()


def test_chained_band_smoothing_equals_dense_product(chain_dir: Path, n_grid: int, rng: np.random.Generator) -> None:
    w = MultiscaleSpreadLossWrapper(SpreadScaledCRPS(alpha=0.95), weights=[1.0, 1.0, 1.0], per_scale_spread_adjustment=[1.0, 1.0, 1.0],
                                    multiscale_config=_cfg(chain_dir))
    assert w.num_scales == 3 and isinstance(w.smoothing_matrices[0], ChainedProjectionProvider)
    P = _dense(chain_dir, "up.npz") @ _dense(chain_dir, "g_coarse.npz") @ _dense(chain_dir, "down.npz")
    x = rng.standard_normal((2, 1, 2, 2 * n_grid, 3)).astype(np.float32)
    got = w._project_chain(torch.from_numpy(x), w.smoothing_matrices[0]).numpy()
    np.testing.assert_allclose(got, np.einsum("ij,...jv->...iv", P, x), rtol=1e-5, atol=1e-6)


def test_chained_band_partition_sums_to_identity_and_loss_matches_reference(chain_dir: Path, n_grid: int, rng: np.random.Generator) -> None:
    """The three bands (P f, F f − P f, f − F f) sum to f, and the wrapper loss equals the numpy per-band reference."""
    nf = 2 * n_grid
    P = _dense(chain_dir, "up.npz") @ _dense(chain_dir, "g_coarse.npz") @ _dense(chain_dir, "down.npz")
    F = _dense(chain_dir, "fine5.npz")
    x = rng.standard_normal((2, 1, 2, nf, 3)).astype(np.float32)
    y = rng.standard_normal((2, 1, 1, nf, 3)).astype(np.float32)
    bands = lambda f: [np.einsum("ij,...jv->...iv", P, f), np.einsum("ij,...jv->...iv", F - P, f), f - np.einsum("ij,...jv->...iv", F, f)]
    np.testing.assert_allclose(sum(bands(x)), x, rtol=1e-6, atol=1e-6)
    a_band, weights, alpha = [1.3, 1.1, 1.0], [1.0, 0.7, 0.5], 0.95
    w = MultiscaleSpreadLossWrapper(SpreadScaledCRPS(alpha=alpha), weights=weights, per_scale_spread_adjustment=a_band, multiscale_config=_cfg(chain_dir))
    add_uniform_node_weights(w, nf)
    out = w(torch.from_numpy(x), torch.from_numpy(y), squash=False).numpy()
    ref = np.zeros(3)
    for wt, a, bx, by in zip(weights, a_band, bands(x), bands(y), strict=True):
        ref += wt * m2_reference(bx[:, :, 0], bx[:, :, 1], by[:, :, 0], alpha, a).mean(axis=2).sum(axis=1).mean(axis=0)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_chain_shape_mismatch_raises(chain_dir: Path) -> None:
    bad = {"loss_matrices_path": str(chain_dir), "loss_matrices": [["up.npz", "down.npz", "g_coarse.npz"], "fine5.npz", None]}
    with pytest.raises(ValueError, match="compose|square"):
        MultiscaleSpreadLossWrapper(SpreadScaledCRPS(), weights=[1, 1, 1], per_scale_spread_adjustment=[1.0, 1.0, 1.0], multiscale_config=bad)


def test_chain_row_variance_is_exact(chain_dir: Path) -> None:
    U, G = _dense(chain_dir, "up.npz"), _dense(chain_dir, "g_coarse.npz")
    Gu = G / np.linalg.norm(G, axis=1, keepdims=True)
    exact = np.diag((U @ Gu) @ (U @ Gu).T)
    np.testing.assert_allclose(chain_row_variance([sp.csr_matrix(U)], sp.csr_matrix(G), chunk_rows=7), exact, rtol=1e-6)  # float32 toy matrices


def test_chained_noise_unit_variance_and_table_amplitudes(chain_dir: Path, n_grid: int) -> None:
    """Pointwise variance 1 at every fine node, and the perturbation's std per variable = its table amplitude (± 2 %)."""
    amps = torch.tensor([0.5, 0.0, 2.0])                      # 3 channels; channel 1 untouched
    tn = ChainedTargetNoise([chain_dir / "up.npz", chain_dir / "g_coarse.npz"], amps, seed=7)
    assert tn.n_grid == 2 * n_grid and tn.n_coarse == n_grid
    f = tn.coloured_unit_noise(40000, device=torch.device("cpu"), seed=11).numpy()   # (n_fine, fields)
    var = f.var(axis=1)
    assert abs(var.mean() - 1.0) < 0.01 and np.abs(var - 1.0).max() < 0.05   # MC error of a variance at n = 4e4: 0.7 %
    y = torch.zeros(1, 1, 1, 2 * n_grid, 3)
    eta = torch.stack([tn.perturb(y, global_step=s, rollout_step=0, group_id=0) for s in range(2000)])[..., :]  # (S, 1,1,1, g, 3)
    std = eta.reshape(-1, 2 * n_grid, 3).std(dim=0).mean(dim=0).numpy()
    np.testing.assert_allclose(std[[0, 2]], [0.5, 2.0], rtol=0.02)
    assert std[1] == 0.0
