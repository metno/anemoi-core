# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared fixtures for the anemoi.training.spread tests (CPU only, seconds of runtime).

Toy geometry: a 1-D periodic ring of ``N`` grid points.  Smoothers are row-normalised
moving averages of odd width ``w`` (so ``D(const) = const``), stored as scipy sparse ``.npz``
files exactly the way ``MultiscaleLossWrapper``'s file mode expects them
(``ProjectionGraphProvider(file_path=...)`` → ``scipy.sparse.load_npz``, shape ``(out, in)``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

def ring_moving_average(n: int, width: int) -> np.ndarray:
    """Dense ``(n, n)`` circular moving-average matrix of odd ``width`` (rows sum to 1)."""
    assert width % 2 == 1
    half = width // 2
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for k in range(-half, half + 1):
            mat[i, (i + k) % n] = 1.0 / width
    return mat


@pytest.fixture(scope="session")
def n_grid() -> int:
    return 96


@pytest.fixture(scope="session")
def toy_smoothers(n_grid: int) -> dict[str, np.ndarray]:
    """Coarsest first, as the wrapper expects."""
    return {"coarse": ring_moving_average(n_grid, 15), "fine": ring_moving_average(n_grid, 5)}


@pytest.fixture(scope="session")
def toy_smoother_dir(tmp_path_factory: pytest.TempPathFactory, toy_smoothers: dict[str, np.ndarray]) -> Path:
    d = tmp_path_factory.mktemp("smoothers")
    for name, mat in toy_smoothers.items():
        sp.save_npz(d / f"{name}.npz", sp.csr_matrix(mat))
    return d


@pytest.fixture
def uniform_node_weights(n_grid: int) -> torch.Tensor:
    return torch.full((n_grid,), 1.0 / n_grid, dtype=torch.float32)


def add_uniform_node_weights(loss, n_grid: int) -> None:  # noqa: ANN001
    """Attach uniform area weights so that BaseLoss.reduce yields grid *means*."""
    loss.add_scaler(3, torch.full((n_grid,), 1.0 / n_grid, dtype=torch.float32), name="node_weights")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260817)


# --------------------------------------------------------------------------- real anemoi data indices
#: toy variable set: prognostic (z_500, t_850, u_850, tcw, q_500, tcc), diagnostic (tp, ssrd), forcing (lsm)
TOY_VARIABLES = ["z_500", "tp", "lsm", "t_850", "tcw", "ssrd", "q_500", "u_850", "tcc"]
TOY_FORCING = ["lsm"]
TOY_DIAGNOSTIC = ["tp", "ssrd"]


def make_index_collection(names: list[str] | None = None, forcing: list[str] | None = None, diagnostic: list[str] | None = None):  # noqa: ANN201
    """A real ``anemoi.models.data_indices.collection.IndexCollection`` for the toy variable set."""
    from omegaconf import DictConfig

    from anemoi.models.data_indices.collection import IndexCollection

    names = list(TOY_VARIABLES if names is None else names)
    cfg = DictConfig({"forcing": list(TOY_FORCING if forcing is None else forcing), "diagnostic": list(TOY_DIAGNOSTIC if diagnostic is None else diagnostic)})
    return IndexCollection(cfg, {n: i for i, n in enumerate(names)})


@pytest.fixture
def toy_data_indices():  # noqa: ANN201
    return make_index_collection()
