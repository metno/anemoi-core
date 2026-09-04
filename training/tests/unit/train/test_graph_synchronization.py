# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import pytest
from pytest_mock import MockerFixture
from torch_geometric.data import HeteroData

from anemoi.training.train.train import _distributed_global_rank
from anemoi.training.train.train import _wait_for_graph


def test_distributed_global_rank_uses_torchrun_then_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_PROCID", "3")
    assert _distributed_global_rank() == 3

    monkeypatch.setenv("RANK", "7")
    assert _distributed_global_rank() == 7


def test_wait_for_graph_loads_graph_after_atomic_publication(tmp_path: Path, mocker: MockerFixture) -> None:
    graph_path = tmp_path / "graph.pt"
    graph = HeteroData()
    exists = mocker.patch.object(Path, "exists", side_effect=[False, True])
    load = mocker.patch("anemoi.training.train.train.load_graph_from_file", return_value=graph)
    mocker.patch("anemoi.training.train.train.time.monotonic", side_effect=[0.0, 0.0, 1.0])
    sleep = mocker.patch("anemoi.training.train.train.time.sleep")

    result = _wait_for_graph(graph_path, timeout=5.0)

    assert result is graph
    assert exists.call_count == 2
    sleep.assert_called_once()
    load.assert_called_once_with(graph_path)


def test_wait_for_graph_times_out(tmp_path: Path, mocker: MockerFixture) -> None:
    graph_path = tmp_path / "graph.pt"
    mocker.patch.object(Path, "exists", return_value=False)
    mocker.patch("anemoi.training.train.train.time.monotonic", side_effect=[0.0, 0.0, 5.0])
    mocker.patch("anemoi.training.train.train.time.sleep")

    with pytest.raises(TimeoutError, match="waiting for rank zero"):
        _wait_for_graph(graph_path, timeout=5.0)