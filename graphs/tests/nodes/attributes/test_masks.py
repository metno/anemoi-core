# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from pyproj import Transformer
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.attributes import CutOutMask
from anemoi.graphs.nodes.attributes import GridsMask
from anemoi.graphs.nodes.attributes import RegularGridIndices
from anemoi.graphs.nodes.attributes.masks import BaseCombineAnemoiDatasetsMask


def test_cutout_mask(mocker, graph_with_nodes: HeteroData, mock_anemoi_dataset_cutout):
    """Test attribute builder for CutOutMask."""
    # Add dataset attribute required by CutOutMask
    graph_with_nodes["test_nodes"]["_dataset"] = {}

    mocker.patch("anemoi.datasets.open_dataset", return_value=mock_anemoi_dataset_cutout)
    mask = CutOutMask().compute(graph_with_nodes, "test_nodes")

    assert mask is not None
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]


def test_get_mask_from_grid_size():
    grid_sizes1 = (1, 2, 3, 2, 1)
    grid_sizes2 = (4,)
    grid_ids1 = [0, 1, 4]
    grid_ids2 = [1]

    grids_mask1 = BaseCombineAnemoiDatasetsMask.get_mask_from_grid_sizes(grid_sizes1, grid_ids1)
    grids_mask2 = BaseCombineAnemoiDatasetsMask.get_mask_from_grid_sizes(grid_sizes1, grid_ids2)

    with pytest.raises(AssertionError):
        BaseCombineAnemoiDatasetsMask.get_mask_from_grid_sizes(grid_sizes2, grid_ids2)

    assert all(grids_mask1 == torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 1], dtype=torch.bool))
    assert all(grids_mask2 == torch.tensor([0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool))


def test_regular_grid_indices_projects_and_preserves_source_order() -> None:
    target_crs = "EPSG:3857"
    inverse = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    y, x = np.indices((3, 4))
    projected = np.column_stack((x.reshape(-1) * 2_000.0, y.reshape(-1) * 3_000.0))
    longitudes, latitudes = inverse.transform(projected[:, 0], projected[:, 1])
    order = np.random.default_rng(12).permutation(len(projected))
    graph = HeteroData()
    regular_coordinates = torch.tensor(
        np.column_stack((latitudes[order], longitudes[order])),
        dtype=torch.float64,
    )
    graph["data"].x = torch.deg2rad(torch.cat((regular_coordinates, torch.tensor([[0.0, 120.0]]))))
    graph["data"].cutout_mask = torch.tensor([True] * 12 + [False])

    indices = RegularGridIndices(
        proj4_string=target_crs,
        mask_node_attr_name="cutout_mask",
        x_spacing=2_000.0,
        y_spacing=3_000.0,
        absolute_tolerance=1.0e-3,
    ).compute(graph, "data")

    assert indices.dtype == torch.int64
    assert indices.shape == (13, 2)
    assert torch.equal(indices[-1], torch.tensor([-1, -1]))
    linear_indices = indices[:-1, 0] * 4 + indices[:-1, 1]
    np.testing.assert_array_equal(order[torch.argsort(linear_indices).numpy()], np.arange(12))


@pytest.mark.parametrize("mask_class", [CutOutMask, GridsMask])
def test_combined_datasets_mask_missing_dataset(graph_with_nodes: HeteroData, mask_class):
    """Test CutOutMask fails when dataset attribute is missing."""
    node_attr_builder = mask_class()
    with pytest.raises(AssertionError):
        node_attr_builder.compute(graph_with_nodes, "test_nodes")


if __name__ == "__main__":
    pytest.main([__file__])
