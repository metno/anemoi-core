# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
from pyproj import CRS
from pyproj import Transformer
from torch_geometric.data.storage import NodeStorage

from anemoi.datasets import open_dataset
from anemoi.graphs.generate.regular_grid import find_largest_regular_grid
from anemoi.graphs.nodes.attributes.base_attributes import BaseNodeAttribute
from anemoi.graphs.nodes.attributes.base_attributes import BooleanBaseNodeAttribute

LOGGER = logging.getLogger(__name__)


class RegularGridIndices(BaseNodeAttribute):
    """Projected regular-grid row and column indices for spectral transforms.

    Selected nodes receive zero-based ``(row, column)`` indices. Nodes outside
    the largest complete rectangle receive ``(-1, -1)``.
    """

    def __init__(
        self,
        proj4_string: str,
        mask_node_attr_name: str | None = None,
        x_spacing: float | None = None,
        y_spacing: float | None = None,
        relative_tolerance: float = 1.0e-4,
        absolute_tolerance: float = 1.0e-6,
        maximum_grid_cells: int = 100_000_000,
    ) -> None:
        super().__init__(norm=None, dtype="int64")
        self.target_crs = CRS.from_user_input(proj4_string)
        self.mask_node_attr_name = mask_node_attr_name
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self.maximum_grid_cells = maximum_grid_cells

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        """Project node coordinates and encode the largest regular rectangle."""
        coordinates = nodes.x.detach().cpu().numpy()
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(f"Node coordinates must have shape (N, 2), got {coordinates.shape}.")

        num_nodes = len(coordinates)
        candidate_indices = np.arange(num_nodes)
        if self.mask_node_attr_name is not None:
            if self.mask_node_attr_name not in nodes:
                raise ValueError(f"Node storage has no mask attribute {self.mask_node_attr_name!r}.")
            source_mask = nodes[self.mask_node_attr_name].detach().cpu().numpy().astype(bool).squeeze()
            if source_mask.ndim != 1 or len(source_mask) != len(coordinates):
                raise ValueError(
                    f"Mask attribute {self.mask_node_attr_name!r} must have one value per node, "
                    f"got shape {source_mask.shape}."
                )
            candidate_indices = candidate_indices[source_mask]
            coordinates = coordinates[source_mask]

        latitudes, longitudes = np.rad2deg(coordinates).T
        transformer = Transformer.from_crs("EPSG:4326", self.target_crs, always_xy=True)
        x, y = transformer.transform(longitudes, latitudes)
        subset = find_largest_regular_grid(
            np.column_stack((x, y)),
            x_spacing=self.x_spacing,
            y_spacing=self.y_spacing,
            relative_tolerance=self.relative_tolerance,
            absolute_tolerance=self.absolute_tolerance,
            maximum_grid_cells=self.maximum_grid_cells,
        )

        indices = np.full((num_nodes, 2), -1, dtype=np.int64)
        selected_indices = candidate_indices[subset.node_indices]
        indices[selected_indices, 0] = subset.rows
        indices[selected_indices, 1] = subset.columns
        LOGGER.info(
            "Largest complete regular grid in %s has shape (y=%d, x=%d) and %d points.",
            self.target_crs.to_string(),
            subset.shape[0],
            subset.shape[1],
            len(subset.node_indices),
        )
        return torch.from_numpy(indices)


class BaseAnemoiDatasetVariable(BooleanBaseNodeAttribute):
    """Base class for computing mask based on a variable in an Anemoi dataset."""

    def __init__(self, variable: str) -> None:
        super().__init__()
        self.variable = variable

    @abstractmethod
    def _get_mask(self, ds) -> np.ndarray: ...

    def _read_data(self, nodes: NodeStorage, **kwargs) -> np.ndarray:
        return open_dataset(nodes["_dataset"], select=self.variable)[0].squeeze()

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:

        assert nodes["node_type"] in [
            "ZarrDatasetNodes",
            "AnemoiDatasetNodes",
        ], f"{self.__class__.__name__} can only be used with AnemoiDatasetNodes."
        ds = self._read_data(nodes)
        return torch.from_numpy(self._get_mask(ds))


class NonmissingAnemoiDatasetVariable(BaseAnemoiDatasetVariable):
    """Mask of valid (not missing) values of an Anemoi dataset variable.

    It reads a variable from an Anemoi dataset and returns a boolean mask of nonmissing values in the first timestep.

    Attributes
    ----------
    variable : str
        Variable to read from the Anemoi dataset.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the attribute for each node.
    """

    def __init__(self, variable: str) -> None:
        super().__init__(variable)
        self.variable = variable

    def _get_mask(self, ds) -> np.ndarray:
        return ~np.isnan(ds)


class NonzeroAnemoiDatasetVariable(BaseAnemoiDatasetVariable):
    """Mask of non-zero values of an Anemoi dataset variable.

    Reads a variable from an Anemoi dataset and returns a boolean mask of non-zero values in the first timestep.

    Attributes
    ----------
    variable : str
        Variable to read from the Anemoi dataset.

    Methods
    -------
    compute(self, graph, nodes_name)
        Computer the attribute for each node.
    """

    def __init__(self, variable: str) -> None:
        super().__init__(variable)
        self.variable = variable

    def _get_mask(self, ds) -> np.ndarray:
        return ds != 0


class BaseCombineAnemoiDatasetsMask(BooleanBaseNodeAttribute, ABC):
    """Base class for computing mask based on anemoi-datasets combining operations."""

    grids: list[int] | None = None

    def __init__(self) -> None:
        super().__init__()
        if self.grids is None:
            raise AttributeError(f"{self.__class__.__name__} class must set 'grids' attribute.")

    def get_grid_sizes(self, nodes):
        from anemoi.datasets import open_dataset

        assert "_dataset" in nodes and isinstance(
            nodes["_dataset"], (dict, str)
        ), "The '_dataset' attribute must be a dictionary or string."

        return open_dataset(nodes["_dataset"]).grids

    @staticmethod
    def get_mask_from_grid_sizes(grid_sizes: tuple[int], masked_grids_posisitons: list[int]):
        assert isinstance(masked_grids_posisitons, list), "masked_grids_positions must be a list"
        assert min(masked_grids_posisitons) >= 0, "masked_grids_positions must be non-negative"
        assert max(masked_grids_posisitons) < len(grid_sizes), f"masked_grids_positions must be < {len(grid_sizes)}"
        mask = torch.zeros(sum(grid_sizes), dtype=torch.bool)
        for grid_id in masked_grids_posisitons:
            mask[sum(grid_sizes[:grid_id]) : sum(grid_sizes[: grid_id + 1])] = True
        return mask

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        grid_sizes = self.get_grid_sizes(nodes)
        return BaseCombineAnemoiDatasetsMask.get_mask_from_grid_sizes(grid_sizes, self.grids)


class CutOutMask(BaseCombineAnemoiDatasetsMask):
    """Cut out mask.

    It computes a mask for the first dataset in the cutout operation.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the attribute for each node.
    """

    def __init__(self) -> None:
        self.grids = [0]  # It sets as true the nodes from the first (index=0) grid
        super().__init__()


class GridsMask(BaseCombineAnemoiDatasetsMask):
    """Grids mask.

    It reads a variable from a Anemoi dataset and returns a boolean mask of nonmissing values in the first timestep.

    Attributes
    ----------
    grids : int | list[int], optional
        Grid positions to set as True. Defaults to 0, which sets True only the nodes from the first dataset.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the attribute for each node.
    """

    def __init__(self, grids: int | list[int] = 0) -> None:
        self.grids = [grids] if isinstance(grids, int) else grids
        super().__init__()


class LimitedAreaMask(BooleanBaseNodeAttribute):
    """Limited area mask.

    It adds a mask based on an area of interest. This mask is only defined
    for nodes built with a subclass of `StretchedIcosahedronNodes`.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the attribute for each node.
    """

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        assert nodes["node_type"] in [
            "StretchedTriNodes"
        ], f"{self.__class__.__name__} can only be used with StretchedIcosahedronNodes."
        lam_mask = nodes["_area_mask_builder"].get_mask(nodes.x)
        return lam_mask
