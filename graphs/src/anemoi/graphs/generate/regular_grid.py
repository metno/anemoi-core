# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Utilities for finding complete regular rectangles in projected point sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RegularGridSubset:
    """Largest complete regular rectangle found in projected coordinates."""

    node_indices: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    shape: tuple[int, int]
    spacing: tuple[float, float]


def _infer_axis_spacing(
    coordinates: np.ndarray,
    axis: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> float:
    """Infer one lattice spacing from locally axis-aligned point pairs."""
    neighbours = min(17, len(coordinates))
    distances, indices = cKDTree(coordinates).query(coordinates, k=neighbours)
    candidates: list[float] = []

    for point_index in range(len(coordinates)):
        for distance, neighbour_index in zip(distances[point_index, 1:], indices[point_index, 1:], strict=True):
            if not np.isfinite(distance):
                continue
            delta = np.abs(coordinates[neighbour_index] - coordinates[point_index])
            axis_delta = float(delta[axis])
            cross_delta = float(delta[1 - axis])
            if axis_delta > absolute_tolerance and cross_delta <= max(
                absolute_tolerance,
                relative_tolerance * axis_delta,
            ):
                candidates.append(axis_delta)

    if not candidates:
        raise ValueError(f"Could not infer projected grid spacing along axis {axis}.")

    candidates_array = np.sort(np.asarray(candidates, dtype=np.float64))
    best_cluster = candidates_array[:1]
    cluster_start = 0
    for candidate_index in range(1, len(candidates_array) + 1):
        if candidate_index < len(candidates_array):
            cluster = candidates_array[cluster_start : candidate_index + 1]
            median = float(np.median(cluster))
            if abs(float(cluster[-1]) - median) <= max(absolute_tolerance, relative_tolerance * median):
                continue
        cluster = candidates_array[cluster_start:candidate_index]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
        cluster_start = candidate_index

    spacing = float(np.median(best_cluster))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"Inferred invalid projected grid spacing {spacing} along axis {axis}.")
    return spacing


def _largest_complete_rectangle(occupancy: np.ndarray) -> tuple[int, int, int, int]:
    """Return inclusive-exclusive bounds of the largest all-true rectangle."""
    heights = np.zeros(occupancy.shape[1], dtype=np.int64)
    best: tuple[int, int, int, int] | None = None
    best_key: tuple[int, int, int, int, int] | None = None

    for bottom in range(occupancy.shape[0]):
        heights = np.where(occupancy[bottom], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for column in range(occupancy.shape[1] + 1):
            height = int(heights[column]) if column < occupancy.shape[1] else 0
            start = column
            while stack and stack[-1][1] > height:
                left, rectangle_height = stack.pop()
                start = left
                top = bottom - rectangle_height + 1
                right = column
                point_count = rectangle_height * (right - left)
                interior_area = (rectangle_height - 1) * (right - left - 1)
                key = (point_count, interior_area, -top, -left, rectangle_height)
                if rectangle_height >= 2 and right - left >= 2 and (best_key is None or key > best_key):
                    best_key = key
                    best = (top, bottom + 1, left, right)
            if not stack or stack[-1][1] < height:
                stack.append((start, height))

    if best is None:
        raise ValueError("Projected points do not contain a complete regular rectangle of at least 2 by 2 points.")
    return best


def find_largest_regular_grid(
    projected_coordinates: np.ndarray,
    *,
    x_spacing: float | None = None,
    y_spacing: float | None = None,
    relative_tolerance: float = 1.0e-4,
    absolute_tolerance: float = 1.0e-6,
    maximum_grid_cells: int = 100_000_000,
) -> RegularGridSubset:
    """Find the complete axis-aligned regular rectangle containing the most points.

    Parameters
    ----------
    projected_coordinates : np.ndarray
        Projected ``(x, y)`` point coordinates with shape ``(N, 2)``.
    x_spacing, y_spacing : float, optional
        Grid spacing in projected units. Missing values are inferred from local
        axis-aligned neighbours.
    relative_tolerance, absolute_tolerance : float
        Maximum coordinate residual from the inferred lattice.
    maximum_grid_cells : int
        Safety limit for the intermediate occupancy matrix.
    """
    coordinates = np.asarray(projected_coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"projected_coordinates must have shape (N, 2), got {coordinates.shape}.")
    if len(coordinates) < 4:
        raise ValueError("At least four projected points are required to find a 2D regular rectangle.")
    if not np.isfinite(coordinates).all():
        raise ValueError("Projected coordinates must contain only finite values.")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("Grid tolerances must be non-negative.")
    if maximum_grid_cells <= 0:
        raise ValueError("maximum_grid_cells must be positive.")

    spacing_x = (
        _infer_axis_spacing(coordinates, 0, relative_tolerance, absolute_tolerance)
        if x_spacing is None
        else float(x_spacing)
    )
    spacing_y = (
        _infer_axis_spacing(coordinates, 1, relative_tolerance, absolute_tolerance)
        if y_spacing is None
        else float(y_spacing)
    )
    if not np.isfinite([spacing_x, spacing_y]).all() or spacing_x <= 0 or spacing_y <= 0:
        raise ValueError("Projected grid spacing must be finite and positive.")

    origin = coordinates.min(axis=0)
    lattice = np.rint((coordinates - origin) / np.array([spacing_x, spacing_y])).astype(np.int64)
    reconstructed = origin + lattice * np.array([spacing_x, spacing_y])
    tolerance = np.maximum(
        absolute_tolerance,
        relative_tolerance * np.array([spacing_x, spacing_y]),
    )
    residual = np.abs(coordinates - reconstructed)
    if np.any(residual > tolerance):
        worst = np.unravel_index(np.argmax(residual / tolerance), residual.shape)
        raise ValueError(
            "Projected coordinates do not lie on one regular lattice within tolerance: "
            f"node {worst[0]}, axis {worst[1]}, residual {residual[worst]:.6g}."
        )

    lattice -= lattice.min(axis=0)
    n_columns = int(lattice[:, 0].max()) + 1
    n_rows = int(lattice[:, 1].max()) + 1
    if n_rows * n_columns > maximum_grid_cells:
        raise ValueError(
            f"Inferred occupancy grid has {n_rows * n_columns} cells, exceeding maximum_grid_cells="
            f"{maximum_grid_cells}. Check the spacing and tolerance settings."
        )

    occupancy = np.zeros((n_rows, n_columns), dtype=bool)
    source_indices = np.full((n_rows, n_columns), -1, dtype=np.int64)
    for node_index, (column, row) in enumerate(lattice):
        if occupancy[row, column]:
            raise ValueError(f"Multiple projected points occupy lattice cell (row={row}, column={column}).")
        occupancy[row, column] = True
        source_indices[row, column] = node_index

    top, bottom, left, right = _largest_complete_rectangle(occupancy)
    selected = source_indices[top:bottom, left:right].reshape(-1)
    rows, columns = np.indices((bottom - top, right - left))
    return RegularGridSubset(
        node_indices=selected,
        rows=rows.reshape(-1),
        columns=columns.reshape(-1),
        shape=(bottom - top, right - left),
        spacing=(spacing_y, spacing_x),
    )