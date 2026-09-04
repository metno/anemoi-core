# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from anemoi.graphs.generate.regular_grid import find_largest_regular_grid


def _shuffled_grid(rows: int, columns: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices((rows, columns))
    coordinates = np.column_stack((2.0 * x.reshape(-1), 3.0 * y.reshape(-1)))
    order = np.random.default_rng(seed).permutation(len(coordinates))
    return coordinates[order], order


def test_finds_shuffled_complete_grid_and_row_major_indices() -> None:
    coordinates, order = _shuffled_grid(3, 4)

    result = find_largest_regular_grid(coordinates)

    assert result.shape == (3, 4)
    assert result.spacing == pytest.approx((3.0, 2.0))
    np.testing.assert_array_equal(order[result.node_indices], np.arange(12))
    np.testing.assert_array_equal(result.rows, np.repeat(np.arange(3), 4))
    np.testing.assert_array_equal(result.columns, np.tile(np.arange(4), 3))


def test_finds_largest_complete_rectangle_around_holes() -> None:
    coordinates, _ = _shuffled_grid(4, 5)
    coordinates = coordinates[~np.all(coordinates == np.array([4.0, 3.0]), axis=1)]

    result = find_largest_regular_grid(coordinates, x_spacing=2.0, y_spacing=3.0)

    assert result.shape == (4, 2)
    selected_coordinates = coordinates[result.node_indices]
    assert set(selected_coordinates[:, 0]) in ({0.0, 2.0}, {6.0, 8.0})


def test_accepts_small_coordinate_jitter() -> None:
    coordinates, _ = _shuffled_grid(3, 3)
    coordinates += np.random.default_rng(4).normal(scale=1.0e-6, size=coordinates.shape)

    result = find_largest_regular_grid(
        coordinates,
        x_spacing=2.0,
        y_spacing=3.0,
        absolute_tolerance=1.0e-5,
    )

    assert result.shape == (3, 3)


@pytest.mark.parametrize(
    "coordinates, match",
    [
        (np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), "Multiple projected points"),
        (np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]), "axis 1"),
        (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [np.nan, 1.0]]), "finite"),
    ],
)
def test_rejects_invalid_grids(coordinates: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        find_largest_regular_grid(coordinates)