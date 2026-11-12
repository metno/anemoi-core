import cv2
import numpy as np
from scipy import ndimage

from anemoi.models.layers.precipitation_motion import _local_velocity
from anemoi.models.layers.precipitation_motion import _motion_pair
from anemoi.models.layers.precipitation_motion import _regional_forecast


def test_opencv_cleanup_matches_scipy_median_filter() -> None:
    rng = np.random.default_rng(12)
    field = rng.normal(size=(67, 71)).astype(np.float32)
    field[rng.random(field.shape) < 0.1] = np.nan
    clean = np.nan_to_num(field, nan=0)

    expected = ndimage.median_filter(clean, size=5, mode="nearest")
    actual = cv2.medianBlur(np.ascontiguousarray(clean), 5)

    np.testing.assert_array_equal(actual, expected)


def test_regional_forecast_reuses_fields_without_changing_result() -> None:
    rng = np.random.default_rng(21)
    initial = rng.uniform(0, 4, size=(64, 72)).astype(np.float32)
    fields = np.stack([initial + rng.normal(0, 0.05, initial.shape) for _ in range(3)]).astype(np.float32)
    settings = {
        "pooling_factor": 4,
        "motion_log_scale_mm_h": 0.05,
        "motion_image_sigma_coarse_cells": 1.0,
        "motion_rain_minimum_mm_h": 0.05,
        "motion_tile_stride_native": 32,
        "motion_tile_size_native": 64,
        "maximum_motion_native_cells_per_step": 20,
    }

    expected = []
    previous = initial
    for field in fields:
        expected.append(_local_velocity(_motion_pair(previous, field, settings), settings))
        previous = field

    np.testing.assert_array_equal(_regional_forecast(fields, initial, settings), np.stack(expected))
