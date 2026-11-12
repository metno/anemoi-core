# (C) Copyright 2026 Anemoi contributors.
# This software is licensed under the terms of the Apache Licence Version 2.0.

"""Correct complete native precipitation rollouts with rigid radar drift and graph-seam cleanup."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np
import torch
from scipy import ndimage
from torch import Tensor
from torch import nn


def _motion_field(field: np.ndarray, cfg: dict) -> dict:
    factor = int(cfg.get("pooling_factor", 4))
    height, width = field.shape
    padding = ((0, (-height) % factor), (0, (-width) % factor))
    valid = np.pad(np.isfinite(field), padding, mode="edge")
    rain = np.pad(np.maximum(np.nan_to_num(field, nan=0), 0), padding, mode="edge")
    shape = (rain.shape[0] // factor, factor, rain.shape[1] // factor, factor)
    scale = float(cfg.get("motion_log_scale_mm_h", .05))
    image = np.log1p(rain / scale).reshape(shape).mean(axis=(1, 3))
    image *= 255 / np.log1p(5 / scale)
    image = ndimage.gaussian_filter(image.clip(0, 255), float(cfg.get("motion_image_sigma_coarse_cells", 1)))
    return {
        "image": image.astype(np.float32),
        "rain": rain.reshape(shape).mean(axis=(1, 3)),
        "valid": valid.reshape(shape).mean(axis=(1, 3)) >= .999,
        "pooling_factor": factor,
        "native_shape": (height, width),
    }


def _motion_pair_from_fields(previous: dict, current: dict) -> dict:
    factor = previous["pooling_factor"]
    images = [previous["image"], current["image"]]
    forward = cv2.calcOpticalFlowFarneback(images[0], images[1], None, .5, 4, 21, 5, 5, 1.2, 0)
    gradient = np.maximum(np.hypot(ndimage.sobel(images[0], 0) / 8, ndimage.sobel(images[0], 1) / 8),
                          np.hypot(ndimage.sobel(images[1], 0) / 8, ndimage.sobel(images[1], 1) / 8))
    return {"images": images, "rain": [previous["rain"], current["rain"]],
        "valid": previous["valid"] & current["valid"],
        "forward": forward, "gradient": gradient, "pooling_factor": factor,
        "native_shape": previous["native_shape"]}


def _motion_pair(previous: np.ndarray, current: np.ndarray, cfg: dict) -> dict:
    return _motion_pair_from_fields(_motion_field(previous, cfg), _motion_field(current, cfg))


def _local_velocity(pair: dict, cfg: dict) -> np.ndarray:
    flow = pair["forward"]
    height, width = flow.shape[:2]
    factor = pair["pooling_factor"]
    usable = (pair["valid"] & (np.minimum(*pair["rain"]) >= cfg["motion_rain_minimum_mm_h"])
              & (pair["gradient"] >= 1))
    fallback = np.median(flow[usable], axis=0) if usable.any() else np.zeros(2, dtype=np.float32)
    step = cfg["motion_tile_stride_native"] // factor
    radius = cfg["motion_tile_size_native"] // factor // 2
    rows, columns = np.arange(0, height + step, step), np.arange(0, width + step, step)
    knots = np.empty((len(rows), len(columns), 2), dtype=np.float32)
    for yi, y in enumerate(rows):
        for xi, x in enumerate(columns):
            selection = (slice(max(0, y - radius), min(height, y + radius)),
                         slice(max(0, x - radius), min(width, x + radius)))
            selected = usable[selection]
            knots[yi, xi] = np.median(flow[selection][selected], axis=0) if selected.sum() >= 8 else fallback
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    result = cv2.remap(knots, xx / step, yy / step, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    result = ndimage.gaussian_filter(result, sigma=(1, 1, 0))
    maximum = cfg["maximum_motion_native_cells_per_step"] / factor
    result *= np.minimum(1, maximum / np.maximum(np.linalg.norm(result, axis=-1), 1e-6))[..., None]
    return result


def _fit_rigid_velocity(velocity: np.ndarray, mask: np.ndarray, factor: int) -> tuple:
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]].astype(np.float64)
    if not mask.any():
        mask = np.ones(mask.shape, dtype=bool)
    centre = np.asarray((xx[mask].mean() * factor, yy[mask].mean() * factor))
    x, y = xx[mask] * factor - centre[0], yy[mask] * factor - centre[1]
    design = np.zeros((2 * len(x), 3), dtype=np.float64)
    design[:len(x), 0], design[len(x):, 1] = 1, 1
    design[:len(x), 2], design[len(x):, 2] = -y, x
    vectors = velocity[mask].astype(np.float64) * factor
    target = np.r_[vectors[:, 0], vectors[:, 1]]
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return coefficients, centre


def _inverse_rigid_map(x: np.ndarray, y: np.ndarray, coefficients: np.ndarray,
                      centre: np.ndarray, steps: int) -> tuple:
    u, v, omega = coefficients
    angle = omega * steps
    cosine, sine = np.cos(angle), np.sin(angle)
    along = steps * np.sinc(angle / np.pi)
    across = steps * .5 * angle * np.sinc(angle / (2 * np.pi)) ** 2
    tx, ty = along * u - across * v, across * u + along * v
    dx, dy = x - centre[0] - tx, y - centre[1] - ty
    return ((cosine * dx + sine * dy + centre[0]).astype(np.float32),
            (-sine * dx + cosine * dy + centre[1]).astype(np.float32))


def _regional_forecast(fields: np.ndarray, initial: np.ndarray, cfg: dict) -> np.ndarray:
    prepared = [_motion_field(initial, cfg)]
    prepared.extend(_motion_field(field, cfg) for field in fields)
    return np.stack([
        _local_velocity(_motion_pair_from_fields(previous, current), cfg)
        for previous, current in zip(prepared[:-1], prepared[1:])
    ])


class PrecipitationMotionHead(nn.Module):
    """Apply seam cleanup and a common rigid drift after the raw neural rollout.

    Predictions and history contain physical rain rates on the full native grid.
    The same member-zero drift correction is used for every ensemble member;
    each member retains its own forecast evolution and local detail motion.
    This CPU inference operation does not change autoregressive feedback.
    """

    minimum_forecast_steps = 3

    def __init__(
        self,
        grid_shape: tuple[int, int],
        settings: Mapping,
        *,
        graph,
        radar_dataset: str = "radar",
        radar_variable: str = "lwe_precipitation_rate",
        hidden_dataset: str = "hidden",
        timestep_minutes: int = 5,
    ) -> None:
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.settings = dict(settings)
        self.radar_dataset = radar_dataset
        self.radar_variable = radar_variable
        self.timestep_minutes = timestep_minutes
        self._grid = None
        self._seam = None
        edges = graph[(hidden_dataset, "to", radar_dataset)].edge_index.detach().cpu().numpy()
        count = int(np.prod(self.grid_shape))
        degree = np.bincount(edges[1], minlength=count)
        if len(degree) != count or not np.all(degree == 2):
            raise ValueError("Precipitation seam cleanup requires two hidden sources per native node")
        pairs = np.empty((count, 2), dtype=np.int64)
        pairs[:, 0], pairs[:, 1] = np.iinfo(np.int64).max, -1
        np.minimum.at(pairs[:, 0], edges[1], edges[0])
        np.maximum.at(pairs[:, 1], edges[1], edges[0])
        if not np.all(pairs[:, 0] < pairs[:, 1]):
            raise ValueError("Precipitation seam cleanup requires two distinct hidden sources per native node")
        pairs = pairs.reshape(*self.grid_shape, 2)
        horizontal = np.any(pairs[:, 1:] != pairs[:, :-1], axis=-1)
        vertical = np.any(pairs[1:] != pairs[:-1], axis=-1)
        seam = np.zeros(self.grid_shape, dtype=bool)
        seam[:, :-1] |= horizontal
        seam[:, 1:] |= horizontal
        seam[:-1] |= vertical
        seam[1:] |= vertical
        self.register_buffer("seam_mask", torch.from_numpy(ndimage.binary_dilation(seam, iterations=1)))

    def __getstate__(self):
        state = super().__getstate__()
        state["_grid"] = None
        state["_seam"] = None
        return state

    def prepare_grid(self) -> None:
        """Build the immutable native remapping grid before threaded inference."""
        if self._grid is None:
            self._grid = np.mgrid[:self.grid_shape[0], :self.grid_shape[1]].astype(np.float32)
        if getattr(self, "_seam", None) is None:
            self._seam = self.seam_mask.detach().cpu().numpy()

    def prepare_correction(self, radar_history: Tensor, reference_prediction: Tensor) -> dict:
        """Prepare the correction state shared by every ensemble member."""
        count = int(np.prod(self.grid_shape))
        if radar_history.shape != (4, count):
            raise ValueError("Precipitation motion requires four radar history frames")
        reference = reference_prediction.detach().float().cpu().numpy().reshape(-1, *self.grid_shape)
        if len(reference) < self.minimum_forecast_steps:
            raise ValueError("Precipitation motion requires the member-zero forecast through three output steps")

        cv2.setNumThreads(1)
        history = radar_history.detach().float().cpu().numpy().reshape(4, *self.grid_shape)
        cfg = self.settings
        factor = cfg["pooling_factor"]
        prepared_history = [_motion_field(field, cfg) for field in history]
        past = [
            _motion_pair_from_fields(previous, current)
            for previous, current in zip(prepared_history[:-1], prepared_history[1:])
        ]
        historical = np.median(np.stack([_local_velocity(pair, cfg) for pair in past]), axis=0)
        mask = past[-1]["valid"] & (np.minimum(*past[-1]["rain"]) >= cfg["motion_rain_minimum_mm_h"])
        control = _regional_forecast(reference[:3], history[-1], cfg)
        bias, centre = _fit_rigid_velocity(historical - np.median(control[:3], axis=0), mask, factor)
        cy, cx = np.mgrid[:historical.shape[0], :historical.shape[1]].astype(np.float32)
        self.prepare_grid()
        return {
            "history": history[-1],
            "historical": historical,
            "bias": bias,
            "centre": centre,
            "cy": cy,
            "cx": cx,
            "ny": self._grid[0],
            "nx": self._grid[1],
            "source_valid": np.isfinite(history[-1]).astype(np.float32),
            "source": np.maximum(np.nan_to_num(history[-1], nan=0), 0),
            "seam": self._seam,
            "factor": factor,
        }

    def forward(
        self,
        prediction: Tensor,
        radar_history: Tensor,
        lead_minutes,
        *,
        reference_prediction: Tensor | None = None,
        prepared_correction: Mapping | None = None,
    ) -> Tensor:
        """Return CPU float32 ``[lead, member, node]`` without changing the inputs.

        Serial nonzero members must supply the same member-zero forecast through
        the first three leads. Short-horizon requests also need that reference.
        """
        count = int(np.prod(self.grid_shape))
        if prediction.ndim != 3 or prediction.shape[-1] != count or radar_history.shape != (4, count):
            raise ValueError("Precipitation motion requires full native forecasts and four radar history frames")
        if not np.array_equal(lead_minutes, self.timestep_minutes * np.arange(1, len(prediction) + 1)):
            raise ValueError("Precipitation motion requires consecutive positive forecast leads")
        cv2.setNumThreads(1)
        fields = prediction.detach().float().cpu().numpy()
        cfg = self.settings
        if prepared_correction is None:
            reference = fields[:, 0] if reference_prediction is None else reference_prediction
            if not isinstance(reference, Tensor):
                reference = torch.from_numpy(reference)
            prepared_correction = self.prepare_correction(radar_history, reference)
        history = prepared_correction["history"]
        historical = prepared_correction["historical"]
        bias = prepared_correction["bias"]
        centre = prepared_correction["centre"]
        cy, cx = prepared_correction["cy"], prepared_correction["cx"]
        ny, nx = prepared_correction["ny"], prepared_correction["nx"]
        source_valid = prepared_correction["source_valid"]
        source = prepared_correction["source"]
        seam = prepared_correction["seam"]
        factor = prepared_correction["factor"]
        result = np.empty_like(fields)
        for member in range(fields.shape[1]):
            original = fields[:, member].reshape(-1, *self.grid_shape)
            regional = _regional_forecast(original, history, cfg)
            model_backward = np.zeros_like(historical)
            for lead, image in enumerate(original):
                median = cv2.medianBlur(np.ascontiguousarray(np.nan_to_num(image, nan=0)), cfg["cleanup_size"])
                cleaned = np.maximum(np.where(seam & np.isfinite(image), median, image), 0)
                mass = np.nansum(cleaned, dtype=np.float64)
                if mass > 0:
                    cleaned *= np.nansum(np.maximum(image, 0), dtype=np.float64) / mass
                velocity = regional[lead]
                model_backward = -velocity + cv2.remap(
                    model_backward, cx - velocity[..., 0], cy - velocity[..., 1], cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                native_backward = cv2.resize(
                    model_backward, (historical.shape[1] * factor, historical.shape[0] * factor),
                    interpolation=cv2.INTER_LINEAR,
                )[:self.grid_shape[0], :self.grid_shape[1]] * factor
                mx, my = _inverse_rigid_map(nx, ny, bias, centre, lead + 1)
                transported_backward = cv2.remap(native_backward, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                ax, ay = mx + transported_backward[..., 0], my + transported_backward[..., 1]
                member_history = cv2.remap(source, ax, ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                detail_valid = cv2.remap(
                    source_valid, ax, ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                ) >= .99999
                valid = cv2.remap(
                    np.isfinite(cleaned).astype(np.float32), mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                ) >= .99999
                moved = cv2.remap(np.nan_to_num(cleaned, nan=0), mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                moved[~valid] = member_history[~valid]
                cutoff, full = cfg["protect_above_mm_h"], cfg["full_correction_below_mm_h"]
                detail = np.minimum(member_history, np.nextafter(np.float32(cutoff), np.float32(0)))
                phase = np.clip((moved - full) / (cutoff - full), 0, 1)
                gate = 1 - phase * phase * (3 - 2 * phase)
                editable = np.isfinite(moved) & detail_valid & (moved < cutoff)
                proposal = moved + np.float32(cfg["detail_strength"]) * gate * (detail - moved)
                proposal = np.clip(proposal, 0, np.nextafter(np.float32(cutoff), np.float32(0)))
                moved[editable] = proposal[editable]
                result[lead, member] = moved.ravel()
        return torch.from_numpy(result)
