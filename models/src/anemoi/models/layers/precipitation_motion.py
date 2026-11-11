# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Transport native radar texture using regional forecast motion and rain evolution."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np
import torch
from scipy import ndimage
from torch import Tensor
from torch import nn


def _motion_pair(previous: np.ndarray, current: np.ndarray, cfg: dict) -> dict:
    factor = int(cfg.get("pooling_factor", 4))
    height, width = previous.shape
    images, rains, validity = [], [], []
    for field in (previous, current):
        padding = ((0, (-height) % factor), (0, (-width) % factor))
        valid = np.pad(np.isfinite(field), padding, mode="edge")
        rain = np.pad(np.maximum(np.nan_to_num(field, nan=0), 0), padding, mode="edge")
        shape = (rain.shape[0] // factor, factor, rain.shape[1] // factor, factor)
        scale = float(cfg.get("motion_log_scale_mm_h", 0.05))
        image = np.log1p(rain / scale).reshape(shape).mean(axis=(1, 3))
        image *= 255 / np.log1p(5 / scale)
        image = ndimage.gaussian_filter(image.clip(0, 255), float(cfg.get("motion_image_sigma_coarse_cells", 1)))
        images.append(image.astype(np.float32))
        rains.append(rain.reshape(shape).mean(axis=(1, 3)))
        validity.append(valid.reshape(shape).mean(axis=(1, 3)) >= 0.999)
    forward = cv2.calcOpticalFlowFarneback(images[0], images[1], None, 0.5, 4, 21, 5, 5, 1.2, 0)
    gradient = np.maximum(
        np.hypot(ndimage.sobel(images[0], 0) / 8, ndimage.sobel(images[0], 1) / 8),
        np.hypot(ndimage.sobel(images[1], 0) / 8, ndimage.sobel(images[1], 1) / 8),
    )
    return {
        "images": images,
        "rain": rains,
        "valid": validity[0] & validity[1],
        "forward": forward,
        "gradient": gradient,
        "pooling_factor": factor,
        "native_shape": (height, width),
    }


def _local_velocity(pair: dict, cfg: dict) -> np.ndarray:
    flow = pair["forward"]
    height, width = flow.shape[:2]
    factor = pair["pooling_factor"]
    usable = pair["valid"] & (np.minimum(*pair["rain"]) >= cfg["motion_rain_minimum_mm_h"]) & (pair["gradient"] >= 1)
    fallback = np.median(flow[usable], axis=0) if usable.any() else np.zeros(2, dtype=np.float32)
    step = cfg["motion_tile_stride_native"] // factor
    radius = cfg["motion_tile_size_native"] // factor // 2
    rows, columns = np.arange(0, height + step, step), np.arange(0, width + step, step)
    knots = np.empty((len(rows), len(columns), 2), dtype=np.float32)
    for yi, y in enumerate(rows):
        for xi, x in enumerate(columns):
            selection = (
                slice(max(0, y - radius), min(height, y + radius)),
                slice(max(0, x - radius), min(width, x + radius)),
            )
            selected = usable[selection]
            knots[yi, xi] = (
                np.median(flow[selection][selected], axis=0)
                if selected.sum() >= cfg["minimum_motion_support_coarse_pixels"]
                else fallback
            )
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    result = cv2.remap(knots, xx / step, yy / step, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    result = ndimage.gaussian_filter(result, sigma=(1, 1, 0))
    maximum = cfg["maximum_motion_native_cells_per_step"] / factor
    result *= np.minimum(1, maximum / np.maximum(np.linalg.norm(result, axis=-1), 1e-6))[..., None]
    return result


def _fit_rigid_velocity(velocity: np.ndarray, mask: np.ndarray, factor: int) -> tuple:
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]].astype(np.float64)
    if not mask.any():
        mask = np.ones(mask.shape, dtype=bool)
    centre = np.asarray((xx[mask].mean() * factor, yy[mask].mean() * factor))
    x, y = xx[mask] * factor - centre[0], yy[mask] * factor - centre[1]
    design = np.zeros((2 * len(x), 3), dtype=np.float64)
    design[: len(x), 0], design[len(x) :, 1] = 1, 1
    design[: len(x), 2], design[len(x) :, 2] = -y, x
    vectors = velocity[mask].astype(np.float64) * factor
    target = np.r_[vectors[:, 0], vectors[:, 1]]
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return coefficients, centre


def _incompressible_velocity(velocity, support):
    height, width = velocity.shape[:2]
    if not np.isfinite(velocity).all():
        raise ValueError("Motion projection requires finite regional velocities")
    if not support.any():
        support = np.ones((height, width), dtype=bool)
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    xx -= xx[support].mean()
    yy -= yy[support].mean()
    design = np.zeros((2 * int(support.sum()), 3), dtype=np.float64)
    count = int(support.sum())
    design[:count, 0], design[count:, 1] = 1, 1
    design[:count, 2], design[count:, 2] = -yy[support], xx[support]
    original = velocity.astype(np.float64)
    coefficients = np.linalg.lstsq(design, np.r_[original[..., 0][support], original[..., 1][support]], rcond=None)[0]
    rigid = np.stack((coefficients[0] - coefficients[2] * yy, coefficients[1] + coefficients[2] * xx), axis=-1)
    residual = original - rigid
    extended = np.concatenate((residual, residual[:, ::-1]), axis=1)
    extended = np.concatenate((extended, extended[::-1]), axis=0)
    transformed = np.fft.rfft2(extended, axes=(0, 1))
    ky = (2 * np.pi * np.fft.fftfreq(2 * height))[:, None]
    kx = (2 * np.pi * np.fft.rfftfreq(2 * width))[None, :]
    squared = kx * kx + ky * ky
    squared[0, 0] = 1
    along = (kx * transformed[..., 0] + ky * transformed[..., 1]) / squared
    transformed[..., 0] -= kx * along
    transformed[..., 1] -= ky * along
    projected = np.fft.irfft2(transformed, s=(2 * height, 2 * width), axes=(0, 1))[:height, :width]
    projected_rigid = np.linalg.lstsq(
        design, np.r_[projected[..., 0][support], projected[..., 1][support]], rcond=None
    )[0]
    projected[..., 0] -= projected_rigid[0] - projected_rigid[2] * yy
    projected[..., 1] -= projected_rigid[1] + projected_rigid[2] * xx
    result = (rigid + projected).astype(np.float32)
    return result


def _advect_points(velocity, x, y, *, direction=1.0, substeps=4, return_valid=False):
    x, y = np.asarray(x, dtype=np.float32).copy(), np.asarray(y, dtype=np.float32).copy()
    valid = np.ones(x.shape, dtype=bool) if return_valid else None
    step = np.float32(direction / substeps)
    half = np.float32(0.5) * step
    for _ in range(substeps):
        k1 = cv2.remap(velocity, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        p2x, p2y = x + half * k1[..., 0], y + half * k1[..., 1]
        k2 = cv2.remap(velocity, p2x, p2y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        p3x, p3y = x + half * k2[..., 0], y + half * k2[..., 1]
        k3 = cv2.remap(velocity, p3x, p3y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        p4x, p4y = x + step * k3[..., 0], y + step * k3[..., 1]
        k4 = cv2.remap(velocity, p4x, p4y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if return_valid:
            for px, py in ((x, y), (p2x, p2y), (p3x, p3y), (p4x, p4y)):
                valid &= (px >= 0) & (px <= velocity.shape[1] - 1) & (py >= 0) & (py <= velocity.shape[0] - 1)
        x += step * (k1[..., 0] + 2 * k2[..., 0] + 2 * k3[..., 0] + k4[..., 0]) / 6
        y += step * (k1[..., 1] + 2 * k2[..., 1] + 2 * k3[..., 1] + k4[..., 1]) / 6
        if return_valid:
            valid &= (x >= 0) & (x <= velocity.shape[1] - 1) & (y >= 0) & (y <= velocity.shape[0] - 1)
    return (x, y, valid) if return_valid else (x, y)


def _native_map(coarse_map, grid, factor):
    x, y, valid = coarse_map
    yy, xx = grid
    cy, cx = np.mgrid[: x.shape[0], : x.shape[1]].astype(np.float32)
    width, height = x.shape[1] * factor, x.shape[0] * factor
    nx = (
        xx
        + cv2.resize((x - cx) * factor, (width, height), interpolation=cv2.INTER_LINEAR)[: xx.shape[0], : xx.shape[1]]
    )
    ny = (
        yy
        + cv2.resize((y - cy) * factor, (width, height), interpolation=cv2.INTER_LINEAR)[: yy.shape[0], : yy.shape[1]]
    )
    good = (
        cv2.resize(valid.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)[
            : yy.shape[0], : yy.shape[1]
        ]
        >= 0.99999
    )
    good &= (nx >= 0) & (ny >= 0) & (nx <= xx.shape[1] - 1) & (ny <= yy.shape[0] - 1)
    return nx, ny, good


class PrecipitationMotionHead(nn.Module):
    """Restore native radar texture after a complete autoregressive rollout.

    The network retains its original internal feedback. Its regional motion and
    aligned log-rain evolution modify recent radar, sampled once at each lead.
    The optional velocity projection suppresses artificial compression. Only
    rain evolution is spatially filtered; the native radar field is not blurred.

    Inputs and outputs are physical rain rates on the full gathered native grid.
    This CPU inference head uses the existing ``anemoi-models[lightning]`` extra.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        settings: Mapping,
        *,
        radar_dataset: str = "radar",
        radar_variable: str = "lwe_precipitation_rate",
        timestep_minutes: int = 5,
        divergence_free: bool = True,
    ) -> None:
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.settings = dict(settings)
        self.radar_dataset = radar_dataset
        self.radar_variable = radar_variable
        self.timestep_minutes = timestep_minutes
        self.divergence_free = divergence_free
        self.lightning_geometry = None
        self._grid = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_grid"] = None
        return state

    def compute_motion(
        self, prediction: Tensor, radar_history: Tensor, lead_minutes, *, reference_prediction=None
    ) -> list[dict]:
        """Return maps using one common reference forecast to retain member drift.

        ``reference_prediction[lead, node]`` is the member-zero first block. It
        defaults to the first supplied member; serial nonzero members must pass
        the same member-zero reference explicitly.
        """
        if prediction.ndim != 3 or prediction.shape[-1] != int(np.prod(self.grid_shape)):
            raise ValueError("Precipitation motion requires full native [lead, member, node] fields")
        if radar_history.shape != (4, prediction.shape[-1]):
            raise ValueError("Precipitation motion requires four radar frames ending at the forecast issue")
        if not np.array_equal(lead_minutes, self.timestep_minutes * np.arange(1, len(prediction) + 1)):
            raise ValueError("Precipitation motion requires consecutive leads from the first timestep")
        cv2.setNumThreads(1)
        fields = prediction.detach().float().cpu().numpy()
        history = radar_history.detach().float().cpu().numpy().reshape(4, *self.grid_shape)
        cfg = self.settings
        past = [_motion_pair(a, b, cfg) for a, b in zip(history[:-1], history[1:])]
        historical = np.median(np.stack([_local_velocity(pair, cfg) for pair in past]), axis=0)
        last = past[-1]
        past_support = (
            last["valid"] & (np.minimum(*last["rain"]) >= cfg["motion_rain_minimum_mm_h"]) & (last["gradient"] >= 1)
        )
        accepted_history = bool(past_support.sum() >= cfg["minimum_motion_support_coarse_pixels"])
        if not accepted_history:
            historical[:] = 0
        height, width = historical.shape[:2]
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        reference_fields = (
            fields[:, 0] if reference_prediction is None else reference_prediction.detach().float().cpu().numpy()
        )
        reference_fields = reference_fields.reshape(-1, *self.grid_shape)
        early_reference = []
        for index in cfg["reference_future_pair_indices"]:
            if index < len(reference_fields):
                pair = _motion_pair(reference_fields[index - 1], reference_fields[index], cfg)
                support = (
                    pair["valid"]
                    & (np.minimum(*pair["rain"]) >= cfg["motion_rain_minimum_mm_h"])
                    & (pair["gradient"] >= 1)
                )
                if support.sum() >= cfg["minimum_motion_support_coarse_pixels"]:
                    early_reference.append(_local_velocity(pair, cfg))
        common_reference = (
            np.median(np.stack(early_reference), axis=0) if early_reference else np.zeros_like(historical)
        )
        if accepted_history:
            bias, centre = _fit_rigid_velocity(historical - common_reference, past_support, 1)
        else:
            bias, centre = np.zeros(3), np.asarray(((width - 1) / 2, (height - 1) / 2))
        rigid_bias = np.stack(
            (bias[0] - bias[2] * (yy - centre[1]), bias[1] + bias[2] * (xx - centre[0])), axis=-1
        ).astype(np.float32)
        result = []
        for member in range(fields.shape[1]):
            member_fields = fields[:, member].reshape(-1, *self.grid_shape)
            pairs = [_motion_pair(a, b, cfg) for a, b in zip(member_fields[:-1], member_fields[1:])]
            masks = [
                pair["valid"] & (np.minimum(*pair["rain"]) >= cfg["motion_rain_minimum_mm_h"]) & (pair["gradient"] >= 1)
                for pair in pairs
            ]
            regional = [_local_velocity(pair, cfg) for pair in pairs]
            accepted = [bool(mask.sum() >= cfg["minimum_motion_support_coarse_pixels"]) for mask in masks]
            early = [
                regional[index - 1]
                for index in cfg["reference_future_pair_indices"]
                if index <= len(regional) and accepted[index - 1]
            ]
            reference = np.median(np.stack(early), axis=0) if early else np.zeros_like(historical)
            source_velocity = np.stack(
                [reference, *[value if ok else reference for value, ok in zip(regional, accepted)]]
            )
            desired_velocity = source_velocity + rigid_bias
            if "history_bias_gains" in cfg:
                gains = np.interp(lead_minutes, cfg["history_bias_lead_minutes"], cfg["history_bias_gains"])
                for index, gain in enumerate(gains):
                    desired_velocity[index] = source_velocity[index] + np.float32(gain) * rigid_bias
            if self.divergence_free:
                supports = [past_support, *[mask if ok else past_support for mask, ok in zip(masks, accepted)]]
                desired_velocity = np.stack(
                    [_incompressible_velocity(value, support) for value, support in zip(desired_velocity, supports)]
                )
            if cfg.get("n_step_output"):
                # Source paths retain the actual network jump for interpreting its growth.
                for index in range(cfg["n_step_output"], len(desired_velocity) - 1, cfg["n_step_output"]):
                    desired_velocity[index] = (desired_velocity[index - 1] + desired_velocity[index + 1]) * 0.5
            source_x, source_y = xx.copy(), yy.copy()
            source_valid = np.ones((height, width), dtype=bool)
            source_maps, backward_maps = [], []
            for lead, velocity in enumerate(source_velocity):
                source_x, source_y, good = _advect_points(
                    velocity, source_x, source_y, substeps=cfg["rk_substeps"], return_valid=True
                )
                source_valid &= good
                source_maps.append((source_x.copy(), source_y.copy(), source_valid.copy()))
                x, y, valid = xx.copy(), yy.copy(), np.ones((height, width), dtype=bool)
                for previous in desired_velocity[: lead + 1][::-1]:
                    x, y, good = _advect_points(
                        previous, x, y, direction=-1, substeps=cfg["rk_substeps"], return_valid=True
                    )
                    valid &= good
                backward_maps.append((x, y, valid))
            result.append(
                dict(
                    source_maps=source_maps,
                    backward_maps=backward_maps,
                    pooling_factor=cfg["pooling_factor"],
                    native_shape=self.grid_shape,
                )
            )
        return result

    def forward(self, prediction: Tensor, radar_history: Tensor, lead_minutes, *, motion=None) -> Tensor:
        """Return CPU float32 rain fields without mutating the network forecast."""
        fields = prediction.detach().float().cpu().numpy()
        history = radar_history.detach().float().cpu()
        if motion is None:
            motion = self.compute_motion(prediction, history, lead_minutes)
        if len(motion) != fields.shape[1]:
            raise ValueError("Precipitation motion must contain one set of maps per member")
        initial = history[-1].numpy().reshape(self.grid_shape)
        initial_valid = np.isfinite(initial)
        cfg = self.settings
        factor, scale = cfg["pooling_factor"], cfg["growth_log_scale_mm_h"]
        initial_log = np.log1p(np.maximum(np.where(initial_valid, initial, 0), 0) / scale)
        height, width = self.grid_shape
        padding = ((0, (-height) % factor), (0, (-width) % factor))
        pooled_shape = ((height + padding[0][1]) // factor, factor, (width + padding[1][1]) // factor, factor)
        if self._grid is None:
            self._grid = np.mgrid[:height, :width].astype(np.float32)
        output = np.empty(fields.shape, dtype=np.float32)
        for member, maps in enumerate(motion):
            for lead in range(len(fields)):
                field = fields[lead, member].reshape(self.grid_shape)
                model_valid = np.isfinite(field)
                model_log = np.log1p(np.maximum(np.where(model_valid, field, 0), 0) / scale)
                fx, fy, source_path = _native_map(maps["source_maps"][lead], self._grid, factor)
                aligned = cv2.remap(model_log, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                aligned_valid = (
                    cv2.remap(model_valid.astype(np.float32), fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    >= 0.99999
                )
                valid = initial_valid & aligned_valid & source_path
                weight = np.pad(valid, padding, mode="edge").reshape(pooled_shape).mean(axis=(1, 3)).astype(np.float32)
                difference = (
                    np.pad(np.where(valid, aligned - initial_log, 0), padding, mode="edge")
                    .reshape(pooled_shape)
                    .mean(axis=(1, 3))
                )
                sigma = cfg["growth_sigma_native"] / factor
                smooth_weight = ndimage.gaussian_filter(weight, sigma)
                growth = ndimage.gaussian_filter(difference, sigma) / np.maximum(smooth_weight, 1e-6)
                growth[smooth_weight < 0.5] = 0
                growth = cv2.resize(
                    growth.astype(np.float32),
                    (pooled_shape[2] * factor, pooled_shape[0] * factor),
                    interpolation=cv2.INTER_LINEAR,
                )[:height, :width]
                source_log = np.maximum(initial_log + growth, 0).astype(np.float32)
                source_rain = (scale * np.expm1(source_log)).astype(np.float32)
                cutoff, full = cfg["protect_above_mm_h"], cfg["full_correction_below_mm_h"]
                guide = np.minimum(
                    scale * np.expm1(initial_log), np.nextafter(np.float32(cutoff), np.float32(0))
                ).astype(np.float32)
                phase = np.clip((source_rain - full) / (cutoff - full), 0, 1)
                gate = 1 - phase * phase * (3 - 2 * phase)
                proposal = source_rain + np.float32(cfg["detail_strength"]) * gate * (guide - source_rain)
                proposal = np.clip(proposal, 0, np.nextafter(np.float32(cutoff), np.float32(0)))
                raised = np.isfinite(source_rain) & initial_valid & (source_rain < cutoff) & (proposal > source_rain)
                source_log[raised] = np.log1p(proposal[raised] / scale)
                source = np.where(initial_valid, source_log, 0).astype(np.float32)
                x, y, valid = _native_map(maps["backward_maps"][lead], self._grid, factor)
                known = (
                    cv2.remap(initial_valid.astype(np.float32), x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    >= 0.99999
                )
                moved = cv2.remap(source, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                transformed = (scale * np.expm1(moved)).astype(np.float32)
                output[lead, member] = np.where(valid & known, transformed, field).ravel()
        return torch.from_numpy(output)
