# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""History-based lightning motion correction applied after an autoregressive rollout."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from torch import Tensor
from torch import nn


class LightningMotionHead(nn.Module):
    """Blend lightning forecasts with recent occurrences transported by radar motion.

    Predictions and observations must be in physical units. ``forward`` consumes
    seven radar and five lightning frames, ending at the forecast issue time, and
    returns individual forecast probability scores. It does not change the raw
    decoder outputs or their autoregressive feedback. Motion estimation and spatial
    aggregation run on the CPU; the result is an uncalibrated score.

    Call ``set_grid`` with the full native coordinates before inference. Spatial
    lookup arrays are cached on the host and are rebuilt when a checkpoint loads.
    The optional ``anemoi-models[lightning]`` dependencies provide the projection
    and optical-flow implementations.
    """

    def __init__(
        self,
        *,
        projection: str,
        logit_scale: float,
        grid_size: float = 8000.0,
        history_weight: float = 0.25,
        multiplier: float = 1.0,
        bias: float = -8.0,
        history_sigma: float = 0.5,
        history_decay_minutes: float = 15.0,
        forecast_decay_minutes: float = 60.0,
        timestep_minutes: int = 5,
        max_lead_minutes: int = 30,
        max_speed: float = 40.0,
        radar_dataset: str = "radar",
        lightning_dataset: str = "lightning",
        radar_variable: str = "lwe_precipitation_rate",
        lightning_variable: str = "thunder_count",
        output_variable: str = "lightning_probability_score",
    ) -> None:
        super().__init__()
        if logit_scale <= 0 or grid_size <= 0 or timestep_minutes != 5 or max_lead_minutes != 30:
            raise ValueError("LightningMotionHead requires positive scales and the validated five-minute, 30-minute recipe")
        if not 0 <= history_weight <= 1 or min(history_decay_minutes, forecast_decay_minutes, max_speed) <= 0:
            raise ValueError("Invalid lightning motion weight, decay, or speed")
        self.projection = projection
        self.logit_scale = logit_scale
        self.grid_size = grid_size
        self.multiplier = multiplier
        self.bias = bias
        self.history_sigma = history_sigma
        self.history_decay_minutes = history_decay_minutes
        self.forecast_decay_minutes = forecast_decay_minutes
        self.timestep_minutes = timestep_minutes
        self.max_lead_minutes = max_lead_minutes
        self.max_speed = max_speed
        self.radar_dataset = radar_dataset
        self.lightning_dataset = lightning_dataset
        self.radar_variable = radar_variable
        self.lightning_variable = lightning_variable
        self.output_variable = output_variable
        self.register_buffer("history_weight", torch.tensor(history_weight, dtype=torch.float32))
        self._radar_index = None
        self._lightning_index = None
        self._shape = None
        self._origin = None

    def __getstate__(self):
        state = super().__getstate__()
        for name in ("_radar_index", "_lightning_index", "_shape", "_origin"):
            state[name] = None
        return state

    def set_grid(self, *, radar_latitudes, radar_longitudes, lightning_latitudes, lightning_longitudes) -> None:
        """Prepare native-to-projected-grid mappings without retaining source datasets."""
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", self.projection, always_xy=True)
        self._shape = None
        self._origin = None
        self._lightning_index = self._spatial_index(lightning_latitudes, lightning_longitudes, transformer)
        self._radar_index = self._spatial_index(radar_latitudes, radar_longitudes, transformer)
        np.testing.assert_array_equal(self._lightning_index[0], np.arange(len(lightning_latitudes)))

    def _spatial_index(self, latitudes, longitudes, transformer) -> tuple[np.ndarray, np.ndarray]:
        x, y = transformer.transform(np.asarray(longitudes, dtype=np.float32), np.asarray(latitudes, dtype=np.float32))
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            raise ValueError("No finite coordinates for the lightning motion grid")
        if self._origin is None:
            self._origin = (float(np.floor(x[finite].min() / self.grid_size) * self.grid_size),
                            float(np.floor(y[finite].min() / self.grid_size) * self.grid_size))
            self._shape = (int(np.floor((y[finite].max() - self._origin[1]) / self.grid_size)) + 1,
                           int(np.floor((x[finite].max() - self._origin[0]) / self.grid_size)) + 1)
        indices = np.flatnonzero(finite)
        ix = np.floor((x[indices] - self._origin[0]) / self.grid_size).astype(np.int64)
        iy = np.floor((y[indices] - self._origin[1]) / self.grid_size).astype(np.int64)
        inside = (ix >= 0) & (ix < self._shape[1]) & (iy >= 0) & (iy < self._shape[0])
        return indices[inside], iy[inside] * self._shape[1] + ix[inside]

    def _aggregate(self, values: np.ndarray, index, *, occurrence: bool = False) -> np.ndarray:
        selected = values[index[0]]
        valid = np.isfinite(selected)
        weights = (valid & (selected > 0)).astype(np.float32) if occurrence else np.where(valid, selected, 0)
        total = np.bincount(index[1], weights=weights, minlength=int(np.prod(self._shape)))
        if occurrence:
            return (total > 0).reshape(self._shape).astype(np.float32)
        count = np.bincount(index[1], weights=valid, minlength=len(total))
        return np.divide(total, count, out=np.zeros_like(total), where=count > 0).reshape(self._shape).astype(np.float32)

    def _velocity(self, previous: np.ndarray, current: np.ndarray, minutes: int) -> np.ndarray:
        import cv2

        # Avoid nested OpenCV thread pools across ensemble inference workers.
        cv2.setNumThreads(1)
        images = [np.round(255 * np.clip(np.log1p(np.maximum(field, 0)) / np.log(21), 0, 1)).astype(np.uint8)
                  for field in (previous, current)]
        backward = cv2.calcOpticalFlowFarneback(images[1], images[0], None, 0.5, 3, 15, 5, 5, 1.1, 0)
        velocity = -backward / (minutes / self.timestep_minutes)
        speed = np.linalg.norm(velocity, axis=-1)
        maximum = self.max_speed * self.timestep_minutes * 60 / self.grid_size
        velocity *= np.minimum(1, maximum / np.maximum(speed, 1e-8))[..., None]
        return velocity.astype(np.float32)

    def _transport(self, field: np.ndarray, velocity: np.ndarray, steps: float) -> np.ndarray:
        yy, xx = np.indices(field.shape, dtype=np.float32)
        source = [yy - steps * velocity[..., 1], xx - steps * velocity[..., 0]]
        return ndimage.map_coordinates(field, source, order=1, mode="constant", cval=0, prefilter=False)

    def _shared_source_coordinates(self, entry, coarse_map, geometry):
        x, y, path_valid = coarse_map
        cy, cx = np.indices(path_valid.shape, dtype=np.float32)
        displacement = np.stack((x - cx, y - cy), axis=-1).astype(np.float64)
        factor = entry["pooling_factor"]
        radar_x, radar_y = geometry["radar_x"], geometry["radar_y"]
        coarse_x = (radar_x.astype(np.float64) + .5) / factor - .5
        coarse_y = (radar_y.astype(np.float64) + .5) / factor - .5
        ix, iy = np.floor(coarse_x).astype(np.int64), np.floor(coarse_y).astype(np.int64)
        height, width = path_valid.shape
        inside = (ix >= 0) & (ix + 1 < width) & (iy >= 0) & (iy + 1 < height)
        ix, iy = ix.clip(0, width - 2), iy.clip(0, height - 2)
        finite = np.isfinite(displacement).all(axis=-1) & path_valid
        supported = (geometry["radar_coverage"] & inside & finite[iy, ix] & finite[iy, ix + 1]
            & finite[iy + 1, ix] & finite[iy + 1, ix + 1])
        source_x, source_y = [radar + factor * ndimage.map_coordinates(displacement[..., component],
            [coarse_y, coarse_x], order=1, mode="constant", cval=np.nan, prefilter=False)
            for component, radar in enumerate((radar_x, radar_y))]
        world = RegularGridInterpolator((geometry["sample_y"], geometry["sample_x"]), geometry["radar_world"],
            bounds_error=False, fill_value=np.nan)(np.stack((source_y, source_x), axis=-1))
        delta = (world - geometry["target_radar_world"]) / self.grid_size
        yy, xx = np.indices(self._shape, dtype=np.float64)
        source = np.stack((xx, yy), axis=-1) + delta
        native_height, native_width = geometry["radar_shape"]
        supported &= (np.isfinite(source).all(axis=-1) & (source_x >= 0) & (source_x <= native_width - 1)
            & (source_y >= 0) & (source_y <= native_height - 1)
            & (source[..., 0] >= 0) & (source[..., 0] <= xx.shape[1] - 1)
            & (source[..., 1] >= 0) & (source[..., 1] <= yy.shape[0] - 1))
        source[~supported] = np.stack((xx, yy), axis=-1)[~supported]
        return source[..., 0].astype(np.float32), source[..., 1].astype(np.float32), supported

    def forward(self, prediction: Tensor, radar_history: Tensor, lightning_history: Tensor, lead_minutes,
                *, motion=None, geometry=None) -> Tensor:
        """Correct ``prediction[lead, member, node]`` using past ``history[time, node]``.

        The result is a CPU float32 tensor with the prediction shape. Analysis and
        unvalidated lead times remain NaN. All native fields must be gathered before
        calling the head; no communication or random sampling is performed here.
        Optional shared precipitation maps sample the original issue guide once;
        existing radar-history transport supplies values outside shared coverage.
        """
        if self._radar_index is None or self._lightning_index is None:
            raise RuntimeError("Call LightningMotionHead.set_grid before inference")
        if prediction.ndim != 3 or prediction.shape[-1] != len(self._lightning_index[0]):
            raise ValueError("Lightning motion requires full native prediction fields [lead, member, node]")
        if radar_history.shape[0] != 7 or lightning_history.shape[0] != 5:
            raise ValueError("Lightning motion requires radar T-30..T and lightning T-20..T")
        leads = np.asarray(lead_minutes, dtype=int)
        valid_leads = np.flatnonzero((leads > 0) & (leads <= self.max_lead_minutes))
        if not np.array_equal(leads[valid_leads], np.arange(5, 31, 5)):
            raise ValueError("Lightning motion requires every five-minute forecast through +30 minutes")
        if (motion is None) != (geometry is None):
            raise ValueError("Shared lightning motion requires both motion and geometry")
        if motion is not None:
            if (str(geometry["projection"]) != self.projection or float(geometry["grid_size"]) != self.grid_size
                or tuple(geometry["lightning8_shape"]) != self._shape or not np.array_equal(geometry["origin"], self._origin)):
                raise ValueError("Shared lightning geometry differs from the active projected grid")
            if len(motion) != prediction.shape[1] or any(len(entry["backward_maps"]) < 6
                or tuple(entry["native_shape"]) != tuple(geometry["radar_shape"]) for entry in motion):
                raise ValueError("Shared lightning motion must match all members and six native-radar lead maps")
        radar = radar_history.detach().float().cpu().numpy()
        lightning = lightning_history.detach().float().cpu().numpy()
        radar = np.stack([self._aggregate(frame, self._radar_index) for frame in radar])
        lightning = np.stack([self._aggregate(frame, self._lightning_index, occurrence=True) for frame in lightning])
        velocity = 0.5 * (self._velocity(radar[0], radar[-1], 30) + self._velocity(radar[3], radar[-1], 15))
        history = np.zeros(self._shape, dtype=np.float32)
        for hi in range(5):
            age = 4 - hi
            source = ndimage.gaussian_filter(lightning[hi], sigma=self.history_sigma)
            source = source * np.exp(-age * self.timestep_minutes / self.history_decay_minutes)
            history = np.maximum(history, self._transport(source, velocity, age))
        original = prediction.detach().float().cpu()
        output = torch.full_like(original, torch.nan)
        weight = float(self.history_weight.detach().cpu())
        issue_guide = history.copy() if motion is not None else None
        for index in valid_leads:
            history = self._transport(history, velocity, 1)
            evidence = history.astype(np.float32).reshape(-1)[self._lightning_index[1]]
            evidence = evidence * np.float32(np.exp(-leads[index] / self.forecast_decay_minutes))
            if motion is not None:
                evidence = np.broadcast_to(evidence, (len(motion), evidence.size)).copy()
                for member, entry in enumerate(motion):
                    step = int(leads[index] // self.timestep_minutes) - 1
                    x, y, supported = self._shared_source_coordinates(entry, entry["backward_maps"][step], geometry)
                    shared = ndimage.map_coordinates(issue_guide, [y, x], order=1, mode="constant", cval=0, prefilter=False)
                    shared = np.where(supported, shared, history).astype(np.float32)
                    evidence[member] = shared.reshape(-1)[self._lightning_index[1]] * np.float32(np.exp(-leads[index] / self.forecast_decay_minutes))
            logits = original[index].numpy() / self.logit_scale * self.multiplier + self.bias
            probability = np.empty_like(logits, dtype=np.float32)
            positive = logits >= 0
            probability[positive] = 1 / (1 + np.exp(-logits[positive]))
            exp_values = np.exp(logits[~positive])
            probability[~positive] = exp_values / (1 + exp_values)
            output[index] = torch.from_numpy((1 - weight) * probability + weight * (evidence[None, :] if motion is None else evidence))
        return output
