# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0

"""Absolute-lead ensemble conditioning for multi-step decoders."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch import nn
from torch.distributed import distributed_c10d
from torch.distributed.distributed_c10d import ProcessGroup


class AbsoluteLeadNoiseProcess(nn.Module):
    """Continuous member-noise process indexed by absolute forecast lead."""

    def __init__(
        self,
        *,
        noise_channels: int = 4,
        noise_std: float = 0.2,
        temporal_correlation: float = 0.95,
        temporal_correlation_steps: float = 6.0,
        lead_time_scale_steps: float = 24.0,
    ) -> None:
        super().__init__()
        if noise_channels < 1 or noise_std < 0.0:
            raise ValueError("noise_channels must be positive and noise_std must be non-negative")
        if not 0.0 <= temporal_correlation <= 1.0:
            raise ValueError("temporal_correlation must be between 0 and 1")
        if temporal_correlation_steps <= 0.0 or lead_time_scale_steps <= 0.0:
            raise ValueError("temporal_correlation_steps and lead_time_scale_steps must be positive")

        self.noise_channels = int(noise_channels)
        self.noise_std = float(noise_std)
        self.temporal_correlation = float(temporal_correlation)
        self.temporal_correlation_steps = float(temporal_correlation_steps)
        self.lead_time_scale_steps = float(lead_time_scale_steps)
        self._previous_noise: Tensor | None = None
        self._previous_lead: Tensor | None = None
        self._noise_generator: torch.Generator | None = None
        self._noise_generator_device: torch.device | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_noise_generator"] = None
        state["_noise_generator_device"] = None
        state["_previous_noise"] = None
        state["_previous_lead"] = None
        return state

    def reset_state(self) -> None:
        """Reset state carried between consecutive forecast blocks."""
        self._previous_noise = None
        self._previous_lead = None

    def forward(
        self,
        *,
        lead_steps: Tensor,
        batch_size: int,
        ensemble_size: int,
        dtype: torch.dtype,
        device: torch.device,
        model_comm_group: ProcessGroup | None = None,
        reset_state: bool = False,
    ) -> Tensor:
        if reset_state:
            self.reset_state()
        lead_steps = torch.as_tensor(lead_steps, dtype=torch.float32, device=device)
        if lead_steps.ndim != 1 or lead_steps.numel() < 1 or torch.any(lead_steps[1:] <= lead_steps[:-1]):
            raise ValueError("lead_steps must be a non-empty, strictly increasing one-dimensional tensor")
        if self._previous_lead is not None and lead_steps[0] <= self._previous_lead:
            raise ValueError("lead_steps must continue after the previous forecast block")

        if self._noise_generator is None or self._noise_generator_device != device:
            self._noise_generator = torch.Generator(device=device)
            self._noise_generator.manual_seed(torch.initial_seed() + 104729)
            self._noise_generator_device = device

        process = []
        previous_noise = self._previous_noise
        previous_lead = self._previous_lead
        group_size = distributed_c10d.get_world_size(model_comm_group) if model_comm_group is not None else 1
        group_rank = distributed_c10d.get_rank(model_comm_group) if model_comm_group is not None else 0
        group_src = (
            distributed_c10d.get_global_rank(model_comm_group, 0)
            if model_comm_group is not None and group_size > 1
            else 0
        )
        for lead in lead_steps:
            if group_rank == 0:
                innovation = torch.randn(
                    batch_size,
                    ensemble_size,
                    self.noise_channels,
                    dtype=dtype,
                    device=device,
                    generator=self._noise_generator,
                )
            else:
                innovation = torch.empty(
                    batch_size,
                    ensemble_size,
                    self.noise_channels,
                    dtype=dtype,
                    device=device,
                )
            if group_size > 1:
                distributed_c10d.broadcast(innovation, src=group_src, group=model_comm_group)

            if previous_noise is None:
                current_noise = self.noise_std * innovation
            else:
                delta = float((lead - previous_lead).item())
                correlation = self.temporal_correlation ** (delta / self.temporal_correlation_steps)
                current_noise = correlation * previous_noise + (
                    self.noise_std * math.sqrt(max(0.0, 1.0 - correlation**2)) * innovation
                )
            process.append(current_noise)
            previous_noise = current_noise
            previous_lead = lead

        self._previous_noise = previous_noise.detach()
        self._previous_lead = previous_lead.detach()
        noise = torch.stack(process, dim=2)
        scaled_lead = lead_steps.to(dtype=dtype) / self.lead_time_scale_steps
        time_condition = torch.stack(
            (
                scaled_lead,
                scaled_lead.square(),
                torch.sin(math.pi * scaled_lead),
                torch.cos(math.pi * scaled_lead),
                torch.sin(2.0 * math.pi * scaled_lead),
                torch.cos(2.0 * math.pi * scaled_lead),
            ),
            dim=-1,
        )
        time_condition = time_condition[None, None].expand(batch_size, ensemble_size, -1, -1)
        return torch.cat((time_condition, noise), dim=-1).flatten(0, 1)


class DecoderAbsoluteLeadNoiseConditioner(nn.Module):
    """Zero-start feature modulation shared by every forecast lead."""

    def __init__(
        self,
        *,
        x_dim: int,
        output_steps: int,
        cond_dim: int,
        hidden: int = 128,
        zero_mean_across_output_steps: bool = False,
    ) -> None:
        super().__init__()
        if x_dim < 1 or output_steps < 1 or cond_dim < 1 or hidden < 1:
            raise ValueError("conditioner dimensions must be positive")
        self.x_dim = int(x_dim)
        self.output_steps = int(output_steps)
        self.zero_mean_across_output_steps = bool(zero_mean_across_output_steps)
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * self.x_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        x: Tensor,
        projection: nn.Linear,
        cond: Tensor,
        cond_indices: Tensor | None = None,
    ) -> Tensor:
        if cond.ndim != 3 or cond.shape[1] != self.output_steps:
            raise ValueError(
                f"Expected condition shape (batch*ensemble, {self.output_steps}, channels), got {tuple(cond.shape)}"
            )
        if projection.out_features % self.output_steps != 0:
            raise ValueError(
                f"Decoder output dimension {projection.out_features} is not divisible by {self.output_steps} leads"
            )
        if cond.device != x.device or cond.dtype != x.dtype:
            cond = cond.to(device=x.device, dtype=x.dtype)

        gamma, beta = self.net(cond).chunk(2, dim=-1)
        output_dim = projection.out_features // self.output_steps
        projection_weight = projection.weight.view(self.output_steps, output_dim, self.x_dim)
        if cond.shape[0] == 1:
            modulated_weight = projection_weight * gamma[0, :, None, :]
            correction = torch.einsum("nd,tvd->ntv", x, modulated_weight)
            correction = correction + torch.einsum("td,tvd->tv", beta[0], projection_weight)[None]
            if self.zero_mean_across_output_steps:
                correction = correction - correction.mean(dim=1, keepdim=True)
            return correction.reshape(x.shape[0], projection.out_features)
        if cond_indices is not None:
            if cond_indices.ndim != 1 or cond_indices.shape[0] != x.shape[0]:
                raise ValueError(
                    "Condition indices must have one entry for every decoder feature row: "
                    f"{tuple(cond_indices.shape)} for {x.shape[0]} rows."
                )
            cond_indices = cond_indices.to(device=x.device, dtype=torch.long)
            correction = x.new_empty(x.shape[0], self.output_steps, output_dim)
            for condition_index in range(cond.shape[0]):
                row_indices = torch.nonzero(cond_indices == condition_index, as_tuple=False).flatten()
                if row_indices.numel() == 0:
                    continue
                features = torch.index_select(x, 0, row_indices)
                modulated_weight = projection_weight * gamma[condition_index, :, None, :]
                condition_correction = torch.einsum("nd,tvd->ntv", features, modulated_weight)
                condition_correction = condition_correction + torch.einsum(
                    "td,tvd->tv",
                    beta[condition_index],
                    projection_weight,
                )[None]
                correction.index_copy_(0, row_indices, condition_correction)
            if self.zero_mean_across_output_steps:
                correction = correction - correction.mean(dim=1, keepdim=True)
            return correction.reshape(x.shape[0], projection.out_features)

        if x.shape[0] % cond.shape[0] != 0:
            raise ValueError(
                f"Decoder feature rows ({x.shape[0]}) must be divisible by condition rows ({cond.shape[0]})."
            )

        grid_size = x.shape[0] // cond.shape[0]
        features = x.view(cond.shape[0], grid_size, self.x_dim)
        modulated_weight = projection_weight[None] * gamma[:, :, None, :]
        correction = torch.einsum("bgd,btvd->bgtv", features, modulated_weight)
        correction = correction + torch.einsum("btd,tvd->btv", beta, projection_weight)[:, None]
        if self.zero_mean_across_output_steps:
            correction = correction - correction.mean(dim=2, keepdim=True)
        return correction.reshape(x.shape[0], projection.out_features)
