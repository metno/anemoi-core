# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import math
from abc import ABC
from abc import abstractmethod
from typing import Optional

import einops
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.distributed.utils import model_is_distributed
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.mlp import MLP
from anemoi.models.layers.sparse_projector import SparseProjector
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class BaseNoiseInjector(nn.Module, ABC):
    """Abstract base class for noise injection strategies.

    Subclasses must implement the forward method which takes an input tensor
    and returns a tuple of (modified_tensor, noise_or_none).
    """

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        batch_size: int,
        ensemble_size: int,
        grid_size: int,
        grid_shard_sizes: ShardSizes,
        noise_dtype: torch.dtype = torch.float32,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Forward pass for noise injection.

        Parameters
        ----------
        x : Tensor
            Input tensor to potentially modify
        batch_size : int
            Batch size
        ensemble_size : int
            Ensemble size
        grid_size : int
            Grid size
        grid_shard_sizes : ShardSizes
            Per-rank partition sizes along the sharded dimension, or None if not sharded
        noise_dtype : torch.dtype, optional
            Data type for noise tensor
        model_comm_group : ProcessGroup, optional
            Model communication group

        Returns
        -------
        tuple[Tensor, Optional[Tensor]]
            Tuple of (output_tensor, noise_tensor_or_none):
                - output_tensor: The (potentially) modified input tensor
                - noise_tensor_or_none: The noise tensor for conditioning,
                  or None if noise is injected directly into output_tensor
        """
        ...


class NoOpNoiseInjector(BaseNoiseInjector):
    """No-op noise injector that passes through input unchanged.

    Use this when noise injection is disabled.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize NoOpNoiseInjector."""
        super().__init__()

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        ensemble_size: int,
        grid_size: int,
        grid_shard_sizes: ShardSizes,
        noise_dtype: torch.dtype = torch.float32,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, None]:
        """Pass through input unchanged with no noise."""
        return x, None


class NoiseConditioning(BaseNoiseInjector):
    """Noise Conditioning."""

    def __init__(
        self,
        *,
        noise_std: int,
        noise_channels_dim: int,
        noise_mlp_hidden_dim: int,
        layer_kernels: DotDict,
        noise_matrix: Optional[str] = None,
        noise_edges_name: Optional[tuple[str, str, str]] = None,
        edge_weight_attribute: Optional[str] = None,
        row_normalize_noise_matrix: bool = False,
        autocast: bool = False,
        sparse_projector_num_chunks: int = 1,
        num_channels: Optional[int] = None,
        graph_data: Optional[HeteroData] = None,
    ) -> None:
        """Initialize NoiseConditioning."""
        super().__init__()
        assert noise_channels_dim > 0, "Noise channels must be a positive integer"
        assert noise_mlp_hidden_dim > 0, "Noise channels must be a positive integer"

        self.noise_std = noise_std

        # Noise channels
        self.noise_channels = noise_channels_dim

        self.layer_factory = load_layer_kernels(layer_kernels)

        self.noise_mlp = MLP(
            noise_channels_dim,
            noise_mlp_hidden_dim,
            noise_channels_dim,
            layer_kernels=self.layer_factory,
            n_extra_layers=0,
            final_activation=False,
            layer_norm=True,
        )

        self.noise_graph_provider = None
        self._sparse_projector = None
        assert not (
            noise_matrix is not None and noise_edges_name is not None
        ), "Specify either noise_matrix or noise_edges_name, not both."

        if noise_edges_name is not None:
            assert graph_data is not None, "graph_data must be provided when using noise_edges_name."
            self.noise_graph_provider = ProjectionGraphProvider(
                graph=graph_data,
                edges_name=tuple(noise_edges_name),
                edge_weight_attribute=edge_weight_attribute,
                row_normalize=row_normalize_noise_matrix,
            )
            self._sparse_projector = SparseProjector(autocast=autocast, num_chunks=sparse_projector_num_chunks)
            LOGGER.info("Noise projector matrix shape = %s", self.noise_graph_provider.projection_matrix.shape)

        if noise_matrix is not None:
            self.noise_graph_provider = ProjectionGraphProvider(
                file_path=noise_matrix,
                row_normalize=row_normalize_noise_matrix,
            )
            self._sparse_projector = SparseProjector(autocast=autocast, num_chunks=sparse_projector_num_chunks)
            LOGGER.info("Noise projector matrix shape = %s", self.noise_graph_provider.projection_matrix.shape)

        LOGGER.info("processor noise channels = %d", self.noise_channels)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        ensemble_size: int,
        grid_size: int,
        grid_shard_sizes: ShardSizes,
        noise_dtype: torch.dtype = torch.float32,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor]:

        noise_shape = (
            batch_size,
            ensemble_size,
            grid_size if self.noise_graph_provider is None else self.noise_graph_provider.projection_matrix.shape[1],
            self.noise_channels,
        )

        noise = torch.randn(size=noise_shape, dtype=noise_dtype, device=x.device) * self.noise_std
        noise.requires_grad = False

        if self.noise_graph_provider is not None:
            channel_shard_sizes = get_shard_sizes(noise, -1, model_comm_group)
            noise = shard_tensor(noise, -1, channel_shard_sizes, model_comm_group)  # split across channels

            noise = einops.rearrange(
                noise, "batch ensemble grid vars -> (batch ensemble) grid vars"
            )  # batch and ensemble always 1 when sharded

            projection_matrix = self.noise_graph_provider.get_edges(device=noise.device)
            noise = self._sparse_projector(noise, projection_matrix)  # to shape of hidden grid

            noise = einops.rearrange(noise, "bse grid vars -> (bse grid) vars")  # shape of x
            noise = all_to_all_transpose(
                noise, 0, grid_shard_sizes, -1, channel_shard_sizes, model_comm_group
            )  # sharded grid dim, full channels
        else:
            noise = einops.rearrange(noise, "batch ensemble grid vars -> (batch ensemble grid) vars")  # shape of x
            noise_shard_sizes = get_shard_sizes(noise, 0, model_comm_group)
            noise = shard_tensor(noise, 0, noise_shard_sizes, model_comm_group)  # sharded grid dim, full channels

        noise = checkpoint(self.noise_mlp, noise, use_reentrant=False)

        LOGGER.debug("Noise noise.shape = %s, noise.norm: %.9e", noise.shape, torch.linalg.norm(noise))

        return x, noise


class NoiseInjector(BaseNoiseInjector):
    """Noise Injection Module.

    Generates noise and projects it directly into the input tensor,
    returning None for the noise (since it's already incorporated).
    """

    def __init__(
        self,
        *,
        noise_std: int,
        noise_channels_dim: int,
        noise_mlp_hidden_dim: int,
        num_channels: int,
        layer_kernels: DotDict,
        noise_matrix: Optional[str] = None,
        graph_data: Optional[HeteroData] = None,
    ) -> None:
        """Initialize NoiseInjector.

        Parameters
        ----------
        noise_std : int
            Standard deviation for noise generation
        noise_channels_dim : int
            Number of noise channels
        noise_mlp_hidden_dim : int
            Hidden dimension of noise MLP
        num_channels : int
            Number of model channels for projection
        layer_kernels : DotDict
            Layer kernel configurations
        noise_matrix : str, optional
            Optional path to noise truncation matrix
        graph_data : Optional[HeteroData], optional
            Graph data for noise conditioning.
        """
        super().__init__()

        self._noise_conditioning = NoiseConditioning(
            noise_std=noise_std,
            noise_channels_dim=noise_channels_dim,
            noise_mlp_hidden_dim=noise_mlp_hidden_dim,
            layer_kernels=layer_kernels,
            noise_matrix=noise_matrix,
            graph_data=graph_data,
        )
        self.noise_channels = noise_channels_dim
        self.projection = nn.Linear(num_channels + self.noise_channels, num_channels)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        ensemble_size: int,
        grid_size: int,
        grid_shard_sizes: ShardSizes,
        noise_dtype: torch.dtype = torch.float32,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, None]:
        """Generate noise and inject it into the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor to modify
        batch_size : int
            Batch size
        ensemble_size : int
            Ensemble size
        grid_size : int
            Grid size
        grid_shard_sizes : ShardSizes
            Per-rank partition sizes along the sharded dimension, or None if not sharded
        noise_dtype : torch.dtype, optional
            Data type for noise tensor
        model_comm_group : ProcessGroup, optional
            Model communication group

        Returns
        -------
        tuple[Tensor, None]
            Tuple of (modified_x, None): Modified tensor with noise injected
        """
        x, noise = self._noise_conditioning(
            x=x,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            grid_size=grid_size,
            grid_shard_sizes=grid_shard_sizes,
            noise_dtype=noise_dtype,
            model_comm_group=model_comm_group,
        )

        return (
            self.projection(torch.cat([x, noise], dim=-1)),
            None,
        )


class LatticeNoiseConditioning(nn.Module):
    """Spatially correlated noise on the data nodes, as a conditioning for the decoder.

    Motivation: in the ensemble model the only stochastic input is the noise drawn at the hidden
    nodes, so member-to-member spread at scales below the hidden mesh can only reach the output as
    a pattern locked to the hidden-node positions (the decoder maps every data node from its k
    nearest hidden nodes and has no data-data edges). This module gives the decoder a second noise
    source that lives on the data grid itself and is statistically homogeneous there.

    Independent N(0, 1) noise is drawn on the full regular lattice of the LAM data grid, smoothed
    with a separable Gaussian kernel per channel group (``sigmas`` in lattice cells, with
    ``channels_per_scale`` channels each), rescaled to unit variance, gathered at the LAM data nodes
    (zero at all other data nodes), passed through a small MLP (as the hidden-node noise is) and
    returned with shape ((batch ensemble grid), channels), sharded along the grid dimension of the
    model communication group like the data-node features. It is meant as the destination-side
    ``cond`` of a decoder whose layer norms are ConditionalLayerNorms with
    ``condition_shape = len(sigmas) * channels_per_scale``.

    Channels are split across the ranks of the model communication group for the generation and the
    smoothing (each channel is produced on exactly one rank, so the smoothed field is consistent
    across the grid shards) and transposed to grid-sharded afterwards.

    Requires a node attribute ``lattice_attribute`` on the data nodes of the graph: an integer
    tensor (num_nodes, 2) with the lattice (row, column) of every LAM node and -1 for nodes off the
    lattice.
    """

    def __init__(
        self,
        *,
        sigmas: list[float],
        channels_per_scale: int,
        noise_mlp_hidden_dim: int,
        layer_kernels: DotDict,
        graph_data: HeteroData,
        data_nodes_name: str = "data",
        lattice_attribute: str = "lattice_ij",
        noise_std: float = 1.0,
        truncate: float = 3.0,
        **kwargs,
    ) -> None:
        """Initialize LatticeNoiseConditioning.

        Parameters
        ----------
        sigmas : list[float]
            Standard deviation of the Gaussian smoothing kernel per channel group, in lattice cells.
        channels_per_scale : int
            Number of noise channels per entry of ``sigmas``.
        noise_mlp_hidden_dim : int
            Hidden dimension of the noise MLP.
        layer_kernels : DotDict
            Layer kernel configuration for the MLP.
        graph_data : HeteroData
            Graph with the lattice attribute on the data nodes.
        data_nodes_name : str
            Name of the data nodes in the graph.
        lattice_attribute : str
            Name of the (num_nodes, 2) lattice-index attribute of the data nodes.
        noise_std : float
            Standard deviation of the (smoothed, unit-variance) noise fed to the MLP.
        truncate : float
            Kernel half-width in standard deviations.
        """
        super().__init__()
        if kwargs:
            LOGGER.warning("LatticeNoiseConditioning ignores kwargs %s", list(kwargs))
        self.sigmas = [float(v) for v in sigmas]
        self.channels_per_scale = int(channels_per_scale)
        self.noise_channels = len(self.sigmas) * self.channels_per_scale
        self.noise_std = noise_std

        ij = graph_data[data_nodes_name][lattice_attribute].long().reshape(-1, 2)
        on_lattice = (ij[:, 0] >= 0) & (ij[:, 1] >= 0)
        self.num_nodes = int(ij.shape[0])
        self.register_buffer("lam_nodes", torch.nonzero(on_lattice).squeeze(1), persistent=False)
        self.register_buffer("lam_rows", ij[on_lattice, 0].contiguous(), persistent=False)
        self.register_buffer("lam_cols", ij[on_lattice, 1].contiguous(), persistent=False)
        self.ny = int(ij[on_lattice, 0].max()) + 1
        self.nx = int(ij[on_lattice, 1].max()) + 1

        # one 1-D Gaussian kernel per channel (zero-padded to a common length, normalised to sum 1) and
        # the gain that restores unit variance after the separable 2-D smoothing of white noise:
        # std(smoothed) = sum(k^2) for a separable kernel k x k
        self.radius = max(int(math.ceil(truncate * v)) for v in self.sigmas)
        t = torch.arange(-self.radius, self.radius + 1, dtype=torch.float32)
        kernels, gains = [], []
        for v in self.sigmas:
            k = torch.exp(-0.5 * (t / v) ** 2)
            k[t.abs() > truncate * v] = 0.0
            k = k / k.sum()
            kernels += [k] * self.channels_per_scale
            gains += [1.0 / float((k**2).sum())] * self.channels_per_scale
        self.register_buffer("kernels", torch.stack(kernels), persistent=False)  # (channels, 2 radius + 1)
        self.register_buffer("gains", torch.tensor(gains, dtype=torch.float32), persistent=False)

        self.layer_factory = load_layer_kernels(layer_kernels)
        self.noise_mlp = MLP(
            self.noise_channels,
            noise_mlp_hidden_dim,
            self.noise_channels,
            layer_kernels=self.layer_factory,
            n_extra_layers=0,
            final_activation=False,
            layer_norm=True,
        )
        LOGGER.info(
            "decoder (lattice) noise: %d channels, sigmas %s cells, lattice %d x %d, %d of %d data nodes on it",
            self.noise_channels,
            self.sigmas,
            self.ny,
            self.nx,
            int(on_lattice.sum()),
            self.num_nodes,
        )

    def _smooth(self, noise: Tensor, channel_slice: slice) -> Tensor:
        """Separable Gaussian smoothing of (batch, channels, ny, nx) white noise, unit variance out."""
        k = self.kernels[channel_slice]
        groups = noise.shape[1]
        out = F.pad(noise, (self.radius, self.radius, 0, 0), mode="reflect")
        out = F.conv2d(out, k[:, None, None, :], groups=groups)
        out = F.pad(out, (0, 0, self.radius, self.radius), mode="reflect")
        out = F.conv2d(out, k[:, None, :, None], groups=groups)
        return out * self.gains[channel_slice][None, :, None, None]

    def forward(
        self,
        *,
        batch_size: int,
        ensemble_size: int,
        grid_shard_sizes: ShardSizes,
        model_comm_group: Optional[ProcessGroup] = None,
        device: Optional[torch.device] = None,
        noise_dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Draw the noise and return the conditioning, ((batch ensemble grid), channels).

        grid_shard_sizes: shard sizes of the data nodes along the grid dimension in the model
        communication group, or None when the data nodes are not sharded (then the full tensor is
        returned on every rank).
        """
        bse = batch_size * ensemble_size
        noise = torch.randn((bse, self.noise_channels, self.ny, self.nx), dtype=noise_dtype, device=device)
        noise.requires_grad = False

        # channel sharding: each rank keeps (and smooths) its own channels of its own draw
        channel_shard_sizes = get_shard_sizes(noise, 1, model_comm_group)
        noise = shard_tensor(noise, 1, channel_shard_sizes, model_comm_group)
        if model_is_distributed(model_comm_group):
            rank = torch.distributed.get_rank(group=model_comm_group)
            c0 = int(sum(channel_shard_sizes[:rank]))
        else:
            c0 = 0
        channel_slice = slice(c0, c0 + noise.shape[1])
        noise = self._smooth(noise, channel_slice)

        # lattice -> data nodes (zero off the lattice): (bse, nodes, channels)
        values = noise[:, :, self.lam_rows, self.lam_cols].transpose(1, 2)
        full = torch.zeros((bse, self.num_nodes, noise.shape[1]), dtype=noise_dtype, device=device)
        full[:, self.lam_nodes, :] = values
        full = einops.rearrange(full, "bse grid vars -> (bse grid) vars") * self.noise_std

        if model_is_distributed(model_comm_group):
            if grid_shard_sizes is not None:
                full = all_to_all_transpose(full, 0, grid_shard_sizes, -1, channel_shard_sizes, model_comm_group)
            else:
                full = gather_tensor(full, -1, channel_shard_sizes, model_comm_group)

        cond = checkpoint(self.noise_mlp, full, use_reentrant=False)
        LOGGER.debug("decoder noise cond.shape = %s, norm: %.9e", cond.shape, torch.linalg.norm(cond))
        return cond
