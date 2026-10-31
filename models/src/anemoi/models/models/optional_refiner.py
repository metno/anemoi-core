# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Optional data-grid refiner: message-passing blocks on data -> data edges after the decoder.

Ported from ecmwf/anemoi-core ``feat/optional-refiner-main`` (OptionalRefinerMixin) onto the alpaca branch, with two
additions:

* ``mode: residual`` (default here) keeps the trained decoder and its output head and adds the refiner as a side
  branch: ``out = decoder(x) + gate * (W_out * refiner(W_in * decoder(x) + W_noise * noise) + b_out)`` with the
  per-variable ``gate`` zero-initialised, so a checkpoint without a refiner is reproduced exactly at the fork step (``mode: replace`` is the upstream form: the
  decoder emits ``refiner_channels`` and the last refiner block emits the variables).
* optional noise channels per data node (``DataNodeNoise``), injected at the entrance of the branch with a free
  (unnormalised) learned amplitude.
* optional ``use_data_latent``: the decoder's data-node input (``x_data_latent``: normalised input state, forcings and
  node attributes, i.e. location, land-sea mask, orography) is also projected into the branch, so the refiner knows
  the local context and the domain (LAM / global) beyond what the output variables and the edge lengths tell it.

Config (``model.refiner``)::

    refiner:
      mode: residual                  # residual | replace
      refiner_channels: 256
      num_refiners: 2
      trainable_size: 0
      sub_graph_edge_attributes: [edge_length, edge_dirs]
      block:                          # one GraphTransformerRefinerMapper per refiner
        _target_: anemoi.models.layers.mapper.GraphTransformerRefinerMapper
        num_chunks: 4
        num_heads: 4
        attn_channels: 128
        mlp_hidden_ratio: 4
        layer_kernels: ...
      use_data_latent: true           # optional (residual mode): add W_lat * x_data_latent at the entrance
      noise:                          # optional
        noise_channels: 8
        smooth_channels: 4
        noise_std: 1.0

The graph needs ``(data, to, data)`` edges (e.g. KNNEdges 16).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from hydra.utils import instantiate
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.layers.ensemble import DataNodeNoise
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class OptionalRefinerMixin:
    """Shared helper for the optional data-grid refinement stage."""

    refiner_enabled: bool = False

    def _build_optional_refiner_networks(self, model_config: DotDict) -> None:
        cfg = model_config.model.get("refiner", None)
        self.refiner_enabled = cfg is not None
        if not self.refiner_enabled:
            return
        self.refiner_mode = str(cfg.get("mode", "residual"))
        assert self.refiner_mode in ("residual", "replace"), self.refiner_mode
        channels = int(cfg.refiner_channels)
        num_refiners = int(cfg.num_refiners)
        if num_refiners < 1:
            raise ValueError(f"num_refiners must be >= 1, got {num_refiners}")
        noise_cfg = cfg.get("noise", None)

        self.refiner_graph_provider = nn.ModuleDict()
        self.refiners = nn.ModuleDict()
        self.refiner_in = nn.ModuleDict()
        self.refiner_out = nn.ModuleDict()
        self.refiner_noise = nn.ModuleDict()
        self.refiner_noise_proj = nn.ModuleDict()
        self.refiner_latent_in = nn.ModuleDict()
        self.refiner_gate = nn.ParameterDict()
        self.refiner_use_data_latent = bool(cfg.get("use_data_latent", False)) and self.refiner_mode == "residual"

        for dataset_name in self.dataset_names:
            edge_key = (dataset_name, "to", dataset_name)
            if edge_key not in self._graph_data.edge_types:
                msg = (
                    f"Missing refiner edge type {edge_key}: add a data->data edge block to the graph "
                    "(e.g. anemoi.graphs.edges.KNNEdges)."
                )
                raise ValueError(msg)
            num_nodes = self.node_attributes.num_nodes[dataset_name]
            self.refiner_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[edge_key],
                edge_attributes=cfg.get("sub_graph_edge_attributes"),
                src_size=num_nodes,
                dst_size=num_nodes,
                trainable_size=cfg.get("trainable_size", 0),
            )
            edge_dim = self.refiner_graph_provider[dataset_name].edge_dim

            if self.refiner_mode == "replace":
                # upstream form: the decoder emits refiner_channels instead of the variables
                self.decoder[dataset_name] = instantiate(
                    model_config.model.decoder,
                    _recursive_=False,
                    in_channels_src=self.num_channels,
                    in_channels_dst=self.input_dim[dataset_name],
                    hidden_dim=self.num_channels,
                    out_channels_dst=channels,
                    edge_dim=self.decoder_graph_provider[dataset_name].edge_dim,
                )

            blocks = []
            for idx in range(num_refiners):
                last = idx == num_refiners - 1
                out_channels = self.output_dim[dataset_name] if (last and self.refiner_mode == "replace") else None
                blocks.append(
                    instantiate(
                        cfg.block,
                        _recursive_=False,
                        in_channels_src=channels,
                        in_channels_dst=channels,
                        hidden_dim=channels,
                        out_channels_dst=out_channels,
                        edge_dim=edge_dim,
                    )
                )
            self.refiners[dataset_name] = nn.ModuleList(blocks)

            if self.refiner_mode == "residual":
                self.refiner_in[dataset_name] = nn.Linear(self.output_dim[dataset_name], channels)
                # output projection with default init, gated per output variable by a zero-initialised vector:
                # out = x_out + gate * (W h + b). A zero MATRIX would be moved to +-lr on every entry by Adam's first
                # (sign-like) step and perturb the output by O(lr * |h| * sqrt(channels)); a zero vector gate keeps the
                # first-step perturbation at O(lr * |W h|) and ramps the branch in smoothly (ReZero-style).
                self.refiner_out[dataset_name] = nn.Linear(channels, self.output_dim[dataset_name])
                self.refiner_gate[dataset_name] = nn.Parameter(torch.zeros(self.output_dim[dataset_name]))
                if self.refiner_use_data_latent:
                    self.refiner_latent_in[dataset_name] = nn.Linear(self.input_dim[dataset_name], channels)

            if noise_cfg is not None:
                self.refiner_noise[dataset_name] = DataNodeNoise(
                    noise_channels=int(noise_cfg.noise_channels),
                    smooth_channels=int(noise_cfg.get("smooth_channels", 0)),
                    noise_std=float(noise_cfg.get("noise_std", 1.0)),
                    graph_data=self._graph_data,
                    edges_name=edge_key,
                    data_nodes_name=dataset_name,
                )
                self.refiner_noise_proj[dataset_name] = nn.Linear(int(noise_cfg.noise_channels), channels)

        LOGGER.info(
            "refiner: mode %s, %d block(s) x %d channels on %s edges, noise %s",
            self.refiner_mode,
            num_refiners,
            channels,
            [str(k) for k in self.refiner_graph_provider.keys()],
            "off" if noise_cfg is None else dict(noise_cfg),
        )
        LOGGER.info("refiner: data-latent input %s", "on" if self.refiner_use_data_latent else "off")

    def _apply_optional_refiner(
        self,
        *,
        dataset_name: str,
        x_out: torch.Tensor,
        x_data_latent: Optional[torch.Tensor] = None,
        batch_size: int,
        ensemble_size: int,
        batch_ens_size: int,
        shard_sizes_data: ShardSizes,
        in_out_sharded: bool,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> torch.Tensor:
        """Run the refiner on the decoder output ``x_out`` ((batch ensemble grid), channels), sharded iff
        ``in_out_sharded`` (``shard_sizes_data`` None otherwise)."""
        if not getattr(self, "refiner_enabled", False):
            return x_out

        h = self.refiner_in[dataset_name](x_out) if self.refiner_mode == "residual" else x_out
        if getattr(self, "refiner_use_data_latent", False):
            assert x_data_latent is not None, "use_data_latent needs x_data_latent"
            h = h + self.refiner_latent_in[dataset_name](x_data_latent.to(dtype=h.dtype))
        if dataset_name in self.refiner_noise:
            noise = self.refiner_noise[dataset_name](
                batch_size=batch_size,
                ensemble_size=ensemble_size,
                grid_shard_sizes=shard_sizes_data if in_out_sharded else None,
                model_comm_group=model_comm_group,
                device=x_out.device,
            )
            h = h + self.refiner_noise_proj[dataset_name](noise.to(dtype=h.dtype))

        edge_attr, edge_index, edge_shard_sizes = self.refiner_graph_provider[dataset_name].get_edges(
            batch_size=batch_ens_size,
            model_comm_group=model_comm_group,
        )
        shard_info = BipartiteGraphShardInfo(
            src_nodes=shard_sizes_data,  # None if the data nodes are not sharded
            dst_nodes=shard_sizes_data,
            edges=edge_shard_sizes,
        )
        for block in self.refiners[dataset_name]:
            h = block(
                (h, h),
                batch_size=batch_ens_size,
                shard_info=shard_info,
                edge_attr=edge_attr,
                edge_index=edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=in_out_sharded,
            )
        if self.refiner_mode == "residual":
            delta = self.refiner_out[dataset_name](h) * self.refiner_gate[dataset_name].to(dtype=h.dtype)
            return x_out + delta.to(dtype=x_out.dtype)
        return h
