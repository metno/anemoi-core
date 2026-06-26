# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import ast
import logging
from typing import Optional

import einops
import numpy as np
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.preprocessing.qc_flags import QCDecodeBits
from anemoi.models.preprocessing.qc_flags import QCFeaturizer
from anemoi.models.preprocessing.qc_flags import QCPackedEmbedding
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDecUnionDecoderForecaster(AnemoiModelEncProcDec):
    """Forecaster variant with provider-grouped decoding (e.g. union mesh).

    This keeps the standard multi-output forecaster behavior but decodes once per
    provider mesh and maps decoded outputs back to dataset meshes using provider
    slices (``_slices``) plus coordinate remapping when needed.
    """

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
    ) -> None:
        model_config = DotDict(model_config)

        qc_cfg = getattr(model_config.data, "qc", None)
        self._qc_enabled = qc_cfg is not None
        self.qc: dict[str, QCFeaturizer] = {}
        self.qc_var: dict[str, str] = {}
        if self._qc_enabled:
            for ds_name, cfg in qc_cfg.items():
                cfg = DotDict(cfg)
                self.qc_var[str(ds_name)] = str(cfg.get("qc_var", "qc_flags"))
                self.qc[str(ds_name)] = QCFeaturizer(method=cfg.get("method", "decode_bits"), qc_cfg=cfg)

        station_dropout_cfg = DotDict(model_config.model.get("station_dropout", {}) or {})
        self.station_dropout_dataset = str(station_dropout_cfg.get("dataset", "netatmo"))
        self.station_dropout_rate = float(station_dropout_cfg.get("rate", 0.0))
        self.station_dropout_fill_value = float(station_dropout_cfg.get("fill_value", 0.0))
        self.station_dropout_variables = tuple(str(var) for var in station_dropout_cfg.get("variables", []))

        datasets = list(data_indices.keys())
        raw_dataset_time_indices = model_config.training.get("dataset_time_indices", None)
        dataset_time_indices_cfg = raw_dataset_time_indices
        if hasattr(raw_dataset_time_indices, "get"):
            dataset_time_indices_cfg = raw_dataset_time_indices.get("datasets", raw_dataset_time_indices)
        self.dataset_input_time_indices: dict[str, tuple[int, ...]] = {}
        if dataset_time_indices_cfg is not None:
            if not hasattr(dataset_time_indices_cfg, "items"):
                raise ValueError(
                    "`training.dataset_time_indices` must be a mapping, optionally under key `datasets`."
                )
            unknown_dataset_keys = sorted(set(dataset_time_indices_cfg).difference(set(datasets)))
            if unknown_dataset_keys:
                raise ValueError(
                    f"`training.dataset_time_indices` provided for unknown datasets: {unknown_dataset_keys}. "
                    f"Known datasets: {datasets}"
                )
            for dataset_name, dataset_cfg in dataset_time_indices_cfg.items():
                if not hasattr(dataset_cfg, "get"):
                    raise ValueError(
                        f"`training.dataset_time_indices[{dataset_name}]` must define `input` and `target` lists."
                    )
                raw_input = dataset_cfg.get("input", None)
                if raw_input is None:
                    raise ValueError(
                        f"`training.dataset_time_indices[{dataset_name}]` must define an `input` list."
                    )
                input_indices = tuple(int(value) for value in raw_input)
                self.dataset_input_time_indices[str(dataset_name)] = input_indices

        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
        )
        self.qc = nn.ModuleDict(self.qc)

    def _append_qc_features(self, x: torch.Tensor, *, dataset_name: str) -> torch.Tensor:
        if not self._qc_enabled or dataset_name not in self.qc:
            return x

        name_to_index = getattr(self.data_indices[dataset_name].model.input, "name_to_index", None)
        if name_to_index is None or self.qc_var[dataset_name] not in name_to_index:
            raise KeyError(
                f"QC enabled but '{self.qc_var[dataset_name]}' not found in model input variables for dataset '{dataset_name}'."
            )

        qc_idx = int(name_to_index[self.qc_var[dataset_name]])
        qc_flags = x[..., qc_idx]
        feats = self.qc[dataset_name](qc_flags)
        x_noqc = torch.cat([x[..., :qc_idx], x[..., qc_idx + 1 :]], dim=-1)
        return torch.cat([x_noqc, feats], dim=-1)

    def _calculate_input_dim(self, dataset_name: str) -> int:
        dataset_n_step_input = len(self.dataset_input_time_indices.get(dataset_name, tuple(range(self.n_step_input))))
        base = dataset_n_step_input * self.num_input_channels[dataset_name]
        qc_extra = 0
        if self._qc_enabled and dataset_name in self.qc:
            featurizer = self.qc[dataset_name]
            if hasattr(featurizer, "mask"):
                qc_extra += 1
            if isinstance(featurizer.feat, QCDecodeBits):
                qc_extra += int(len(featurizer.feat.bits))
            elif isinstance(featurizer.feat, QCPackedEmbedding):
                qc_extra += int(featurizer.feat.emb_dim)
            # qc_flags itself is removed from numeric channels when qc is enabled.
            base -= dataset_n_step_input

        node_attr = self.node_attributes.attr_ndims[dataset_name]
        return base + dataset_n_step_input * qc_extra + node_attr

    def _build_networks(self, model_config: DotDict) -> None:
        self.encoder_dataset_names = tuple(
            dataset_name for dataset_name in self.dataset_names if self.num_input_channels[dataset_name] > 0
        )

        self.encoder_graph_provider = nn.ModuleDict()
        self.encoder = nn.ModuleDict()
        for dataset_name in self.encoder_dataset_names:
            self.encoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[(dataset_name, "to", self._graph_name_hidden)],
                edge_attributes=model_config.model.encoder.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes.num_nodes[dataset_name],
                dst_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                trainable_size=model_config.model.encoder.get("trainable_size", 0),
            )

            self.encoder[dataset_name] = instantiate(
                model_config.model.encoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.input_dim[dataset_name],
                in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
                hidden_dim=self.num_channels,
                edge_dim=self.encoder_graph_provider[dataset_name].edge_dim,
            )

        self.processor_graph_provider = create_graph_provider(
            graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            edge_attributes=model_config.model.processor.get("sub_graph_edge_attributes"),
            src_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            trainable_size=model_config.model.processor.get("trainable_size", 0),
        )

        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            num_channels=self.num_channels,
            edge_dim=self.processor_graph_provider.edge_dim,
        )

        decoder_sel_cfg = DotDict(model_config.model.get("decoder_graph_selection", {}))
        decoder_provider_cfg = DotDict(decoder_sel_cfg.get("provider_datasets", {}))
        decoder_target_datasets = decoder_sel_cfg.get("target_datasets", None)
        if decoder_target_datasets is None:
            self.decoder_target_dataset_names = tuple(self.dataset_names)
        else:
            self.decoder_target_dataset_names = tuple(str(dataset_name) for dataset_name in decoder_target_datasets)
            unknown_dataset_names = sorted(set(self.decoder_target_dataset_names).difference(self.dataset_names))
            if unknown_dataset_names:
                raise ValueError(
                    f"decoder_graph_selection.target_datasets contains unknown datasets: {unknown_dataset_names}. "
                    f"Known datasets: {self.dataset_names}"
                )

        self.decoder_provider_by_dataset = {
            dataset_name: str(decoder_provider_cfg.get(dataset_name, dataset_name))
            for dataset_name in self.decoder_target_dataset_names
        }
        self.decoder_graph_provider = torch.nn.ModuleDict()
        self.decoder = torch.nn.ModuleDict()
        self.decoder_template_dataset_by_provider: dict[str, str] = {}
        for dataset_name in self.decoder_target_dataset_names:
            provider_name = self.decoder_provider_by_dataset[dataset_name]
            if provider_name not in self.node_attributes.num_nodes:
                raise KeyError(
                    f"Decoder provider '{provider_name}' for dataset '{dataset_name}' is not a graph node type. "
                    f"Available node types: {list(self.node_attributes.num_nodes.keys())}"
                )
            provider_input_dim = (
                self.input_dim[provider_name]
                if provider_name in self.input_dim
                else self.node_attributes.attr_ndims[provider_name]
            )
            if provider_name in self.decoder:
                template_dataset = self.decoder_template_dataset_by_provider[provider_name]
                if int(self.output_dim[dataset_name]) != int(self.output_dim[template_dataset]):
                    raise ValueError(
                        "Datasets mapped to the same decoder provider must share output dimensions: "
                        f"{dataset_name}={self.output_dim[dataset_name]}, "
                        f"{template_dataset}={self.output_dim[template_dataset]}, "
                        f"provider={provider_name}"
                    )
                continue

            self.decoder_graph_provider[provider_name] = create_graph_provider(
                graph=self._graph_data[(self._graph_name_hidden, "to", provider_name)],
                edge_attributes=model_config.model.decoder.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                dst_size=self.node_attributes.num_nodes[provider_name],
                trainable_size=model_config.model.decoder.get("trainable_size", 0),
            )

            self.decoder[provider_name] = instantiate(
                model_config.model.decoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.num_channels,
                in_channels_dst=provider_input_dim,
                hidden_dim=self.num_channels,
                out_channels_dst=self.output_dim[dataset_name],
                edge_dim=self.decoder_graph_provider[provider_name].edge_dim,
            )
            self.decoder_template_dataset_by_provider[provider_name] = dataset_name

    def _apply_station_dropout(self, x: torch.Tensor, *, dataset_name: str) -> torch.Tensor:
        if (
            not self.training
            or self.station_dropout_rate <= 0.0
            or dataset_name != self.station_dropout_dataset
            or x.shape[3] == 0
        ):
            return x

        if self.station_dropout_variables:
            name_to_index = getattr(self.data_indices[dataset_name].model.input, "name_to_index", None)
            if name_to_index is None:
                raise KeyError(f"Cannot resolve station dropout variables for dataset {dataset_name!r}.")
            variable_indices = [int(name_to_index[var]) for var in self.station_dropout_variables]
        else:
            variable_indices = [int(idx) for idx in self.data_indices[dataset_name].model.input.prognostic]

        if len(variable_indices) == 0:
            return x

        dropout_rate = min(max(self.station_dropout_rate, 0.0), 1.0)
        keep_mask = torch.rand(x.shape[0], x.shape[3], device=x.device) >= dropout_rate
        if not bool(keep_mask.any(dim=1).all()):
            empty_batches = torch.nonzero(~keep_mask.any(dim=1), as_tuple=False).squeeze(-1)
            keep_index = torch.randint(x.shape[3], (empty_batches.numel(),), device=x.device)
            keep_mask[empty_batches, keep_index] = True

        drop_mask = ~keep_mask[:, None, None, :, None]
        x_dropped = x.clone()
        variable_index = torch.tensor(variable_indices, device=x.device, dtype=torch.long)
        values = x_dropped.index_select(-1, variable_index)
        fill_value = torch.as_tensor(self.station_dropout_fill_value, device=x.device, dtype=x.dtype)
        values = torch.where(drop_mask, fill_value, values)
        x_dropped[..., variable_index] = values
        return x_dropped

    def _get_provider_decode_index(
        self,
        *,
        provider_name: str,
        dataset_name: str,
        start: int,
        end: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Map provider slice indices to the target dataset ordering."""

        cpu_cache = getattr(self, "_provider_decode_index_cache_cpu", None)
        if cpu_cache is None:
            cpu_cache = {}
            self._provider_decode_index_cache_cpu = cpu_cache

        key = (provider_name, dataset_name, int(start), int(end))
        remap_idx_cpu = cpu_cache.get(key, None)

        if key not in cpu_cache:
            provider_coords = self._graph_data[provider_name].x[start:end].detach().cpu().contiguous().numpy()
            dataset_coords = self._graph_data[dataset_name].x.detach().cpu().contiguous().numpy()

            if provider_coords.shape[0] == dataset_coords.shape[0]:
                cpu_cache[key] = None
                return None

            if provider_coords.shape[1:] != dataset_coords.shape[1:]:
                raise ValueError(
                    f"Cannot remap provider '{provider_name}' slice to dataset '{dataset_name}': "
                    f"coordinate dimensions differ ({provider_coords.shape[1:]} vs {dataset_coords.shape[1:]})."
                )

            provider_view = np.ascontiguousarray(provider_coords).view(
                np.dtype((np.void, provider_coords.dtype.itemsize * provider_coords.shape[1]))
            )
            dataset_view = np.ascontiguousarray(dataset_coords).view(
                np.dtype((np.void, dataset_coords.dtype.itemsize * dataset_coords.shape[1]))
            )
            provider_view = provider_view.reshape(-1)
            dataset_view = dataset_view.reshape(-1)

            order = np.argsort(provider_view, kind="mergesort")
            sorted_provider = provider_view[order]
            positions = np.searchsorted(sorted_provider, dataset_view, side="left")

            valid = positions < sorted_provider.shape[0]
            if valid.any():
                valid_pos = positions[valid]
                valid[valid] = sorted_provider[valid_pos] == dataset_view[valid]

            if not bool(valid.all()):
                missing = int((~valid).sum())
                raise ValueError(
                    f"Cannot remap provider '{provider_name}' slice to dataset '{dataset_name}': "
                    f"{missing} dataset coordinates were not found in provider slice."
                )

            mapped = order[positions].astype(np.int64, copy=False)
            if np.unique(mapped).shape[0] != dataset_view.shape[0]:
                raise ValueError(
                    f"Cannot remap provider '{provider_name}' slice to dataset '{dataset_name}': "
                    "mapping is not one-to-one."
                )

            remap_idx_cpu = torch.from_numpy(mapped)
            cpu_cache[key] = remap_idx_cpu

        if remap_idx_cpu is None:
            return None

        device_cache = getattr(self, "_provider_decode_index_cache_device", None)
        if device_cache is None:
            device_cache = {}
            self._provider_decode_index_cache_device = device_cache

        device_key = (*key, str(device))
        if device_key not in device_cache:
            device_cache[device_key] = remap_idx_cpu.to(device=device, dtype=torch.long)
        return device_cache[device_key]

    def forward(
        self,
        x: dict[str, Tensor],
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: dict[str, list] | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass with provider-grouped decode routing."""
        del kwargs
        dataset_names = list(x.keys())

        if grid_shard_shapes is None:
            grid_shard_shapes = {dataset_name: None for dataset_name in dataset_names}

        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        in_out_sharded = {}
        for dataset_name, shard_shapes in grid_shard_shapes.items():
            in_out_sharded[dataset_name] = shard_shapes is not None
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_shapes_data_dict = {}
        shard_shapes_hidden_dict = {}

        for dataset_name in dataset_names:
            x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
            shard_shapes_hidden_dict[dataset_name] = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

            x_in = x[dataset_name]
            if x_in.is_floating_point() and x_hidden_latent.is_floating_point() and x_in.dtype != x_hidden_latent.dtype:
                x_in = x_in.to(dtype=x_hidden_latent.dtype)
            x_in = self._apply_station_dropout(x_in, dataset_name=dataset_name)
            x_aug = self._append_qc_features(x_in, dataset_name=dataset_name)

            x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
                x_aug,
                batch_size=batch_size,
                grid_shard_shapes=grid_shard_shapes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            x_skip_dict[dataset_name] = x_skip
            shard_shapes_data_dict[dataset_name] = shard_shapes_data

            if dataset_name in self.encoder:
                encoder_edge_attr, encoder_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[
                    dataset_name
                ].get_edges(
                    batch_size=batch_size,
                    model_comm_group=model_comm_group,
                )

                x_data_latent, x_latent = self.encoder[dataset_name](
                    (x_data_latent, x_hidden_latent),
                    batch_size=batch_size,
                    shard_shapes=(shard_shapes_data, shard_shapes_hidden_dict[dataset_name]),
                    edge_attr=encoder_edge_attr,
                    edge_index=encoder_edge_index,
                    model_comm_group=model_comm_group,
                    x_src_is_sharded=in_out_sharded[dataset_name],
                    x_dst_is_sharded=False,
                    keep_x_dst_sharded=True,
                    edge_shard_shapes=enc_edge_shard_shapes,
                )
                dataset_latents[dataset_name] = x_latent

            x_data_latent_dict[dataset_name] = x_data_latent

        x_latent = sum(dataset_latents.values())

        shard_shapes_hidden = shard_shapes_hidden_dict[dataset_names[0]]
        assert all(
            shard_shape == shard_shapes_hidden for shard_shape in shard_shapes_hidden_dict.values()
        ), "All datasets must have the same shard shapes for the hidden graph."

        processor_edge_attr, processor_edge_index, proc_edge_shard_shapes = self.processor_graph_provider.get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x=x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
            edge_shard_shapes=proc_edge_shard_shapes,
        )
        x_latent_proc = x_latent_proc + x_latent

        provider_by_dataset = getattr(self, "decoder_provider_by_dataset", {})
        decode_groups: dict[str, list[str]] = {}
        for dataset_name in dataset_names:
            if dataset_name not in provider_by_dataset:
                continue
            provider_name = str(provider_by_dataset[dataset_name])
            decode_groups.setdefault(provider_name, []).append(dataset_name)

        provider_x_dst_base: dict[str, torch.Tensor] = {}
        provider_shard_shapes: dict[str, list] = {}
        provider_dst_sharded: dict[str, bool] = {}
        for provider_name in decode_groups:
            if provider_name in x_data_latent_dict:
                provider_x_dst_base[provider_name] = x_data_latent_dict[provider_name]
                provider_shard_shapes[provider_name] = shard_shapes_data_dict[provider_name]
                provider_dst_sharded[provider_name] = in_out_sharded[provider_name]
                continue

            x_dst_provider = self.node_attributes(provider_name, batch_size=batch_size)
            if x_dst_provider.device != x_latent_proc.device:
                x_dst_provider = x_dst_provider.to(x_latent_proc.device)
            if (
                x_dst_provider.is_floating_point()
                and x_latent_proc.is_floating_point()
                and x_dst_provider.dtype != x_latent_proc.dtype
            ):
                x_dst_provider = x_dst_provider.to(dtype=x_latent_proc.dtype)

            provider_x_dst_base[provider_name] = x_dst_provider
            provider_shard_shapes[provider_name] = get_shard_shapes(x_dst_provider, 0, model_comm_group)
            provider_dst_sharded[provider_name] = False

        provider_slices_cache: dict[str, dict[str, tuple[int, int]] | None] = {}

        def _get_provider_slices(provider_name: str) -> dict[str, tuple[int, int]] | None:
            if provider_name in provider_slices_cache:
                return provider_slices_cache[provider_name]

            if provider_name not in self._graph_data.node_types:
                provider_slices_cache[provider_name] = None
                return None

            node_store = self._graph_data[provider_name]
            slices_value = node_store.get("_slices", None)
            if slices_value is None:
                slices_value = getattr(node_store, "_slices", None)
            if slices_value is None:
                provider_slices_cache[provider_name] = None
                return None

            if isinstance(slices_value, str):
                parsed = ast.literal_eval(slices_value)
            else:
                parsed = slices_value

            provider_slices_cache[provider_name] = {
                str(key): (int(value[0]), int(value[1])) for key, value in dict(parsed).items()
            }
            return provider_slices_cache[provider_name]

        y_pred: dict[str, torch.Tensor] = {}
        for provider_name, provider_targets in decode_groups.items():
            exemplar_dataset = provider_targets[0]
            exemplar_output_dim = int(self.output_dim[exemplar_dataset])
            for dataset_name in provider_targets[1:]:
                if int(self.output_dim[dataset_name]) != exemplar_output_dim:
                    raise ValueError(
                        "Datasets mapped to the same decoder provider must have equal output dimensions: "
                        f"{exemplar_dataset}={self.output_dim[exemplar_dataset]}, "
                        f"{dataset_name}={self.output_dim[dataset_name]}"
                    )

            dec_edge_attr, dec_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider[provider_name].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )

            x_out_provider = self.decoder[provider_name](
                (x_latent_proc, provider_x_dst_base[provider_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, provider_shard_shapes[provider_name]),
                edge_attr=dec_edge_attr,
                edge_index=dec_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,
                x_dst_is_sharded=provider_dst_sharded[provider_name],
                keep_x_dst_sharded=provider_dst_sharded[provider_name],
                edge_shard_shapes=dec_edge_shard_shapes,
            )

            provider_concat = (
                einops.rearrange(
                    x_out_provider,
                    "(batch ensemble grid) (time vars) -> batch time ensemble grid vars",
                    batch=batch_size,
                    ensemble=ensemble_size,
                    time=self.n_step_output,
                )
                .to(x[exemplar_dataset].dtype)
                .clone()
            )

            provider_slices = _get_provider_slices(provider_name)

            for dataset_name in provider_targets:
                if provider_name == dataset_name:
                    concat = provider_concat.clone()
                else:
                    if provider_slices is None or dataset_name not in provider_slices:
                        raise KeyError(
                            f"Provider '{provider_name}' has no slice for dataset '{dataset_name}'. "
                            "For union decoding, provider nodes must expose '_slices' with dataset ranges."
                        )

                    start, end = provider_slices[dataset_name]
                    concat = provider_concat[..., start:end, :].clone()
                    if concat.shape[3] != x_skip_dict[dataset_name].shape[3]:
                        remap_idx = self._get_provider_decode_index(
                            provider_name=provider_name,
                            dataset_name=dataset_name,
                            start=start,
                            end=end,
                            device=concat.device,
                        )
                        if remap_idx is not None:
                            concat = torch.index_select(concat, 3, remap_idx)

                if concat.shape[3] != x_skip_dict[dataset_name].shape[3]:
                    raise ValueError(
                        f"Decoded grid size for '{dataset_name}' ({concat.shape[3]}) does not match "
                        f"residual input grid size ({x_skip_dict[dataset_name].shape[3]}). "
                        f"Provider='{provider_name}'."
                    )

                concat[..., self._internal_output_idx[dataset_name]] += x_skip_dict[dataset_name][
                    ..., self._internal_input_idx[dataset_name]
                ]

                for bounding in self.boundings[dataset_name]:
                    concat = bounding(concat)

                y_pred[dataset_name] = concat

        return y_pred


class AnemoiEnsModelEncProcDecUnionDecoderForecaster(AnemoiModelEncProcDecUnionDecoderForecaster):
    """Sparse union forecaster with ensemble noise injected into the summed latent state."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
    ) -> None:
        self.condition_on_residual = DotDict(model_config).model.condition_on_residual
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
        )

    def _build_networks(self, model_config: DotDict) -> None:
        super()._build_networks(model_config)
        model_cfg = DotDict(model_config).model
        self.noise_injector = instantiate(
            model_cfg.noise_injector,
            _recursive_=False,
            num_channels=self.num_channels,
        )

    def _calculate_input_dim(self, dataset_name: str) -> int:
        input_dim = super()._calculate_input_dim(dataset_name)
        input_dim += 1
        if self.condition_on_residual:
            input_dim += self.num_input_channels_prognostic[dataset_name]
        return input_dim

    def _assemble_input(
        self,
        x: torch.Tensor,
        *,
        fcstep: int,
        batch_size: int,
        grid_shard_shapes: dict | None,
        model_comm_group=None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list]]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes(dataset_name, batch_size=batch_size)
        grid_shard_shapes = grid_shard_shapes[dataset_name] if grid_shard_shapes is not None else None

        x_skip = self.residual[dataset_name](
            x,
            grid_shard_shapes=grid_shard_shapes,
            model_comm_group=model_comm_group,
            n_step_output=self.n_step_output,
        )

        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data,
                0,
                shard_shapes_dim=grid_shard_shapes,
                model_comm_group=model_comm_group,
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
                torch.ones(batch_size * x.shape[3], device=x.device).unsqueeze(-1) * fcstep,
            ),
            dim=-1,
        )

        if self.condition_on_residual:
            x_skip_cond = x_skip[:, 0] if x_skip.ndim == 5 else x_skip
            prognostic_idx = self._internal_input_idx.get(dataset_name, None)
            if prognostic_idx is not None and len(prognostic_idx) > 0:
                x_skip_cond = x_skip_cond[..., prognostic_idx]
                x_data_latent = torch.cat(
                    (
                        x_data_latent,
                        einops.rearrange(x_skip_cond, "batch ensemble grid vars -> (batch ensemble grid) vars"),
                    ),
                    dim=-1,
                )

        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent,
            0,
            shard_shapes_dim=grid_shard_shapes,
            model_comm_group=model_comm_group,
        )

        return x_data_latent, x_skip, shard_shapes_data

    def forward(
        self,
        x: dict[str, Tensor],
        *,
        fcstep: int = 0,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: dict[str, list] | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass with ensemble noise before hidden processing."""
        del kwargs
        dataset_names = list(x.keys())

        if grid_shard_shapes is None:
            grid_shard_shapes = {dataset_name: None for dataset_name in dataset_names}

        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        in_out_sharded = {}
        for dataset_name, shard_shapes in grid_shard_shapes.items():
            in_out_sharded[dataset_name] = shard_shapes is not None
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        fcstep = min(1, int(fcstep))
        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_shapes_data_dict = {}
        shard_shapes_hidden_dict = {}

        for dataset_name in dataset_names:
            x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
            shard_shapes_hidden_dict[dataset_name] = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

            x_in = x[dataset_name]
            if x_in.is_floating_point() and x_hidden_latent.is_floating_point() and x_in.dtype != x_hidden_latent.dtype:
                x_in = x_in.to(dtype=x_hidden_latent.dtype)
            x_in = self._apply_station_dropout(x_in, dataset_name=dataset_name)
            x_aug = self._append_qc_features(x_in, dataset_name=dataset_name)

            x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
                x_aug,
                fcstep=fcstep,
                batch_size=batch_size,
                grid_shard_shapes=grid_shard_shapes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            x_skip_dict[dataset_name] = x_skip
            shard_shapes_data_dict[dataset_name] = shard_shapes_data

            if dataset_name in self.encoder:
                encoder_edge_attr, encoder_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[
                    dataset_name
                ].get_edges(
                    batch_size=batch_size,
                    model_comm_group=model_comm_group,
                )

                x_data_latent, x_latent = self.encoder[dataset_name](
                    (x_data_latent, x_hidden_latent),
                    batch_size=batch_size,
                    shard_shapes=(shard_shapes_data, shard_shapes_hidden_dict[dataset_name]),
                    edge_attr=encoder_edge_attr,
                    edge_index=encoder_edge_index,
                    model_comm_group=model_comm_group,
                    x_src_is_sharded=in_out_sharded[dataset_name],
                    x_dst_is_sharded=False,
                    keep_x_dst_sharded=True,
                    edge_shard_shapes=enc_edge_shard_shapes,
                )
                dataset_latents[dataset_name] = x_latent

            x_data_latent_dict[dataset_name] = x_data_latent

        shard_shapes_hidden = shard_shapes_hidden_dict[dataset_names[0]]
        assert all(
            shard_shape == shard_shapes_hidden for shard_shape in shard_shapes_hidden_dict.values()
        ), "All datasets must have the same shard shapes for the hidden graph."

        x_latent = sum(dataset_latents.values())
        x_latent_proc, latent_noise = self.noise_injector(
            x=x_latent,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            shard_shapes_ref=shard_shapes_hidden,
            noise_dtype=x_latent.dtype,
            model_comm_group=model_comm_group,
        )

        processor_edge_attr, processor_edge_index, proc_edge_shard_shapes = self.processor_graph_provider.get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )
        processor_kwargs = {"cond": latent_noise} if latent_noise is not None else {}

        x_latent_proc = self.processor(
            x=x_latent_proc,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
            edge_shard_shapes=proc_edge_shard_shapes,
            **processor_kwargs,
        )
        x_latent_proc = x_latent_proc + x_latent

        provider_by_dataset = getattr(self, "decoder_provider_by_dataset", {})
        decode_groups: dict[str, list[str]] = {}
        for dataset_name in dataset_names:
            if dataset_name not in provider_by_dataset:
                continue
            provider_name = str(provider_by_dataset[dataset_name])
            decode_groups.setdefault(provider_name, []).append(dataset_name)

        provider_x_dst_base: dict[str, torch.Tensor] = {}
        provider_shard_shapes: dict[str, list] = {}
        provider_dst_sharded: dict[str, bool] = {}
        for provider_name in decode_groups:
            if provider_name in x_data_latent_dict:
                provider_x_dst_base[provider_name] = x_data_latent_dict[provider_name]
                provider_shard_shapes[provider_name] = shard_shapes_data_dict[provider_name]
                provider_dst_sharded[provider_name] = in_out_sharded[provider_name]
                continue

            x_dst_provider = self.node_attributes(provider_name, batch_size=batch_size)
            if x_dst_provider.device != x_latent_proc.device:
                x_dst_provider = x_dst_provider.to(x_latent_proc.device)
            if (
                x_dst_provider.is_floating_point()
                and x_latent_proc.is_floating_point()
                and x_dst_provider.dtype != x_latent_proc.dtype
            ):
                x_dst_provider = x_dst_provider.to(dtype=x_latent_proc.dtype)

            provider_x_dst_base[provider_name] = x_dst_provider
            provider_shard_shapes[provider_name] = get_shard_shapes(x_dst_provider, 0, model_comm_group)
            provider_dst_sharded[provider_name] = False

        provider_slices_cache: dict[str, dict[str, tuple[int, int]] | None] = {}

        def _get_provider_slices(provider_name: str) -> dict[str, tuple[int, int]] | None:
            if provider_name in provider_slices_cache:
                return provider_slices_cache[provider_name]

            if provider_name not in self._graph_data.node_types:
                provider_slices_cache[provider_name] = None
                return None

            node_store = self._graph_data[provider_name]
            slices_value = node_store.get("_slices", None)
            if slices_value is None:
                slices_value = getattr(node_store, "_slices", None)
            if slices_value is None:
                provider_slices_cache[provider_name] = None
                return None

            if isinstance(slices_value, str):
                parsed = ast.literal_eval(slices_value)
            else:
                parsed = slices_value

            provider_slices_cache[provider_name] = {
                str(key): (int(value[0]), int(value[1])) for key, value in dict(parsed).items()
            }
            return provider_slices_cache[provider_name]

        y_pred: dict[str, torch.Tensor] = {}
        for provider_name, provider_targets in decode_groups.items():
            exemplar_dataset = provider_targets[0]
            exemplar_output_dim = int(self.output_dim[exemplar_dataset])
            for dataset_name in provider_targets[1:]:
                if int(self.output_dim[dataset_name]) != exemplar_output_dim:
                    raise ValueError(
                        "Datasets mapped to the same decoder provider must have equal output dimensions: "
                        f"{exemplar_dataset}={self.output_dim[exemplar_dataset]}, "
                        f"{dataset_name}={self.output_dim[dataset_name]}"
                    )

            dec_edge_attr, dec_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider[provider_name].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )

            x_out_provider = self.decoder[provider_name](
                (x_latent_proc, provider_x_dst_base[provider_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, provider_shard_shapes[provider_name]),
                edge_attr=dec_edge_attr,
                edge_index=dec_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,
                x_dst_is_sharded=provider_dst_sharded[provider_name],
                keep_x_dst_sharded=provider_dst_sharded[provider_name],
                edge_shard_shapes=dec_edge_shard_shapes,
            )

            provider_concat = (
                einops.rearrange(
                    x_out_provider,
                    "(batch ensemble grid) (time vars) -> batch time ensemble grid vars",
                    batch=batch_size,
                    ensemble=ensemble_size,
                    time=self.n_step_output,
                )
                .to(x[exemplar_dataset].dtype)
                .clone()
            )

            provider_slices = _get_provider_slices(provider_name)

            for dataset_name in provider_targets:
                if provider_name == dataset_name:
                    concat = provider_concat.clone()
                else:
                    if provider_slices is None or dataset_name not in provider_slices:
                        raise KeyError(
                            f"Provider '{provider_name}' has no slice for dataset '{dataset_name}'. "
                            "For union decoding, provider nodes must expose '_slices' with dataset ranges."
                        )

                    start, end = provider_slices[dataset_name]
                    concat = provider_concat[..., start:end, :].clone()
                    if concat.shape[3] != x_skip_dict[dataset_name].shape[3]:
                        remap_idx = self._get_provider_decode_index(
                            provider_name=provider_name,
                            dataset_name=dataset_name,
                            start=start,
                            end=end,
                            device=concat.device,
                        )
                        if remap_idx is not None:
                            concat = torch.index_select(concat, 3, remap_idx)

                if concat.shape[3] != x_skip_dict[dataset_name].shape[3]:
                    raise ValueError(
                        f"Decoded grid size for '{dataset_name}' ({concat.shape[3]}) does not match "
                        f"residual input grid size ({x_skip_dict[dataset_name].shape[3]}). "
                        f"Provider='{provider_name}'."
                    )

                concat[..., self._internal_output_idx[dataset_name]] += x_skip_dict[dataset_name][
                    ..., self._internal_input_idx[dataset_name]
                ]

                for bounding in self.boundings[dataset_name]:
                    concat = bounding(concat)

                y_pred[dataset_name] = concat

        return y_pred
