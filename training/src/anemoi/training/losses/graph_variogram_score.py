# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools

import torch
import torch.utils.checkpoint
from torch_geometric.data import HeteroData

from anemoi.training.losses.graph_edge_score_base import BaseGraphEdgeScoreLoss
from anemoi.training.losses.graph_score_graph import GraphScoreGraph


class GraphVariogramScoreLoss(BaseGraphEdgeScoreLoss):
    """Variogram score over node pairs stored by a CSR graph."""

    def __init__(
        self,
        loss_graph: dict,
        graph_data: HeteroData,
        p: float = 1.0,
        fair: bool = True,
        no_autocast: bool = True,
        ignore_nans: bool = False,
        edge_chunks: int = 1,
        checkpoint_chunks: bool = True,
    ) -> None:
        assert p > 0.0, "p must be strictly positive."
        assert edge_chunks >= 1, "edge_chunks must be a positive integer."
        graph = GraphScoreGraph.from_definition(
            loss_graph,
            graph_data,
            graph_name="Graph variogram neighbourhood",
            allow_none=False,
            remove_self_edges=True,
        )
        assert graph is not None
        super().__init__(graph=graph, no_autocast=no_autocast, ignore_nans=ignore_nans)
        self.p = p
        self.fair = fair
        # Memory: the edge tensors have shape (..., num_edges, num_vars). With ~2e7 edges (KNN-9 on a 2 M-node
        # grid) and ~50 variables per shard that is several GB per intermediate, so the score can be computed in
        # edge chunks that are reduced into the per-node accumulator one at a time; with checkpoint_chunks the
        # chunk intermediates are not kept for the backward pass (recomputed). edge_chunks=1 reproduces the
        # unchunked computation exactly.
        self.edge_chunks = int(edge_chunks)
        self.checkpoint_chunks = bool(checkpoint_chunks)

    def _graph_kernel_tensors(self, reference: torch.Tensor):  # noqa: ANN202
        """Edge tensors cached per device and dtype.

        The base class rebuilds source/destination indices and weights from the CSR matrix on every call
        (~20 bytes per edge, ~0.4 GB for 2e7 edges); allocating that afresh every step fragments the GPU
        cache next to a large model. The graph is static, so the tensors are built once and reused.
        """
        key = (reference.device, reference.dtype)
        if getattr(self, "_edge_tensor_cache_key", None) != key:
            self._edge_tensor_cache = super()._graph_kernel_tensors(reference)
            self._edge_tensor_cache_key = key
        return self._edge_tensor_cache

    @property
    def name(self) -> str:
        prefix = "f" if self.fair else ""
        return f"{prefix}graph_variogram_score_p{self.p:g}"

    def _edge_variogram(
        self,
        node_values: torch.Tensor,
        source_index: torch.Tensor,
        destination_index: torch.Tensor,
        edge_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        edge_difference = self._edge_difference(node_values, source_index, destination_index)
        if edge_valid is not None:
            edge_difference = torch.where(
                edge_valid,
                edge_difference,
                torch.zeros_like(edge_difference),
            )
        return torch.abs(edge_difference).pow(self.p)

    def _edge_scores(
        self,
        y_pred_ens: torch.Tensor,
        y_target: torch.Tensor,
        source_index: torch.Tensor,
        destination_index: torch.Tensor,
        edge_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-edge (fair) variogram score for the given edge subset, shape (..., num_edges, num_vars)."""
        ensemble_size = y_pred_ens.shape[2]
        observed_variogram = self._edge_variogram(y_target, source_index, destination_index, edge_valid)
        member_sum = torch.zeros_like(observed_variogram)
        if self.fair:
            member_cross_sum = torch.zeros_like(observed_variogram)
            running_sum = torch.zeros_like(observed_variogram)

        for member in range(ensemble_size):
            member_variogram = self._edge_variogram(
                y_pred_ens[:, :, member],
                source_index,
                destination_index,
                edge_valid,
            )
            member_sum = member_sum + member_variogram
            if self.fair:
                member_cross_sum = member_cross_sum + member_variogram * running_sum
                running_sum = running_sum + member_variogram

        member_mean = member_sum / ensemble_size
        if self.fair:
            return (
                observed_variogram.square()
                - 2.0 * observed_variogram * member_mean
                + 2.0 * member_cross_sum / (ensemble_size * (ensemble_size - 1))
            )
        return (member_mean - observed_variogram).square()

    def _compute_local_score_tensor(
        self,
        y_pred_ens: torch.Tensor,
        y_target: torch.Tensor,
        matrix: torch.Tensor | None,
        source_index: torch.Tensor | None,
        destination_index: torch.Tensor | None,
        edge_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        assert matrix is None
        assert source_index is not None
        assert destination_index is not None
        assert edge_weights is not None
        node_valid, edge_valid, valid_weight_sum = self._compute_edge_validity(
            y_pred_ens,
            y_target,
            source_index,
            destination_index,
            edge_weights,
        )

        if self.edge_chunks == 1:
            edge_score = self._edge_scores(y_pred_ens, y_target, source_index, destination_index, edge_valid)
            return self._aggregate_edge_values(
                edge_score,
                destination_index,
                edge_weights,
                y_target.shape[-2],
                node_valid=node_valid,
                edge_valid=edge_valid,
                valid_weight_sum=valid_weight_sum,
            )

        # Chunked over edges: accumulate the weighted per-node sums chunk by chunk (the row normalisation / NaN
        # masking of
        # _aggregate_edge_values is applied once at the end, on the accumulated node sums).
        num_nodes = y_target.shape[-2]
        num_edges = source_index.shape[0]
        node_values = torch.zeros(
            (*y_target.shape[:-2], num_nodes, y_target.shape[-1]),
            dtype=y_target.dtype,
            device=y_target.device,
        )
        weight_shape = (1,) * (y_target.ndim - 2) + (-1, 1)
        bounds = torch.linspace(0, num_edges, self.edge_chunks + 1, dtype=torch.int64).tolist()

        def chunk_sum(
            y_pred_chunk: torch.Tensor,
            y_target_chunk: torch.Tensor,
            src: torch.Tensor,
            dst: torch.Tensor,
            weights: torch.Tensor,
            valid: torch.Tensor | None,
        ) -> torch.Tensor:
            score = self._edge_scores(y_pred_chunk, y_target_chunk, src, dst, valid)
            if valid is not None:
                score = torch.where(valid, score, torch.zeros_like(score))
            acc = torch.zeros_like(node_values)
            acc.index_add_(-2, dst, score * weights.to(dtype=score.dtype).view(weight_shape))
            return acc

        for start, stop in itertools.pairwise(bounds):
            if stop <= start:
                continue
            src = source_index[start:stop]
            dst = destination_index[start:stop]
            weights = edge_weights[start:stop]
            valid = None if edge_valid is None else edge_valid[..., start:stop, :]
            if self.checkpoint_chunks and torch.is_grad_enabled():
                acc = torch.utils.checkpoint.checkpoint(
                    chunk_sum,
                    y_pred_ens,
                    y_target,
                    src,
                    dst,
                    weights,
                    valid,
                    use_reentrant=False,
                )
            else:
                acc = chunk_sum(y_pred_ens, y_target, src, dst, weights, valid)
            node_values = node_values + acc

        if node_valid is None:
            return node_values
        assert valid_weight_sum is not None
        if self.row_normalize:
            safe_weight_sum = torch.where(valid_weight_sum > 0, valid_weight_sum, torch.ones_like(valid_weight_sum))
            node_values = node_values / safe_weight_sum
        return node_values.masked_fill((valid_weight_sum <= 0) | ~node_valid, torch.nan)
