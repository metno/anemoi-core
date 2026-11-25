# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch


class _SparseMMPreparedTranspose(torch.autograd.Function):
    """``matrix @ x`` whose backward pass uses a transposed matrix prepared in advance.

    Same result as ``torch.sparse.mm``; the built-in backward transposes the CSR matrix on every call.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, matrix: torch.Tensor, matrix_t: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        ctx.matrix_t = matrix_t
        return torch.sparse.mm(matrix, x)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:  # noqa: ANN001
        matrix_t = ctx.matrix_t
        with torch.amp.autocast(device_type=grad_output.device.type, enabled=False):
            grad = torch.sparse.mm(matrix_t, grad_output.to(matrix_t.dtype))
        return grad.to(grad_output.dtype), None, None


class SparseProjector(torch.nn.Module):
    """Applies a sparse projection matrix to input tensors.

    The projection matrix is provided by the GraphProvider in CSR format.

    Stateless: the matrix is passed to :meth:`forward`, not stored.
    """

    def __init__(self, autocast: bool = False, num_chunks: int = 1) -> None:
        """Initialize SparseProjector.

        Parameters
        ----------
        autocast : bool
            Use automatic mixed precision
        num_chunks : int
            Default number of chunks to project with sparse matmul.
        """
        super().__init__()
        self.autocast = autocast
        self.num_chunks = num_chunks

    def forward(
        self,
        x: torch.Tensor,
        projection_matrix: torch.Tensor,
        num_chunks: int | None = None,
        transposed_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply sparse projection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``[..., input_nodes, channels]``.
        projection_matrix : torch.Tensor
            Sparse projection matrix (assumed to be on the correct device)
        num_chunks : int, optional
            Number of chunks to project with sparse matmul. ``1``
            projects all in one matmul. Defaults to the value configured
            when constructing the projector.
        transposed_matrix : torch.Tensor, optional
            The transposed projection matrix (CSR, same device). If given it is used in the backward
            pass instead of letting autograd transpose ``projection_matrix`` on every call; the
            result is the same.

        Returns
        -------
        torch.Tensor
            Projected tensor
        """
        num_chunks = self.num_chunks if num_chunks is None else num_chunks
        input_shape = x.shape
        x = x.reshape(-1, *input_shape[-2:])

        try:
            if num_chunks == 1:
                out = self._project_flattened(x, projection_matrix, transposed_matrix)
            else:
                out = torch.cat(
                    [
                        self._project_flattened(chunk, projection_matrix, transposed_matrix)
                        for chunk in torch.chunk(x, num_chunks, dim=0)
                    ],
                    dim=0,
                )
        except torch.cuda.OutOfMemoryError as e:
            raise torch.cuda.OutOfMemoryError(
                f"Out of memory during sparse projection. Consider doubling the number of chunks (currently {num_chunks}). This will reduce memory usage at the cost of additional kernel launches, which can increase runtime. Original error: {e}"
            ) from e

        return out.reshape(*input_shape[:-2], *out.shape[-2:])

    def _project_flattened(
        self,
        x: torch.Tensor,
        projection_matrix: torch.Tensor,
        transposed_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project an input whose leading dimensions have already been flattened.

        Expected x shape: [flat_batch, input_nodes, channels]
        projection_matrix shape: [output_nodes, input_nodes]
        """
        batch_size = x.shape[0]
        input_nodes = x.shape[1]
        trailing_shape = x.shape[2:]

        # [B, N, ...] -> [B, N, C]
        x_flat = x.reshape(batch_size, input_nodes, -1)

        # Move batch/features into the dense RHS columns:
        # [B, N, C] -> [N, B, C] -> [N, B*C]
        x_rhs = x_flat.permute(1, 0, 2).reshape(input_nodes, -1)

        with torch.amp.autocast(device_type=x.device.type, enabled=self.autocast):
            # One sparse matmul instead of B sparse matmuls:
            # [M, N] @ [N, B*C] -> [M, B*C]
            if transposed_matrix is not None and x_rhs.requires_grad:
                out = _SparseMMPreparedTranspose.apply(x_rhs, projection_matrix, transposed_matrix)
            else:
                out = torch.sparse.mm(projection_matrix, x_rhs)

        output_nodes = out.shape[0]

        # [M, B*C] -> [M, B, C] -> [B, M, C] -> [B, M, ...]
        out = out.reshape(output_nodes, batch_size, -1).permute(1, 0, 2)
        return out.reshape(batch_size, output_nodes, *trailing_shape)
