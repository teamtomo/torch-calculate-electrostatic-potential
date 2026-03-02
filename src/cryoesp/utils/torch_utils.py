"""PyTorch utilities."""

import torch


def batched_with_indices(tensor: torch.Tensor, batch_size: int):
    """
    Yields (indices, batch) pairs where:
    - indices: global row indices
    - batch:   slice of the tensor
    """
    total = tensor.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield torch.arange(start, end, device=tensor.device), tensor[start:end]
