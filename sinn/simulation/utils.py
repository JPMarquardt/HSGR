import torch

from sinn.noise.utils import dist_matrix

def find_max_k_dist(tensor: torch.Tensor, k: int) -> torch.Tensor:
    """
    finds the maximum kth distance between atoms in a tensor
    """
    dists = dist_matrix(tensor)
    dist_n = torch.argsort(dists, dim=-1)

    # 1 accounts for self distance
    k_dist = dists[dist_n == k + 1]
    max_k_dist = torch.max(k_dist)

    return max_k_dist

