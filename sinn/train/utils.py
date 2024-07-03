import torch

def gen_to_func(y, device):
    if isinstance(y, tuple):
        return lambda x: tuple(graph_part.to(device) for graph_part in x)
    else:
        return lambda x: x.to(device)

def gaussian_noise(tensor, noise_size):
    """Add Gaussian noise to a tensor.

    Args:
        tensor (torch.Tensor): Input tensor.
        noise (torch): The magnitude of the noise

    Returns:
        torch.Tensor: Noisy tensor.
    """
    min_dist = find_min_dist(tensor)
    noise = torch.randn(tensor.shape) * min_dist * noise_size
    return tensor + noise

def dist_matrix(tensor: torch.Tensor):
    x1 = torch.unsqueeze(tensor, 0)
    x2 = torch.unsqueeze(tensor, 1)
    dists = torch.norm(x1 - x2, dim=-1)
    return dists

def find_min_dist(tensor: torch.Tensor) -> torch.Tensor:
    dists = dist_matrix(tensor)
    min_dist = torch.min(dists[dists != 0])
    return min_dist
