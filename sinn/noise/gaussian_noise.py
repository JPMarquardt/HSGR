import torch
from sinn.noise.utils import dist_matrix, find_min_dist

def gaussian_noise(tensor, ratio=0.1):
    """Add Gaussian noise to a tensor.

    Args:
        tensor (torch.Tensor): Input tensor.
        ratio (float, optional): Ratio of the standard deviation of the Gaussian noise to the mean distance between atoms

    Returns:
        torch.Tensor: Noisy tensor.
    """
    min_dist = find_min_dist(tensor)

    return tensor + torch.randn(tensor.size()) * min_dist * ratio

def check_gaussian_noise(tensor, noisy_tensor):
    """Check that the Gaussian noise is added correctly.

    Args:
        tensor (torch.Tensor): Input tensor.
        noise (torch.Tensor): Gaussian noise tensor.

    Returns:
        torch.Tensor: Noisy tensor.
    """
    #????
    raise NotImplementedError()

    if is_ok:
        return True
    else:
        return False