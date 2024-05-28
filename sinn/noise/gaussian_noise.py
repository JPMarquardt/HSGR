import torch
from sinn.noise.utils import dist_matrix, find_min_dist

def gaussian_noise(tensor, noise):
    """Add Gaussian noise to a tensor.

    Args:
        tensor (torch.Tensor): Input tensor.
        ratio (float, optional): Ratio of the standard deviation of the Gaussian noise to the mean distance between atoms

    Returns:
        torch.Tensor: Noisy tensor.
    """
    min_dist = find_min_dist(tensor)

    return tensor + noise * min_dist
    
def noise_regression(tensor):
    noise_stdev = torch.rand()
    noise = torch.randn(tensor.shape) * noise_stdev

    return (noise_stdev, noisy_tensor)
    