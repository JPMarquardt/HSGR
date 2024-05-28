import torch
from sinn.noise.utils import dist_matrix, find_min_dist

def gaussian_noise(tensor, noise_size):
    """Add Gaussian noise to a tensor.

    Args:
        tensor (torch.Tensor): Input tensor.
        noise (torch): The magnitude of the noise

    Returns:
        torch.Tensor: Noisy tensor.
    """
    min_dist = find_min_dist(tensor)

    return tensor + torch.randn(tensor.shape) * min_dist * noise_size
    
def noise_regression(tensor):
    noise_stdev = torch.rand()
    noisy_tensor = gaussian_noise(tensor, noise_stdev)

    return (noise_stdev, noisy_tensor)
    