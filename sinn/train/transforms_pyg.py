import torch
import torch.nn as nn
from typing import Union

from sinn.graph.graph_pyg import create_periodic_knn_graph, create_aperiodic_knn_graph, Graph


class PeriodicKNN_PyG(nn.Module):
    def __init__(self, k: int = 19):
        super(PeriodicKNN_PyG, self).__init__()
        self.k = k

    def forward(self, datapoint):
        return create_periodic_knn_graph(datapoint, self.k)
    
class AperiodicKNN_PyG(nn.Module):
    def __init__(self, k: int = 19):
        super(AperiodicKNN_PyG, self).__init__()
        self.k = k

    def forward(self, datapoint: dict[str, torch.Tensor]) -> Graph:
        return create_aperiodic_knn_graph(datapoint, self.k)
    
class AddNoise(nn.Module):
    def __init__(self, std: float = 0.1, transform: Union[nn.Module, None] = None):
        super(AddNoise, self).__init__()
        if transform is None:
            transform = nn.Identity()
        self.transform = transform
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        data = x['positions']

        dist_matrix = data[:, None, :] - data[None, :, :]
        dist_matrix = torch.norm(dist_matrix, dim=-1)
        min_dist = torch.min(dist_matrix[dist_matrix > 0])
        noise = torch.randn_like(data) * self.std * min_dist
        data += noise
        
        return self.transform(data)