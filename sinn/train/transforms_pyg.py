import torch
import torch.nn as nn

from sinn.graph.graph_pyg import create_periodic_knn_graph, create_aperiodic_knn_graph, Graph


class PeriodicKNN_PyG(nn.Module):
    def __init__(self, k: int = 17):
        super(PeriodicKNN_PyG, self).__init__()
        self.k = k

    def forward(self, datapoint):
        return create_periodic_knn_graph(datapoint, self.k)
    
class AperiodicKNN_PyG(nn.Module):
    def __init__(self, k: int = 17):
        super(AperiodicKNN_PyG, self).__init__()
        self.k = k

    def forward(self, datapoint: dict[str, torch.Tensor]) -> Graph:
        return create_aperiodic_knn_graph(datapoint, self.k)
    
class AddNoise(nn.Module):
    def __init__(self, std: float = 0.1, transform: nn.Module = None):
        super(AddNoise, self).__init__()
        self.transform = transform
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std