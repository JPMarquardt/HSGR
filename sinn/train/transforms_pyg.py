import torch
import torch.nn as nn
import os

from random import sample
from typing import Union
from MDAnalysis import Universe

from sinn.dataset.dataset import universe2df
from sinn.graph.graph_pyg import create_periodic_knn_graph, create_aperiodic_knn_graph, Graph


class PeriodicKNN_PyG(nn.Module):
    def __init__(self, k: int = 19):
        super(PeriodicKNN_PyG, self).__init__()
        self.k = k

    def forward(self, datapoint: dict[str, torch.Tensor]) -> Graph:
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
        x['positions'] += noise

        return self.transform(x)

class RandomTransform(nn.Module):
    def __init__(self, probability: float, transform1: Union[nn.Module, None] = None, transform2: Union[nn.Module, None] = None):
        super(RandomTransform, self).__init__()
        if transform1 is None:
            transform1 = nn.Identity()
        if transform2 is None:
            transform2 = nn.Identity()

        self.transform1 = transform1
        self.transform2 = transform2

        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.probability:
            return self.transform1(x)
        else:
            return self.transform2(x)

class ReplaceWithLiquid(nn.Module):
    def __init__(self, path: str):
        super(ReplaceWithLiquid, self).__init__()
        self.path = path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        file_list = os.listdir(self.path)
        file = sample(file_list, 1)[0]

        data = Universe(f'{self.path}/{file}')
        
        return universe2df(data)
