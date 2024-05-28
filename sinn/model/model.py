import dgl
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from sinn.layers.layers import SchnetConv
from sinn.layers.embeddings import color_invariant_duplet, color_invariant_triplet
from sinn.layers.utils import radial_basis_func, MLP, SmoothCutoff
from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)

# Create a GNN model with modular layers and hidden features
"""
kwargs = {range: tuple[int] = (0, 1), cutoff: tuple[int] = (0.8, 1.0)}
"""

class Alignn(nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden_features, **kwargs):
        super(Alignn, self).__init__()
        self.kwargs = kwargs

        self.radial_embedding = nn.ModuleList(
            radial_basis_func(hidden_features, (0, 1)),
            MLP(hidden_features, hidden_features),
        )
        self.angle_embedding = nn.ModuleList(
            radial_basis_func(hidden_features, (-1, 1)),
            MLP(hidden_features, hidden_features),
        )

        self.edge_embedding = color_invariant_duplet(hidden_features)
        self.triplet_embedding = color_invariant_triplet(hidden_features)

        self.layers = nn.ModuleList()
        self.layers.append(SchnetConv(num_features, hidden_features, **kwargs))
        for _ in range(num_layers - 1):
            self.layers.append(SchnetConv(hidden_features, hidden_features))
        self.fc = nn.Linear(hidden_features, num_classes)

    def get_bf_cutoff(self, g) -> None:
        if g.edata.get('cutoff') is None:
            bf = self.basis_func()
            cutoff = self.cutoff()

            g.edata['bf'] = bf * cutoff
            g.edata['cutoff'] = cutoff
        return 

    def forward(self, g):
        g, h = g

        self.get_bf_cutoff(g)
        self.get_bf_cutoff(h)

        g = g.local_var()
        h = h.local_var()

        for layer in self.layers:
            x = F.relu(layer(g, x))

        x = dgl.mean_nodes(g, x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class SchNet(nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden_features, **kwargs):
        super(SchNet, self).__init__()
        self.kwargs = kwargs

        self.radial_embedding = nn.ModuleList(
            radial_basis_func(hidden_features, (0, 1)),
            MLP(hidden_features, hidden_features),
        )

        self.edge_embedding = color_invariant_duplet(hidden_features)

        self.layers = nn.ModuleList()
        self.layers.append(SchnetConv(num_features, hidden_features, **kwargs))
        for _ in range(num_layers - 1):
            self.layers.append(SchnetConv(hidden_features, hidden_features))
        self.fc = nn.Linear(hidden_features, num_classes)

    def get_bf_cutoff(self, g) -> None:
        if g.edata.get('cutoff') is None:
            bf = self.basis_func()
            cutoff = self.cutoff()

            g.edata['bf'] = bf * cutoff
            g.edata['cutoff'] = cutoff
        return 

    def forward(self, g):
        self.get_bf_cutoff(g)

        g = g.local_var()

        for layer in self.layers:
            x = F.relu(layer(g, x))

        x = dgl.mean_nodes(g, x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)