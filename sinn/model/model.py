import dgl
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from sinn.layers.layers import SchnetConv
from sinn.layers.embeddings import color_invariant_duplet, color_invariant_triplet, SchnetEmbedding
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

        self.radial_embedding = nn.ModuleList([
            radial_basis_func(hidden_features, (0, 1)),
            MLP(hidden_features, hidden_features),
        ])
        self.angle_embedding = nn.ModuleList([
            radial_basis_func(hidden_features, (-1, 1)),
            MLP(hidden_features, hidden_features),
        ])

        self.node_embedding = SchnetEmbedding(in_feats=hidden_features,
                                              out_feats=hidden_features,
                                              radial_feats=hidden_features,
                                              var='d',
                                              cutoff=True,
                                              in_range=(0, 1.0))
        self.edge_embedding = color_invariant_duplet(hidden_features)
        self.triplet_embedding = color_invariant_triplet(hidden_features)

        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(SchnetConv(in_feats=hidden_features,
                                              out_feats=hidden_features,
                                              radial_feats=hidden_features,
                                              var='d',
                                              cutoff=True,
                                              in_range=(0, 1.0)))
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

        x = self.node_embedding(g)
        y = self.edge_embedding(g)

        for layer in self.layers:
            x = F.relu(layer(g, x))

        x = dgl.mean_nodes(g, x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class SchNet(nn.Module):
    def __init__(self, num_classes: int, num_layers: int, hidden_features: int, **kwargs):
        super(SchNet, self).__init__()
        self.kwargs = kwargs

        self.radial_embedding = radial_basis_func(hidden_features, (0, 1.0))
        self.cutoff = SmoothCutoff(1.0)

        self.node_embedding = SchnetEmbedding(in_feats=hidden_features,
                                              out_feats=hidden_features,
                                              radial_feats=hidden_features,
                                              var='d',
                                              cutoff=True,
                                              in_range=(0, 1.0))
        self.edge_embedding = color_invariant_duplet(hidden_features)

        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(SchnetConv(in_feats=hidden_features,
                                              out_feats=hidden_features,
                                              radial_feats=hidden_features,
                                              var='d',
                                              cutoff=True,
                                              in_range=(0, 1.0)))
        self.fc = nn.Linear(hidden_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def get_bf_cutoff(self, g) -> None:
        if g.edata.get('cutoff') is None:
            bf = self.radial_embedding(g.edata['d'])
            cutoff = self.cutoff(g.edata['d']).unsqueeze(-1)

            g.edata['bf'] = bf * cutoff
            g.edata['cutoff'] = cutoff
        return 

    def forward(self, g):

        self.get_bf_cutoff(g)

        g = g.local_var()
        g.edata['h'] = self.edge_embedding(g)

        x = self.node_embedding(g)
        g.ndata['h'] = x

        for layer in self.layers:
            x += layer(g)
            g.ndata['h'] = x

        x = self.fc(x)
        x = dgl.mean_nodes(g, x)
        x = self.fc2(x)
        if self.kwargs.get('classification'):
            return F.log_softmax(x, dim=1)
        else: return x