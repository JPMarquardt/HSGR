import dgl
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from sinn.layers.layers import SchnetConv, AlignnConv
from sinn.graph.graph import create_linegraph
from sinn.layers.embeddings import color_invariant_duplet, color_invariant_triplet
from sinn.layers.utils import radial_basis_func, MLP, SmoothCutoff
from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)

# Create a GNN model with modular layers and hidden features
"""
kwargs = {range: tuple[int] = (0, 1), cutoff: tuple[int] = (0.8, 1.0)}
"""

class Alignn(nn.Module):
    def __init__(self, num_classes: int, num_layers: int, hidden_features: int, radial_features: int, **kwargs):
        super(Alignn, self).__init__()
        self.kwargs = kwargs

        self.radial_embedding = radial_basis_func(hidden_features, (0, 1.0))
        self.cutoff = SmoothCutoff(1.0)

        self.node_embedding = torch.ones(hidden_features)
        self.edge_embedding = color_invariant_duplet(hidden_features)
        self.triplet_embedding = color_invariant_triplet(hidden_features)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(AlignnConv(in_feats=hidden_features,
                                              out_feats=hidden_features,
                                              radial_feats=radial_features
                                              ))

        self.fc = nn.Linear(hidden_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes)

        self.pooling = dgl.nn.AvgPooling()

    def get_bf_cutoff(self, g) -> None:
        if g.edata.get('cutoff') is None:
            bf = self.radial_embedding(g.edata['d'])
            cutoff = self.cutoff(g.edata['d']).unsqueeze(-1)

            g.edata['bf'] = bf * cutoff
            g.edata['cutoff'] = cutoff
        return 

    def forward(self, g):
        if isinstance(g, tuple):
            g, h = g
        else:
            h = create_linegraph(g)

        g = g.local_var()
        h = h.local_var()

        self.get_bf_cutoff(g)
        self.get_bf_cutoff(h)

        n = g.number_of_nodes()

        #x, y, z
        g.ndata['h'] = self.node_embedding.repeat(n, 1)
        g.edata['h'] = self.edge_embedding(g)
        h.edata['h'] = self.triplet_embedding(h)

        #update node and edge features
        for layer in self.layers:
            g.edata['h'] = h.ndata['h']
            g.ndata['h'], h.ndata['h'] = layer(g, h)

        #final fully connected layers
        x = g.ndata['h']
        x = self.fc(x)
        x = self.pooling(g, x)
        x = self.fc2(x)

        if self.kwargs.get('classification'):
            return F.log_softmax(x, dim=1)
        
        return x.squeeze(1)

class SchNet(nn.Module):
    def __init__(self, num_classes: int, num_layers: int, hidden_features: int, radial_features: int, **kwargs):
        super(SchNet, self).__init__()
        self.kwargs = kwargs

        self.radial_embedding = radial_basis_func(radial_features, (0, 1.0))
        self.cutoff = SmoothCutoff(1.0)

        self.register_buffer('node embedding', torch.ones(hidden_features))
        self.edge_embedding = color_invariant_duplet(hidden_features)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SchnetConv(in_feats=hidden_features,
                                              out_feats=hidden_features,
                                              radial_feats=radial_features,
                                              var='d',
                                              cutoff=True,
                                              in_range=(0, 1.0)))

        self.fc = nn.Linear(hidden_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes)

        self.pooling = dgl.nn.AvgPooling()

    def get_bf_cutoff(self, g) -> None:
        if g.edata.get('cutoff') is None:
            bf = self.radial_embedding(g.edata['d'])
            cutoff = self.cutoff(g.edata['d']).unsqueeze(-1)

            g.edata['bf'] = bf * cutoff
            g.edata['cutoff'] = cutoff
        return 

    def forward(self, g):
        g = g.local_var()
        self.get_bf_cutoff(g)

        n = g.number_of_nodes()
        g.ndata['h'] = self.node_embedding.repeat(n, 1)
        g.edata['h'] = self.edge_embedding(g)

        for layer in self.layers:
            g.ndata['h'] = g.ndata['h'] + layer(g)

        x = g.ndata['h']
        x = self.fc(x)
        x = self.pooling(g, x)
        x = self.fc2(x)

        if self.kwargs.get('classification'):
            return F.log_softmax(x, dim=1)
        
        return x.squeeze(1)