import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from sinn.layers.layers_pyg import AlignnConv
from sinn.graph.graph_pyg import create_linegraph, Graph
from sinn.layers.embeddings_pyg import color_invariant_duplet, color_invariant_triplet
from sinn.layers.utils_pyg import radial_basis_func, MLP, SmoothCutoff
from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)

import torch.nn.functional as F

class Alignn(nn.Module):
    def __init__(self, 
                 num_layers: int = 3,
                 radial_feats: int = 256,
                 hidden_feats: int = 64,
                 out_feats: int = 64, 
                 **kwargs):
        super(Alignn, self).__init__()
        self.kwargs = kwargs

        self.radial_embedding = radial_basis_func(radial_feats, in_range=(0, 1))
        self.cosine_embedding = radial_basis_func(radial_feats, in_range=(-1, 1))

        self.cutoff = SmoothCutoff(1.0)

        self.register_buffer('node_embedding', torch.ones(hidden_feats), persistent=False)
        self.edge_embedding = color_invariant_duplet(hidden_feats)
        self.triplet_embedding = color_invariant_triplet(hidden_feats)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(AlignnConv(in_feats=hidden_feats,
                                              out_feats=hidden_feats,
                                              radial_feats=radial_feats
                                              ))

        self.fc = MLP(hidden_feats, hidden_feats)
        self.fc2 = MLP(hidden_feats, out_feats)

    def forward(self, g: Union[Graph, Tuple[Graph, Graph]], early_return: bool = False):

        if isinstance(g, tuple):
            g, h = g
        else:
            h = create_linegraph(g)

        g['cutoff'] = self.cutoff(g['r'])
        g['bf'] = self.radial_embedding(g['r'])

        h['cutoff'] = g['cutoff'].unsqueeze(-1).expand(-1, -1, h.k)
        h['bf'] = self.cosine_embedding(h['r'])

        n = g.n_nodes

        #x, y, z
        g['h'] = self.node_embedding.unsqueeze(0).expand(n, -1)
        g['h_edge'] = self.edge_embedding(g)
        h['h'] = g['h_edge']
        h['h_edge'] = self.triplet_embedding(h)

        #update node and edge features
        for layer in self.layers:
            g['h_edge'] = h['h']
            x, y = layer((g, h))
            g['h'] = g['h'] + x
            h['h'] = h['h'] + y
        
        #final fully connected layers
        x = g['h']
        x = self.fc(x)

        if early_return:
            return x
        
        x = torch.sum(x, dim=0)
        x = self.fc2(x)

        if self.kwargs.get('classification'):
            return F.softmax(x, dim=-1)
        
        return x.squeeze()
