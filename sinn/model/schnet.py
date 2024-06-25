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

class SchNet(nn.Module):
    def __init__(self, num_classes: int, num_layers: int, hidden_features: int, radial_features: int, **kwargs):
        super(SchNet, self).__init__()
        self.kwargs = kwargs

        self.radial_embedding = radial_basis_func(radial_features, (0, 1.0))
        self.cutoff = SmoothCutoff(1.0)

        self.register_buffer('node_embedding', torch.ones(hidden_features), persistent=False)
        self.edge_embedding = color_invariant_duplet(hidden_features)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SchnetConv(in_feats=hidden_features,
                                              out_feats=hidden_features,
                                              radial_feats=radial_features,
                                              var='r',
                                              cutoff=True,
                                              in_range=(0, 1.0)))

        self.fc = MLP(hidden_features, hidden_features)
        self.fc2 = MLP(hidden_features, num_classes)

        self.pooling = dgl.nn.AvgPooling()

    def get_bf_cutoff(self, g) -> None:
        if g.edata.get('cutoff') is None:
            bf = self.radial_embedding(g.edata['r'])
            cutoff = self.cutoff(g.edata['r']).unsqueeze(-1)

            g.edata['bf'] = bf * cutoff
            g.edata['cutoff'] = cutoff
        return 

    def forward(self, g):
        g = g.local_var()
        self.get_bf_cutoff(g)

        n = g.number_of_nodes()
        print(self.node_embedding.device)
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
    
class SchNet_Multihead(nn.Module):
    def __init__(self, num_classes, num_layers, hidden_features, radial_features):
        super(SchNet_Multihead, self).__init__()
        self.model = SchNet(num_classes=num_classes+1, num_layers=num_layers, hidden_features=hidden_features, radial_features=radial_features)
        self.classifier = MLP(num_classes+1, num_classes)
        self.regression = MLP(num_classes+1, 1)

        self.sm = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.model(x)

        reg_pred = self.regression(x)
        class_pred = self.classifier(x)

        class_pred = self.sm(class_pred)
        
        return class_pred, reg_pred