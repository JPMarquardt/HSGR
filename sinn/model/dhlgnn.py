import dgl
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from sinn.layers.layers import SchnetConv
from sinn.graph.graph import create_linegraph
from sinn.graph.utils import compute_bond_cross_product
from sinn.layers.embeddings import color_invariant_duplet, color_invariant_triplet, color_invariant_quadruplet
from sinn.layers.utils import radial_basis_func, MLP, SmoothCutoff
from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)

class DHLConv(nn.Module):
    """
    Graph convolution layer from ALIGNN model
    Inputs:
    - in_feats: int, input features
    - out_feats: int, output features
    Forward:
    - g: tuple[radius graph, angle graph]
    Outputs:
    - v: torch.tensor, radius graph output
    - e: torch.tensor, angle graph output
    """
    def __init__(self, in_feats: int = 64, out_feats: int = 64, **kwargs):
        super(DHLConv, self).__init__()

        self.r_conv = SchnetConv(in_feats, out_feats, var='r', in_range=(0,1))
        self.angle_conv = SchnetConv(in_feats, out_feats, var='r', in_range=(-1,1))
        self.dihedral_conv = SchnetConv(in_feats, out_feats, var='r', in_range=(-1,1))

    def forward(self, g: Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g, h, i = g

        g = g.local_var()
        h = h.local_var()
        i = i.local_var()

        v = self.r_conv(g)
        e = self.angle_conv(h)
        ee = self.dihedral_conv(i)

        return (v, e, ee)


class DHLgnn(nn.Module):
    def __init__(self, num_classes: int, num_layers: int, hidden_features: int, radial_features: int, **kwargs):
        super(DHLgnn, self).__init__()
        self.kwargs = kwargs

        self.radial_embedding = radial_basis_func(hidden_features, (0, 1.0))
        self.cosine_embedding = radial_basis_func(hidden_features, (-1, 1))
        self.dihedral_embedding = radial_basis_func(hidden_features, (-1, 1))
        self.cutoff = SmoothCutoff(1.0)

        self.register_buffer('node_embedding', torch.ones(hidden_features), persistent=False)
        self.edge_embedding = color_invariant_duplet(hidden_features)
        self.triplet_embedding = color_invariant_triplet(hidden_features)
        self.quadruplet_embedding = color_invariant_quadruplet(hidden_features)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DHLConv(in_feats=hidden_features,
                                              out_feats=hidden_features,
                                              radial_feats=radial_features
                                              ))

        self.fc = MLP(hidden_features, hidden_features)
        self.fc2 = MLP(hidden_features, num_classes)

        self.pooling = dgl.nn.AvgPooling()

    def get_bf_cutoff(self, g, embedding) -> None:
        if g.edata.get('cutoff') is None:
            bf = embedding(g.edata['r'])
            if g.ndata.get('cutoff') is None:
                cutoff = self.cutoff(g.edata['r']).unsqueeze(-1)


            else:
                g.apply_edges(lambda edges: {'cutoff_src': edges.src['cutoff']})
                g.apply_edges(lambda edges: {'cutoff_dst': edges.dst['cutoff']})

                cutoff = torch.min(g.edata.pop('cutoff_src'), g.edata.pop('cutoff_dst'))

            g.edata['bf'] = bf * cutoff
            g.edata['cutoff'] = cutoff
        return 

    def forward(self, g: Union[dgl.DGLGraph, Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph]]) -> torch.Tensor:
        if isinstance(g, tuple):
            g, h, i = g
        else:
            h = create_linegraph(g)
            h.edata['dr'] = compute_bond_cross_product(h)
            i = create_linegraph(h)

        g = g.local_var()
        h = h.local_var()
        i = i.local_var()

        self.get_bf_cutoff(g, self.radial_embedding)
        h.ndata['cutoff'] = g.edata['cutoff']

        self.get_bf_cutoff(h, self.cosine_embedding)
        i.ndata['cutoff'] = h.edata['cutoff']

        self.get_bf_cutoff(i, self.dihedral_embedding)

        n = g.number_of_nodes()

        #x, y, z
        g.ndata['h'] = self.node_embedding.repeat(n, 1)
        g.edata['h'] = self.edge_embedding(g)
        h.ndata['h'] = g.edata['h']
        h.edata['h'] = self.triplet_embedding((g, h))
        i.ndata['h'] = h.edata['h']
        i.edata['h'] = self.quadruplet_embedding((g, h, i))

        #update node and edge features
        for layer in self.layers:
            h.edata['h'] = i.ndata['h']
            g.edata['h'] = h.ndata['h']
            g.ndata['h'], h.ndata['h'], i.ndata['h'] = layer((g, h, i))

        #final fully connected layers
        x = g.ndata['h']
        x = self.fc(x)
        x = self.pooling(g, x)
        x = self.fc2(x)

        if self.kwargs.get('classification'):
            return F.log_softmax(x, dim=1)
        
        return x.squeeze(1)

class DHL_Multihead(nn.Module):
    def __init__(self, num_classes, num_layers, hidden_features, radial_features):
        super(DHL_Multihead, self).__init__()
        self.model = DHLgnn(num_classes=hidden_features, num_layers=num_layers, hidden_features=hidden_features, radial_features=radial_features)
        self.classifier = MLP(hidden_features, num_classes)
        self.regression = MLP(hidden_features, 1)

        self.sm = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.model(x)
        reg_pred = self.regression(out)
        class_pred = self.classifier(out)
        
        class_pred = self.sm(class_pred)
        
        return class_pred, reg_pred