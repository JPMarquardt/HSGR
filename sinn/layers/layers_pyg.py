import torch

import torch.nn as nn
from sinn.layers.utils_pyg import SmoothCutoff, MLP
from sinn.graph.graph_pyg import Graph
from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)

class SchnetConv(nn.Module):
    """
    Graph convolution layer from SchNet model (adds edge features too)
    Inputs:
    - in_feats: int, input features
    - out_feats: int, output features
    - var: str, variable to use for convolution
    Forward:
    - g: dict, input graph
    Outputs:
    - out: torch.tensor, output features
    """
    def __init__(self, 
                 in_feats: int = 64,
                 radial_feats: int = 256,
                 hidden_feats: int = 128,
                 out_feats: int = 64, 
                 var: str = 'r', 
                 **kwargs):
        super(SchnetConv, self).__init__()
        self.var = var

        #filter generation network
        self.FGN_MLP1 = MLP(radial_feats, hidden_feats)
        self.FGN_MLP2 = MLP(hidden_feats, out_feats)

        #interaction block
        self.IB_MLP1 = MLP(out_feats, out_feats)
        self.IB_MLP2 = MLP(out_feats, out_feats)

    def forward(self, g: Graph):
        bf = g['bf']

        print(bf.type)
        print(self.FGN_MLP1.linear.weight.dtype)
        bf = self.FGN_MLP1(bf)
        bf = self.FGN_MLP2(bf)

        g['h_src'] = torch.index_select(g['h'], 0 , g['knn'].flatten())
        g['h_src'] = g['h_src'].unflatten(0, g['knn'].shape)

        h = torch.sum(g['h_src'] * bf * g['h_edge'] * g['cutoff'].unsqueeze(-1), dim=-2)

        out = self.IB_MLP1(h)
        return self.IB_MLP2(out)


class AlignnConv(nn.Module):
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
    def __init__(self, 
                 in_feats: int = 64,
                 radial_feats: int = 256,
                 hidden_feats: int = 128,
                 out_feats: int = 64, 
                 **kwargs):
        super(AlignnConv, self).__init__()

        self.r_conv = SchnetConv(in_feats, radial_feats=radial_feats, hidden_feats=hidden_feats, out_feats=out_feats)
        self.angle_conv = SchnetConv(in_feats, radial_feats=radial_feats, hidden_feats=hidden_feats, out_feats=out_feats)

    def forward(self, g: Tuple[Graph, Graph]):
        g, h = g

        v = self.r_conv(g)
        e = self.angle_conv(h)

        return (v, e)
    