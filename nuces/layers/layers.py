import torch

import dgl
import dgl.function as fn
import torch.nn as nn
from utils import SmoothCutoff, MLP

class SchnetConv(nn.Module):
    """
    Graph convolution layer from SchNet model
    Inputs:
    - in_feats: int, input features
    - out_feats: int, output features
    - var: str, variable to use for convolution
    Forward:
    - g: dgl.DGLGraph, input graph
    Outputs:
    - out: torch.tensor, output features
    """
    def __init__(self, in_feats: int = 64, out_feats: int = 64, var: str = 'd', **kwargs):
        super(SchnetConv, self).__init__()
        self.var = var

        if cutoff in kwargs:
            cutoff = kwargs['cutoff']
            self.onset = cutoff.onset
            self.cutoff = cutoff.cutoff
            cutoff = SmoothCutoff(onset=self.onset, cutoff=self.cutoff)
        else:
            self.cutoff = SmoothCutoff()
            self.onset = 0.8
            self.cutoff = 1.0

        if var == 'd':
            in_range = (0, self.cutoff)
        elif var == 'angle':
            in_range = (-self.cutoff, self.cutoff)


        self.register_buffer('gamma', in_feats / (in_range[1] - in_range[0]))
        self.register_buffer('muk', torch.linspace(in_range[0], in_range[1], in_feats))
        
        self.MLP = MLP(in_feats, out_feats)

    def basis_func(self, dist):
        return torch.exp(-self.gamma * (dist - self.muk)**2)

    def message_func(self, edges):
        src = edges.src['h']        
        dist = edges.data[self.var]
        cutoff = self.cutoff(dist)
        return {'h': src * self.basis_func(dist) * cutoff.unsqueeze(-1)}

    def reduce_func(self, nodes):
        return {'h': torch.sum(nodes.mailbox['h'], dim=1)}

    def forward(self, g):
        g = g.local_var()

        g.update_all(self.message_func, self.reduce_func)
        out = self.MLP(g.ndata['h'])
        return out

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
    def __init__(self, in_feats: int = 64, out_feats: int = 64, **kwargs):
        super(AlignnConv, self).__init__()

        self.r_conv = SchnetConv(in_feats, out_feats, var='d', **kwargs)
        self.angle_conv = SchnetConv(in_feats, out_feats, var='angle', **kwargs)

    def forward(self, g: tuple[dgl.DGLGraph, dgl.DGLGraph]):
        g, h = g

        g = g.local_var()
        h = h.local_var()

        v = self.r_conv(g)
        e = self.angle_conv(h)

        return (v, e)