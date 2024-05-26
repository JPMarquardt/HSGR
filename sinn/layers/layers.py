import torch

import dgl
import dgl.function as fn
import torch.nn as nn
from sinn.layers.utils import SmoothCutoff, MLP, radial_basis_func
from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)

class SchnetConv(nn.Module):
    """
    Graph convolution layer from SchNet model (adds edge features too)
    Inputs:
    - in_feats: int, input features
    - out_feats: int, output features
    - var: str, variable to use for convolution
    Forward:
    - g: dgl.DGLGraph, input graph
    Outputs:
    - out: torch.tensor, output features
    """
    def __init__(self, 
                 in_feats: int = 64,
                 radial_feats: int = 128,
                 out_feats: int = 64, 
                 var: str = 'd', 
                 cutoff: bool = True, 
                 in_range: tuple[float, float] = None, 
                 **kwargs):
        super(SchnetConv, self).__init__()
        self.var = var

        #cutoff function
        if cutoff:
            max = in_range[1]
            self.cutoff = SmoothCutoff(cutoff=max)
        else:
            self.cutoff = SmoothCutoff(cutoff=False)

        #initialize radial basis function
        self.basis_func = radial_basis_func(radial_feats, in_range)
        
        #filter generation network
        self.FGN_MLP1 = MLP(radial_feats, in_feats)
        self.FGN_MLP2 = MLP(in_feats, in_feats)

        #interaction block
        self.IB_MLP = MLP(in_feats, out_feats)

    def cfconv(self, edges):
        src_feat = edges.src['h']
        edge_feat = edges.data['h']

        cutoff = edges.data['cutoff']
        bf = edges.data['bf']

        return {'h': src_feat * edge_feat * bf * cutoff.unsqueeze(-1)}

    def reduce_func(self, nodes):
        return {'h': torch.prod(nodes.mailbox['h'], dim=1)}

    def forward(self, g):
        g = g.local_var()

        e_var = g.edata[self.var]

        if g.edata.get('cutoff') is None:
            bf = self.basis_func(e_var)
            cutoff = self.cutoff(e_var)

            g.edata['bf'] = bf * cutoff
            g.edata['cutoff'] = cutoff

        bf = self.FGN_MLP1(bf)
        bf = self.FGN_MLP2(bf)

        g.update_all(self.cfconv, self.reduce_func)
        out = self.IB_MLP(g.ndata['h'])
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

    def forward(self, g: Tuple[dgl.DGLGraph, dgl.DGLGraph]):
        g, h = g

        g = g.local_var()
        h = h.local_var()

        v = self.r_conv(g)
        e = self.angle_conv(h)

        return (v, e)