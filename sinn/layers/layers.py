import torch

import dgl
import dgl.function as fn
import torch.nn as nn
from sinn.layers.utils import SmoothCutoff, MLP
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
                 var: str = 'r', 
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
        
        #filter generation network
        self.FGN_MLP1 = MLP(radial_feats, in_feats)
        self.FGN_MLP2 = MLP(in_feats, in_feats)

        self.src_mul_edge = fn.u_mul_e('h','h','m')
        self.mean_coll = fn.mean('m', 'h')

        #interaction block
        self.IB_MLP1 = MLP(in_feats, out_feats)
        self.IB_MLP2 = MLP(in_feats, in_feats)

    def forward(self, 
                g: dgl.DGLGraph,
                ):
        g = g.local_var()

        bf = g.edata['bf']

        bf = self.FGN_MLP1(bf)
        bf = self.FGN_MLP2(bf)

        g.edata['h'] = bf * g.edata['h']

        g.update_all(self.src_mul_edge, self.mean_coll)
        
        out = self.IB_MLP1(g.ndata['h'])
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
    def __init__(self, in_feats: int = 64, out_feats: int = 64, **kwargs):
        super(AlignnConv, self).__init__()

        self.r_conv = SchnetConv(in_feats, out_feats, var='r', in_range=(0,1))
        self.angle_conv = SchnetConv(in_feats, out_feats, var='r', in_range=(-1,1))

    def forward(self, g: Tuple[dgl.DGLGraph, dgl.DGLGraph]):
        g, h = g

        g = g.local_var()
        h = h.local_var()

        v = self.r_conv(g)
        e = self.angle_conv(h)

        return (v, e)