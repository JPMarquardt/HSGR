import torch

import dgl
import dgl.function as fn
import torch.nn as nn

from utils import SmoothCutoff, MLP, radial_basis_func
from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)

class MDNetEmbed(nn.Module):
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
                 out_feats: int = 64, 
                 var: str = 'd', 
                 cutoff: bool = True, 
                 in_range: Tuple[float, float] = None, 
                 **kwargs):
        super(MDNetEmbed, self).__init__()
        self.var = var

        #cutoff function
        if cutoff:
            max = in_range[1]
            self.cutoff = SmoothCutoff(cutoff=max)
        else:
            self.cutoff = SmoothCutoff(cutoff=False)

        #initialize radial basis function
        self.basis_func = radial_basis_func(in_feats, in_range)
        
        #filter generation network
        self.FGN_MLP1 = MLP(in_feats, in_feats)
        self.FGN_MLP2 = MLP(in_feats, in_feats)

        #interaction block
        self.IB_MLP = MLP(in_feats, out_feats)

    def forward(self, g):
        return#FIIIEXXXXX


class MDNetAttn(nn.Module):
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
    https://arxiv.org/abs/2202.02541
    """
    def __init__(self, in_feats: int = 64, 
                 hidden_feats: int = 64, #???
                 out_feats: int = 64, 
                 var: str = 'd', 
                 cutoff: bool = True, 
                 in_range: Tuple[float, float] = None, 
                 **kwargs):
        super(MDNetAttn, self).__init__()
        self.var = var

        #cutoff function
        if cutoff:
            max = in_range[1]
            self.cutoff = SmoothCutoff(cutoff=max)
        else:
            self.cutoff = SmoothCutoff(cutoff=False)
        
        #filter generation network
        self.FGN_MLP1 = MLP(in_feats, in_feats)
        self.FGN_MLP2 = MLP(in_feats, in_feats)

        #interaction block
        self.IB_MLP = MLP(in_feats, out_feats)

        #S1 and S2
        self.MLP_S1 = MLP(in_feats, out_feats)
        self.MLP_S2 = MLP(in_feats, out_feats)



    def cfconv(self, edges):
        #make more efficient
        k = edges.src['k']
        q = edges.dst['q']
        v = edges.src['v']

        dV = edges.data['dV']
        dK = edges.data['dK']
        ev = edges.data['h']

        cutoff = edges.data['cutoff']

        weight = torch.nn.SiLU(torch.sum(k * q * dK, dim=-1)) * cutoff
        value = v * ev * dV * cutoff.unsqueeze(-1)

        return {'h': value * weight.unsqueeze(-1)}

    def reduce_func(self, nodes):
        return {'h': torch.prod(nodes.mailbox['h'], dim=1)}

    def forward(self, g):
        g = g.local_var()

        v_feat = g.ndata['h']
        e_var = g.edata[self.var]

        k = self.K_MLP(v_feat)
        q = self.Q_MLP(v_feat)
        v = self.V_MLP(v_feat)

        g.ndata['q'] = q

        if g.edata.get('cutoff') is None:
            bf = self.basis_func(e_var)
            cutoff = self.cutoff(e_var)

            g.edata['bf'] = bf * cutoff
            g.edata['cutoff'] = cutoff

        dV = torch.nn.SiLU(self.dV_MLP(bf))
        dK = torch.nn.SiLU(self.dK_MLP(bf))

        g.edata['dV'] = dV
        g.edata['dK'] = dK

        g.ndata['k'] = k
        g.ndata['v'] = v

        g.update_all(self.cfconv, self.reduce_func)

        y = self.IB_MLP(g.ndata['h'])

        s1 = self.MLP_S1(g.ndata['v'])
        s2 = self.MLP_S2(g.ndata['v'])

        g.ndata['s1'] = s1
        g.ndata['s2'] = s2

        return (y, s1, s2)