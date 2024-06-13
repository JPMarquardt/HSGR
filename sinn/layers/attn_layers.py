import torch

import dgl
import dgl.function as fn
import torch.nn as nn

from sinn.layers.utils import SmoothCutoff, MLP, radial_basis_func
from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)

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
                 radial_feats: int = 256,
                 hidden_features: int = 64,
                 out_feats: int = 64, 
                 var: str = 'd', 
                 cutoff: bool = True, 
                 in_range: Tuple[float, float] = None, 
                 **kwargs):
        super(MDNetAttn, self).__init__()
        self.out_feats = out_feats

        #cutoff function
        if cutoff:
            max = in_range[1]
            self.cutoff = SmoothCutoff(cutoff=max)
        else:
            self.cutoff = SmoothCutoff(cutoff=False)

        #key, query, value
        self.K_MLP = MLP(in_feats, hidden_features)
        self.Q_MLP = MLP(in_feats, hidden_features)
        self.V_MLP = MLP(in_feats, hidden_features)

        #dV and dK
        self.dV_MLP = MLP(radial_feats, hidden_features)
        self.dK_MLP = MLP(radial_feats, hidden_features)

        #interaction block
        self.MLP_A = MLP(hidden_features, out_feats * 3)

        #S1 and S2
        self.MLP_S1 = MLP(hidden_features, 1)
        self.MLP_S2 = MLP(hidden_features, 1)


    def forward(self, g):
        g = g.local_var()

        bf = g.edata['bf']
        cutoff = self.cutoff(g.edata['d']).unsqueeze(-1)

        src_feat = g.edata['x_src']
        dst_feat = g.edata['x_dst']

        k = self.K_MLP(src_feat)
        q = self.Q_MLP(dst_feat)
        v = self.V_MLP(src_feat)

        dV = torch.nn.SiLU(self.dV_MLP(bf))
        dK = torch.nn.SiLU(self.dK_MLP(bf))

        a =  torch.nn.SiLU(torch.sum(k * q * dK, dim = 1, keepdim = True)) * cutoff
        vdV =  v * dV

        g.edata['a'] = a * self.MLP_A(vdV)
        g.update_all(fn.copy_e('a', 'm'), fn.sum('m', 'y'))

        q1, q2, q3 = torch.split(g.ndata['y'], self.out_feats, dim = 1)
        s1 = self.MLP_S1(vdV)
        s2 = self.MLP_S2(vdV)

        return (q1, q2, q3, s1, s2)
    
class MDNetUpdate(nn.Module):
    """
    
    """
    def __init__(self, in_feats: int = 64, 
                 radial_feats: int = 256,
                 hidden_features: int = 64,
                 out_feats: int = 64, 
                 var: str = 'd', 
                 cutoff: bool = True, 
                 in_range: Tuple[float, float] = None, 
                 **kwargs):
        super(MDNetUpdate, self).__init__()

        self.attention = MDNetAttn(in_feats, radial_feats, hidden_features, out_feats, var, cutoff, in_range)

        self.U1 = MLP(3, 3)
        self.U2 = MLP(3, 3)
        self.U3 = MLP(3, 3)

    def forward(self, g):
        g = g.local_var()

        #qi in R^F, si in R
        q1, q2, q3, s1, s2 = self.attention(g)

        #delta_x_i in R^3
        v_i = g.ndata['x']
        delta_x_i = q1 + q2 * torch.sum(self.U1(v_i) * self.U2(v_i), dim = 1, keepdim = True)

        #ndx in R^3
        dx = g.edata['dr']
        ndx = dx / torch.norm(dx, dim = 1, keepdim = True)
        ndx = ndx * s2

        #v_j in R^3
        v_j = g.edata['v_src']
        v_j = v_j * s1

        g.edata['w'] = v_j + ndx
        g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'w_i'))

        g.ndata['x'] = g.ndata['w_i'] + q3.unsqueeze(0) * self.U3(v_i)
        g.ndata['v_i'] = v_i + delta_x_i

        return g