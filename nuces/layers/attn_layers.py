import torch

import dgl
import dgl.function as fn
import torch.nn as nn
from utils import SmoothCutoff, MLP, radial_basis_func

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
    def __init__(self, in_feats: int = 64, out_feats: int = 64, var: str = 'd', **kwargs):
        super(MDNetEmbed, self).__init__()
        self.var = var

        #cutoff function
        if cutoff in kwargs:
            cutoff = kwargs['cutoff']
            self.max = cutoff.cutoff
            self.cutoff = SmoothCutoff(cutoff=self.max)
        else:
            self.cutoff = SmoothCutoff()
            self.max = 1.0

        #input range
        if var == 'd':
            in_range = (0, self.cutoff)
        elif var == 'angle':
            in_range = (-self.cutoff, self.cutoff)

        #initialize radial basis function
        self.basis_func = radial_basis_func(in_feats, in_range)
        
        #filter generation network
        self.FGN_MLP1 = MLP(in_feats, in_feats)
        self.FGN_MLP2 = MLP(in_feats, in_feats)

        #interaction block
        self.IB_MLP = MLP(in_feats, out_feats)

    def forward(self, g):
        return


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
    def __init__(self, in_feats: int = 64, hidden_feats: int = 64, out_feats: int = 64, var: str = 'd', **kwargs):
        super(MDNetAttn, self).__init__()
        self.var = var

        #cutoff function
        if cutoff in kwargs:
            cutoff = kwargs['cutoff']
            self.max = cutoff.cutoff
            self.cutoff = SmoothCutoff(cutoff=self.max)
        else:
            self.cutoff = SmoothCutoff()
            self.max = 1.0

        #input range
        if var == 'd':
            in_range = (0, self.cutoff)
        elif var == 'angle':
            in_range = (-self.cutoff, self.cutoff)


        
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
        weight = torch.nn.SiLU(torch.sum(k * q, dim=-1)) * self.scale

        value = edges.src['v']
        value = value * weight.unsqueeze(-1)

        return {'h': value}

    def reduce_func(self, nodes):
        return {'h': torch.prod(nodes.mailbox['h'], dim=1)}

    def forward(self, g):
        g = g.local_var()

        v_feat = g.ndata['h']
        e_feat = g.edata['feat'] #???
        e_var = g.edata[self.var]

        k = self.K_MLP(v_feat)
        q = self.Q_MLP(v_feat)
        v = self.V_MLP(v_feat)

        g.ndata['q'] = q

        if g.edata.get('cutoff') is not None:
            bf = g.edata['bf']
            cutoff = g.edata['cutoff']
        else:
            bf = self.basis_func(e_var)
            cutoff = self.cutoff(e_var)


        dV = torch.nn.SiLU(self.dV_MLP(bf) * cutoff)
        dK = torch.nn.SiLU(self.dK_MLP(bf) * cutoff)

        g.ndata['k'] = k * dK
        g.ndata['v'] = v * dV

        self.scale = cutoff / torch.sqrt(v_feat.size(-1))

        g.update_all(self.cfconv, self.reduce_func)
        y = self.IB_MLP(g.ndata['h'])

        s1 = self.MLP_S1(g.ndata['v'])
        s2 = self.MLP_S2(g.ndata['v'])

        g.ndata['s1'] = s1
        g.ndata['s2'] = s2

        return (y, s1, s2)