import torch

import dgl
import dgl.function as fn
import torch.nn as nn
from sinn.layers.utils import SmoothCutoff, MLP, radial_basis_func
from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)

"""
class nplet_embedding(nn.Module):

create a dictionary on every node.
add the color of that node to the dictionary
with a value of n and (in the original case 0)
*
transfer that node's dictionary to its edges and add the
destination node's color to the dictionary with a value of n+1
turn this graph into an edge graph and repeat
from * with the new node being the previous edge
and the new edge being the previous triplet
"""

class color_invariant_duplet(nn.Module):
    def __init__(self, in_feats: int = 64):
        super(color_invariant_duplet, self).__init__()

        self.e1 = nn.Embedding(2, in_feats)

    def comparison(self, edges):
        src = edges.src['h']
        dst = edges.dst['h']

        return {'h': src == dst}
    
    def forward(self, g):
        g = g.local_var()

        g.apply_edges(self.comparison)

        return self.e1(g.edata['h'])
    
class color_invariant_triplet(nn.Module):
    def __init__(self, in_feats: int = 64):
        super(color_invariant_triplet, self).__init__()

        self.e1 = nn.Embedding(2, in_feats)
        self.e2 = nn.Embedding(2, in_feats)
        self.e3 = nn.Embedding(2, in_feats)

    def send_h_src(self, edges):
        """Compute bond angle cosines from bond displacement vectors."""
        # line graph edge: (a, b), (b, c)
        # `a -> b -> c`
        # this sends neighbor information to each edge
        k_src = edges.src['h']
        k_dst = edges.dst['h']

        return {"h_src": k_src, "h_dst": k_dst}
    
    def comparison3(self, edges):
        # line graph edge: (a, b), (b, c)
        # `a -> b -> c`
        # this checks if the atomic number of b is in (a, c) and if c = a
        # then it assigns a a symmetry value from 0 to 3
        ha = edges.src['h_src']
        hc = edges.dst['h_dst']
        hb = edges.src['h_dst']

        ha_eq_hc = ha == hc
        ha_eq_hb = ha == hb
        hc_eq_hb = hc == hb

        return {'hac': ha_eq_hc, 'hab': ha_eq_hb, 'hbc': hc_eq_hb}
    
    def forward(self, g):
        g, h = g

        g = g.local_var()
        h = h.local_var()

        g.apply_edges(self.send_h_src)
        h.ndata['h_src'] = g.edata['h_src']
        h.ndata['h_dst'] = g.edata['h_dst']

        h.apply_edges(self.comparison3)
        hac = h.edata['hac']
        hab = h.edata['hab']
        hbc = h.edata['hbc']

        return self.e1(hac) + self.e2(hab) + self.e3(hbc)

class SchnetEmbedding(nn.Module):
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
        super(SchnetEmbedding, self).__init__()
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
        edge_feat = edges.data['h']

        cutoff = edges.data['cutoff']
        bf = edges.data['bf']

        return {'h': edge_feat * bf * cutoff.unsqueeze(-1)}

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