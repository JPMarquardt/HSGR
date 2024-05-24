import torch

import dgl
import dgl.function as fn
import torch.nn as nn
import numpy as np

from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)

class MLP(nn.Module):
    def __init__(self, in_feats: int = 64, out_feats: int = 64):
        super(MLP, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.norm = nn.LayerNorm(out_feats)

    def forward(self, input):
        h = self.linear(input)
        h = self.norm(h)
        h = nn.functional.silu(h)

        return h
    

class SmoothCutoff(nn.Module):
    def __init__(self, cutoff: float = 1.0):
        super(SmoothCutoff, self).__init__()

        self.register_buffer('pi', torch.tensor(np.pi))
        self.register_buffer('cutoff', torch.tensor(cutoff))

        if not cutoff:
            self.early_return = True
        else:
            self.early_return = False

    def forward(self, r):
        if self.early_return:
            return torch.ones_like(r)

        cutoff2 = self.cutoff ** 2

        v = torch.zeros_like(r)
        r2 = r ** 2
        aboveCutOff = r2 > cutoff2
        belowCutOff = ~aboveCutOff

        rm = (r[belowCutOff].abs() - self.onset) / (self.cutoff - self.onset)

        v[aboveCutOff] = 0.0
        v[belowCutOff] = 0.5 * (1 + torch.cos(rm * self.pi))

        #smoothstep functions are marginally faster than cos
        #but they are not as smooth as cos
        #v[margin] = -rm**3 * (rm * (6.0 * rm - 15.0) + 10.0) + 1

        return v
    

class radial_basis_func(nn.Module):
    def __init__(self, in_feats: int = 64, in_range: Tuple[float, float] = None, **kwargs):
        super(radial_basis_func, self).__init__()

        #basis function parameters
        self.register_buffer('gamma', in_feats / (in_range[1] - in_range[0]))
        self.register_buffer('muk', torch.linspace(in_range[0], in_range[1], in_feats))

    def forward(self, dist):
        return torch.exp(-self.gamma * (dist - self.muk)**2)
    

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