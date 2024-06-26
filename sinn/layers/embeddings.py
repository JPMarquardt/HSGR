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
        src = edges.src['z']
        dst = edges.dst['z']

        return {'h': src == dst}
    
    def forward(self, g: dgl.DGLGraph):
        g = g.local_var()

        g.apply_edges(self.comparison)
        return self.e1(g.edata['h'].int())
    
class color_invariant_triplet(nn.Module):
    def __init__(self, in_feats: int = 64):
        super(color_invariant_triplet, self).__init__()

        self.e1 = nn.Embedding(2, in_feats)
        self.e2 = nn.Embedding(2, in_feats)
        self.e3 = nn.Embedding(2, in_feats)

    def send_z_src(self, edges):
        """Compute bond angle cosines from bond displacement vectors."""
        # line graph edge: (a, b), (b, c)
        # `a -> b -> c`
        # this sends neighbor information to each edge
        z_src = edges.src['z']
        z_dst = edges.dst['z']

        return {"z_src": z_src, "z_dst": z_dst}
    
    def comparison3(self, edges):
        # line graph edge: (a, b), (b, c)
        # `a -> b -> c`
        # this checks if the atomic number of b is in (a, c) and if c = a
        # then it assigns a a symmetry value from 0 to 3
        za = edges.src['z_src']
        zc = edges.dst['z_dst']
        zb = edges.src['z_dst']

        za_eq_zc = za == zc
        za_eq_zb = za == zb
        zc_eq_zb = zc == zb

        return {'zac': za_eq_zc, 'zab': za_eq_zb, 'zbc': zc_eq_zb}
    
    def forward(self, g: Union[dgl.DGLGraph, Tuple[dgl.DGLGraph, dgl.DGLGraph]]):
        g, h = g

        g = g.local_var()
        h = h.local_var()

        g.apply_edges(self.send_z_src)
        h.ndata['z_src'] = g.edata['z_src']
        h.ndata['z_dst'] = g.edata['z_dst']

        h.apply_edges(self.comparison3)
        zac = h.edata['zac'].int()
        zab = h.edata['zab'].int()
        zbc = h.edata['zbc'].int()
        h.edata['h'] = self.e1(zac) + self.e2(zab) + self.e3(zbc)

        return self.e1(zac) + self.e2(zab) + self.e3(zbc)
    
