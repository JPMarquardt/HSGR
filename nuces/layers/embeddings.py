import torch

import dgl
import dgl.function as fn
import torch.nn as nn
from utils import SmoothCutoff, MLP, radial_basis_func
from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)

def compute_delta_radius(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # 
    r1 = -edges.src["nr"]
    r2 = edges.dst["nr"]
    r1 = torch.norm(r1, dim=1)
    r2 = torch.norm(r2, dim=1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))

    delta_radius = r2-r1

    return {"dnr": delta_radius}

def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))

    return {"h": bond_cosine}

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