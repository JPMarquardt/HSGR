import torch

import dgl
import dgl.function as fn
import torch.nn as nn
from sinn.graph.graph_pyg import Graph
from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)

class color_invariant_duplet(nn.Module):
    def __init__(self, in_feats: int = 64):
        super(color_invariant_duplet, self).__init__()

        self.e1 = nn.Embedding(2, in_feats)
    
    def forward(self, g: Graph):
        zab = torch.eq(g['src_z'], g['dst_z'][:, None].expand(-1, g['k']))

        return self.e1(zab.int())
    
class color_invariant_triplet(nn.Module):
    def __init__(self, in_feats: int = 64):
        super(color_invariant_triplet, self).__init__()

        self.e1 = nn.Embedding(2, in_feats)
        self.e2 = nn.Embedding(2, in_feats)
        self.e3 = nn.Embedding(2, in_feats)
    
    def forward(self, g: Graph):
        g, h = g

        za = g['dst_z'][:, None, None].expand(-1, g['k'], g['k'])
        zb = torch.select(h['src_z'], -1, 0)
        zc = torch.select(h['src_z'], -1, 1)

        zab = torch.eq(za, zb)
        zac = torch.eq(za, zc)
        zbc = torch.eq(zb, zc)

        return self.e1(zac.int()) + self.e2(zab.int()) + self.e3(zbc.int())