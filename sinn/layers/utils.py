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

    def forward(self, r: torch.Tensor):
        if self.early_return:
            return torch.ones_like(r)

        cutoff2 = self.cutoff ** 2

        v = torch.zeros_like(r)
        r2 = r ** 2
        aboveCutOff = r2 > cutoff2
        belowCutOff = ~aboveCutOff

        rm = r[belowCutOff].abs() / self.cutoff

        v[aboveCutOff] = 0.0
        v[belowCutOff] = 0.5 * (1 + torch.cos(rm * self.pi))

        #smoothstep functions are marginally faster than cos
        #but they are not as smooth as cos
        #v[margin] = -rm**3 * (rm * (6.0 * rm - 15.0) + 10.0) + 1

        return v
    

class radial_basis_func(nn.Module):
    def __init__(self, in_feats: int = 64, out_feats: int = 64, in_range: Tuple[float, float] = None, **kwargs):
        super(radial_basis_func, self).__init__()

        #basis function parameters
        self.register_buffer('gamma', torch.tensor(in_feats / (in_range[1] - in_range[0])))
        self.register_buffer('muk', torch.linspace(in_range[0], in_range[1], in_feats))

        #MLP for radial basis function
        self.MLP = MLP(in_feats, out_feats)

    def forward(self, r: torch.Tensor):
        muk = self.muk.unsqueeze(torch.tensor((None, ) * r.dim()))
        exp = torch.exp(-self.gamma * (r.unsqueeze(-1) - muk)**2)
        return MLP(exp)
    

