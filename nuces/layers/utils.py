import torch

import dgl
import dgl.function as fn
import torch.nn as nn
import numpy as np

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
    

class SmoothCutoffCos(nn.Module):
    def __init__(self, cutoff: float = 1.0):
        super(SmoothCutoffCos, self).__init__()

        self.register_buffer('pi', torch.tensor(np.pi))
        self.register_buffer('cutoff', torch.tensor(cutoff))

    def forward(self, r):
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