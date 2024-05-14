import torch

import dgl
import dgl.function as fn
import torch.nn as nn

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
    def __init__(self, onset: float = 0.8, cutoff: float = 1.0):
        super(SmoothCutoff, self).__init__()
        self.onset = onset
        self.cutoff = cutoff
        self.onset2 = onset ** 2
        self.cutoff2 = cutoff ** 2

    def forward(self, r):
        v = torch.zeros_like(r)
        r2 = r ** 2
        lto = r2 < self.onset2
        ltc = r2 < self.cutoff2
        margin = lto.logical_not() & ltc

        v = torch.where(lto, torch.tensor(1.0), torch.tensor(0.0))
        rm = (r[margin].abs() - self.onset) / (self.cutoff - self.onset)
        v[margin] = -rm**3 * (rm * (6.0 * rm - 15.0) + 10.0) + 1

        return v
    