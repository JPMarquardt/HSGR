import dgl
import torch

import torch.nn as nn
import torch.nn.functional as F
from nuces.layers.layers import SchnetConv

# Create a GNN model with modular layers and hidden features
"""
kwargs = {range: tuple[int] = (0, 1), cutoff: tuple[int] = (0.8, 1.0)}
"""

class GNN(nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden_features, **kwargs):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        self.kwargs = kwargs

        self.layers.append(SchnetConv(num_features, hidden_features, **kwargs))
        for _ in range(num_layers - 1):
            self.layers.append(SchnetConv(hidden_features, hidden_features))
        self.fc = nn.Linear(hidden_features, num_classes)

    def forward(self, g, x):
        for layer in self.layers:
            x = F.relu(layer(g, x))
        x = dgl.mean_nodes(g, x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

