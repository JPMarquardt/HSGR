import torch.nn as nn
import torch
from typing import List, Tuple, Callable
from sklearn.decomposition import IncrementalPCA



class Model_Combiner(nn.Module):
    def __init__(self, pre_eval_func: Callable, model: nn.Module, pca: torch.Tensor):
        super(Model_Combiner, self).__init__()
        self.pre_eval_func = pre_eval_func
        self.model = model
        model.model.fc.register_forward_hook(self.hook_fn)

        self.register_buffer('pca_weights', pca)

    def hook_fn(self, module, input, output):
        self.output = output

    def forward(self, x):
        x = self.pre_eval_func(x)
        x = self.model(x)
        
        return torch.sum(self.pca_weights * self.output)