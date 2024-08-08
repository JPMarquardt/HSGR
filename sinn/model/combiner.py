import torch.nn as nn
import torch
from typing import List, Tuple, Callable
from sklearn.decomposition import IncrementalPCA
#from torchmdnet.models.model import TorchMD_Net



class ModelCombiner(nn.Module):
    def __init__(self, pre_eval_func: Callable, model: nn.Module, index: int):
        super(ModelCombiner, self).__init__()
        self.pre_eval_func = pre_eval_func
        self.model = model
        self.output = torch.Tensor()
        self.index = index

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:       
        x = self.pre_eval_func(x)
        self.output = self.model(x, early_return=False)

        return self.output[self.index]

class ModelCombinerPCA(nn.Module):
    def __init__(self, pre_eval_func: Callable, model: nn.Module, pca: torch.Tensor):
        super(ModelCombinerPCA, self).__init__()
        self.pre_eval_func = pre_eval_func
        self.model = model
        self.output = torch.Tensor()

        self.register_buffer('pca_weights', torch.transpose(pca, 0, 1))


    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:       
        x = self.pre_eval_func(x)
        self.output = self.model(x, early_return=True)

        return torch.sum(self.pca_weights[None, :, :] * self.output[:, :, None], dim=1)