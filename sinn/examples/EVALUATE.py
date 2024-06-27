import torch.nn as nn
import torch
import importlib

from MDAnalysis.coordinates.GSD import GSDReader
from MDAnalysis import Universe

from sinn.dataset.dataset import FilteredAtomsDataset, collate_noise
from sinn.model.schnet import SchNet_Multihead
from sinn.train.train import test_model
from sinn.train.transforms import PeriodicNoiseRegressionEval


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 17
pre_eval_func = PeriodicNoiseRegressionEval(k = k)

model_name = 'SchNet_Multihead-k17-L8-Spg7-n2'
model_path = 'models/24-06-25/'

model = torch.load(model_path + model_name + '.pkl')

dataset_names = ['CsCl', 'Th3P4', 'aggr']
for name in dataset_names:
    dataset = Universe(f'./test_traj/{name}.gsd')

    dataset = FilteredAtomsDataset(source = dataset,
                                transform=pre_eval_func,
                                target = 'target',
                                sparsity=10).dataset


    def hook_fn(module, input, output):
        fc2.append(output)
    model.model.fc.register_forward_hook(hook_fn)

    fc2 = []

    preds = test_model(model = model, 
                    dataset=dataset,
                    device=device,)

    preds = list(map(lambda x: torch.cat(x, dim=1), preds))

    fc_save = torch.stack(fc2, dim=0)
    preds_save = torch.stack(preds)

    torch.save(fc_save, model_path + model_name + f'-{name}_fc2.pkl')
    torch.save(preds_save, model_path + model_name + f'-{name}_preds.pkl')

    fc_save = None
    preds_save = None