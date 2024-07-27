import torch.nn as nn
import torch
import importlib

from MDAnalysis.coordinates.GSD import GSDReader
from MDAnalysis import Universe

from sinn.dataset.dataset import FilteredAtomsDataset, collate_noise
from sinn.model.schnet import SchNet_Multihead
from sinn.train.train import test_model
from sinn.train.transforms import APeriodicNoiseRegressionEval


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = f'models/24-07-25/'
model_name = f'Alignn_Multihead-k17-L8-int5-n7'

k = int(model_name.split('-')[1].split('k')[1])
print(k)
pre_eval_func = APeriodicNoiseRegressionEval(k = k)

model = torch.load(model_path + model_name + '.pkl', map_location=device)

dataset_names = ['expr.xyz', 'CsCl.gsd', 'Th3P4.gsd', 'aggr.gsd']
sparsity = [None, 1000, 1000, 1000]

for n, name in enumerate(dataset_names):
    dataset = Universe(f'./test_traj/{name}')

    dataset = FilteredAtomsDataset(source = dataset,
                                transform=pre_eval_func,
                                target = 'target',
                                sparsity=sparsity[n]).dataset


    def hook_fn(module, input, output):
        fc2.append(output)
    model.model.fc.register_forward_hook(hook_fn)

    fc2 = []
    
    preds = test_model(model = model, 
                    dataset=dataset,
                    device=device,)

    preds = list(map(lambda x: torch.cat(x, dim=1), preds))

    fc_save = fc2 #torch.stack(fc2, dim=0)
    preds_save = preds #torch.stack(preds)

    print(preds_save)
    print(fc_save)
    torch.save(fc_save, model_path + model_name + f'-{name}_fc2.pkl')
    torch.save(preds_save, model_path + model_name + f'-{name}_preds.pkl')

    fc_save = None
    preds_save = None