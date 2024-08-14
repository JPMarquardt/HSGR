import torch.nn as nn
import torch
import importlib

from MDAnalysis.coordinates.GSD import GSDReader
from MDAnalysis import Universe

from sinn.dataset.dataset import FilteredAtomsDataset
from sinn.model.alignn_pyg import Alignn
from sinn.train.train import test_model
from sinn.train.transforms_pyg import AperiodicKNN_PyG


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = f'models/24-08-13/'
model_name = f'Alignn-k17-L4-spg22-n7'

k = int(model_name.split('-')[1].split('k')[1])
print(k)
pre_eval_func = AperiodicKNN_PyG(k = k)

model = torch.load(model_path + model_name + '.pkl', map_location=device)

dataset_names = ['CsCl.gsd', 'aggr.gsd', 'Th3P4.gsd']
sparsity = [1000, 1000, 1000]

dims = [0, -1]
n_interpolation_points = 5
mean_tensor = torch.zeros((len(dataset_names), len(dims)))

for n, name in enumerate(dataset_names):
    dataset = Universe(f'./test_traj/{name}')

    dataset = FilteredAtomsDataset(source = dataset,
                                transform=pre_eval_func,
                                target = 'target',
                                sparsity=sparsity[n])
    
    preds = test_model(model = model, 
                    dataset=dataset,
                    device=device,)

    # Stack the predictions that are in the correct state
    preds_save = torch.stack(preds[len(preds)//2:])
    for d, dim in enumerate(dims):
        pred = preds_save[:, dim]
        mean_tensor[n, d] = torch.mean(pred)

print(mean_tensor)
value_list = []

for n in range(len(dataset_names) - 1):
    for i in range(n_interpolation_points):
        value_list.append(mean_tensor[n] + i * (mean_tensor[n+1] - mean_tensor[n]) / n_interpolation_points)
            

print(value_list)