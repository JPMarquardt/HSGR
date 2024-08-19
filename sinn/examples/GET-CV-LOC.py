import torch.nn as nn
import torch
import math
import importlib

from MDAnalysis.coordinates.GSD import GSDReader
from MDAnalysis import Universe

from sinn.dataset.dataset import FilteredAtomsDataset
from sinn.model.alignn_pyg import Alignn
from sinn.train.train import test_model
from sinn.train.transforms_pyg import AperiodicKNN_PyG, PeriodicKNN_PyG


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = f'models/24-08-19/'
model_name = f'Alignn-k17-L4-spg22-n8'

k = int(model_name.split('-')[1].split('k')[1])
print(k)
pre_eval_func = AperiodicKNN_PyG(k = k)

model = torch.load(model_path + model_name + '.pkl', map_location=device)

dataset_names = ['CsCl.gsd', 'aggr.gsd', 'Th3P4.gsd']
sparsity = [1000, 1000, 1000]

dims = [-7, -8]
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
    print(preds_save.mean(dim=0))
    for d, dim in enumerate(dims):
        pred = preds_save[:, dim]
        mean_tensor[n, d] = torch.mean(pred)

print(mean_tensor)
center_list = []

for n in range(len(dataset_names) - 1):
    for i in range(n_interpolation_points):
        center_list.append(mean_tensor[n] + i * (mean_tensor[n+1] - mean_tensor[n]) / n_interpolation_points)
center_list.append(mean_tensor[-1])

center_list_new = list(center_list)
center_list_new.append(mean_tensor[-1])
center_list_new.insert(0, mean_tensor[0])

variance_list = []

for i in range(len(center_list)):
    variance_list.append(center_list_new[i+2] - center_list_new[i])

variance_list = [1e4/(2*torch.sqrt(torch.abs(v))) for v in variance_list]

first_frame = []
for i in range(len(dataset_names)):
    n_starts = math.floor(len(center_list)/(len(dataset_names)))
    for j in range(n_starts):
        first_frame.append(dataset_names[i].split('.')[0])

for i in range(len(center_list) - len(first_frame)):
    first_frame.append(dataset_names[-1].split('.')[0])


parameter_list = []
for i in range(len(center_list)):
    parameter_list.append({'model_name': f'../{model_name}', 
                          'center': center_list[i].tolist(), 
                          'scale_factor': variance_list[i].tolist(),
                          })
    print(parameter_list[i])



import os
import yaml
import shutil

dirname = os.path.dirname(__file__)
umbrella_path = os.path.join(dirname, f'simulations/{model_path.split("/")[-2]}/')

if not os.path.exists(umbrella_path):
    os.mkdir(umbrella_path)
    print('Created umbrella path')

for i, parameters in enumerate(parameter_list):
    if not os.path.exists(os.path.join(umbrella_path, f'umbrella_{i}')):
        os.mkdir(os.path.join(umbrella_path, f'umbrella_{i}'))

    yaml.dump(parameters, open(os.path.join(umbrella_path, f'umbrella_{i}/bias.yaml'), 'w'))
    shutil.copyfile(os.path.join(dirname, f'simulations/firstframe_{first_frame[i]}.xyz'), os.path.join(umbrella_path, f'umbrella_{i}/firstframe.xyz'))
    shutil.copyfile(os.path.join(dirname, 'simulations/umbrella.sbatch'), os.path.join(umbrella_path, f'umbrella_{i}/run.sbatch'))
