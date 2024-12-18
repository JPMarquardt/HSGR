import torch.nn as nn
import torch
import math
import os
import yaml
import shutil
import importlib

import argparse

from MDAnalysis.coordinates.GSD import GSDReader
from MDAnalysis import Universe

from sinn.dataset.dataset import FilteredAtomsDataset
from sinn.model.alignn_pyg import Alignn
from sinn.train.train import test_model
from sinn.train.transforms_pyg import AperiodicKNN_PyG, PeriodicKNN_PyG

def main(model_path):
    model_name = model_path.split('/')[-1].split('.')[0]
    model_path = model_path.split(model_name)[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    k = int(model_name.split('-')[1].split('k')[1])
    pre_eval_func = AperiodicKNN_PyG(k = k)

    model = torch.load(model_path + model_name + '.pkl', map_location=device)

    dataset_names = ['CsCl.gsd', 'aggr.gsd', 'Th3P4.gsd']
    sparsity = [1000, 1000, 1000]


    pos_dims = [15, 14]
    neg_dims = [None, None]
    n_interpolation_points = 5
    mean_tensor = torch.zeros((len(dataset_names), len(pos_dims)))

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
        for d in range(len(pos_dims)):
            if pos_dims[d] == 'all':
                pos_pred = preds_save
            else:
                pos_pred = torch.index_select(preds_save, 1, torch.tensor(pos_dims[d], device=device))
            if neg_dims[d] is None:
                pred = torch.mean(pos_pred)
            elif neg_dims[d] == 'all':
                neg_pred = preds_save
                pred = torch.mean(pos_pred) - torch.mean(neg_pred)
            else:
                neg_pred = torch.index_select(preds_save, 1, torch.tensor(neg_dims[d], device=device))
                pred = torch.mean(pos_pred) - torch.mean(neg_pred)
            mean_tensor[n, d] = pred

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a directory for umbrella sampling of a model')
    parser.add_argument('model_path', type=str, help='Path to the model file')
    args = parser.parse_args()

    main(args.model_path)