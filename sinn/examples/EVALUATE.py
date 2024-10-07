import torch.nn as nn
import torch
import importlib
import argparse

from MDAnalysis.coordinates.GSD import GSDReader
from MDAnalysis import Universe

from sinn.dataset.dataset import FilteredAtomsDataset
from sinn.model.alignn_pyg import Alignn
from sinn.train.train import test_model
from sinn.train.transforms_pyg import AperiodicKNN_PyG
import os

def main(paths):
    new_paths = []
    for path in paths:
        if not path.endswith('.pkl'):
            folder_path = os.path.dirname(path)
            for file in os.listdir(folder_path):
                if file.endswith('.pkl'):
                    new_paths.append(os.path.join(folder_path, file))
        else:
            new_paths.append(path)

    paths = new_paths

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_names = [path.split('/')[-1].split('.')[0] for path in paths]
    dates = [path.split('/')[-2] for path in paths]

    for i, model_name in enumerate(model_names):
        model_path = f'models/{dates[i]}/'
        k = int(model_name.split('-')[1].split('k')[1])
        print(k)
        pre_eval_func = AperiodicKNN_PyG(k = k)

        model = torch.load(model_path + model_name + '.pkl', map_location=device)

        dataset_names = ['CsCl.gsd', 'Th3P4.gsd', 'aggr.gsd']
        sparsity = [100, 100, 100]

        for n, name in enumerate(dataset_names):
            dataset = Universe(f'./test_traj/{name}')

            dataset = FilteredAtomsDataset(source = dataset,
                                        transform=pre_eval_func,
                                        target = 'target',
                                        sparsity=sparsity[n])
            

            def hook_fn(module, input, output):
                fc2.append(output)
            model.fc.register_forward_hook(hook_fn)

            fc2 = []
            
            preds = test_model(model = model, 
                            dataset=dataset,
                            device=device,)

            fc_save = torch.stack(fc2, dim=0)
            preds_save = torch.stack(preds)

            print(preds_save.shape)
            print(fc_save.shape)

            torch.save(fc_save, model_path + model_name + f'-{name}_fc2.pkl')
            torch.save(preds_save, model_path + model_name + f'-{name}_preds.pkl')

            fc_save = None
            preds_save = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset')
    parser.add_argument('paths', type=str, nargs='+', help='Paths to the model files')

    args = parser.parse_args()
    main(args.paths)