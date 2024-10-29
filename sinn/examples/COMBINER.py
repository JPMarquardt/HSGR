import pickle
import argparse
import torch
from sklearn.decomposition import IncrementalPCA
import os
import pathlib

from sinn.model.combiner import ModelCombiner
from sinn.model.alignn_pyg import Alignn
from sinn.train.transforms_pyg import AperiodicKNN_PyG, PeriodicKNN_PyG


def main(args):
    model_path = args['model_path']

    if model_path.endswith('.pkl'):
        date = model_path.split('/')[-2]
        model_name = model_path.split('/')[-1].split('.')[0]
        model_path = model_path.split(model_name)[0]
        print(model_path)

    elif args.model_path.endswith('/'):
        files = os.listdir(model_path)
        files = [f for f in files if f.endswith('.pkl')]
        args_lsit = [{'model_path': f'{model_path}{f}', 'boundary': args.boundary} for f in files]

        for arg in args_lsit:
            main(arg)
        return


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model = torch.load(f'{model_path}{model_name}.pkl', map_location=device)

    k = int(model_name.split('-')[1].split('k')[1])

    if args['boundary'] == 'periodic':
        pre_eval_func = PeriodicKNN_PyG(k = k)
        periodic = 'p'
    else:
        pre_eval_func = AperiodicKNN_PyG(k = k)
        periodic = 'a'

    if model_name.split('-')[-2][:3] == 'int':
        indices = [-1, -5]
    elif model_name.split('-')[-2][:3] == 'spg':
        indices = [-7, -8]

    for i in range(2):
        #model = Model_Combiner(pre_eval_func=pre_eval_func, model=base_model, pca=pca_model[i].unsqueeze(0))
        model = ModelCombiner(pre_eval_func=pre_eval_func, model=base_model, index=indices[i])
        model.eval()

        if not os.path.exists(f'simulations/{date}'):
            pathlib.Path(f'simulations/{date}').mkdir(parents=True, exist_ok=True)
            
        torch.save(model, f'simulations/{date}/{model_name}-combiner{periodic}{i}.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a directory for umbrella sampling of a model')
    parser.add_argument('model_path', type=str, help='Path to the model file')
    parser.add_argument('--boundary', type=str, default='periodic', help='Boundary conditions for the model')
    args = parser.parse_args()

    main(args)