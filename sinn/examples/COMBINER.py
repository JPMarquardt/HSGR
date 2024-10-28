import pickle
import argparse
import torch
from sklearn.decomposition import IncrementalPCA

from sinn.model.combiner import ModelCombiner
from sinn.model.alignn_pyg import Alignn
from sinn.train.transforms_pyg import AperiodicKNN_PyG


def main(model_path):
    if model_path.endswith('.pkl'):
        date = model_path.split('/')[-2]
        model_name = model_path.split('/')[-1].split('.')[0]
        model_path = model_path.split(model_name)[0]

    if model_path.endswith('/'):
        date = model_path.split('/')[-3]
        model_name = model_path.split('/')[-1]
        model_path = model_path.split('/')[-2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model = torch.load(f'{model_path}{model_name}.pkl', map_location=device)

    k = int(model_name.split('-')[1].split('k')[1])
    pre_eval_func = AperiodicKNN_PyG(k = k)

    indices = [-7, -8]

    for i in range(2):
        #model = Model_Combiner(pre_eval_func=pre_eval_func, model=base_model, pca=pca_model[i].unsqueeze(0))
        model = ModelCombiner(pre_eval_func=pre_eval_func, model=base_model, index=indices[i])
        model.eval()
        torch.save(model, f'simulations/{date}/{model_name}-combiner{i}.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a directory for umbrella sampling of a model')
    parser.add_argument('model_path', type=str, help='Path to the model file')
    args = parser.parse_args()

    main(args.model_path)