import pickle
import torch
from sklearn.decomposition import IncrementalPCA

from sinn.model.combiner import ModelCombiner
from sinn.model.alignn_pyg import Alignn
from sinn.train.transforms_pyg import AperiodicKNN_PyG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = f'models/24-08-15/'
date = model_path.split('/')[-2]
model_name = f'Alignn-k17-L4-spg22-n7'

base_model = torch.load(f'{model_path}{model_name}.pkl', map_location=device)

k = int(model_name.split('-')[1].split('k')[1])
pre_eval_func = AperiodicKNN_PyG(k = k)

indices = [-7, -8]

for i in range(2):
    #model = Model_Combiner(pre_eval_func=pre_eval_func, model=base_model, pca=pca_model[i].unsqueeze(0))
    model = ModelCombiner(pre_eval_func=pre_eval_func, model=base_model, index=indices[i])
    model.eval()
    torch.save(model, f'simulations/{date}/{model_name}-combiner{i}.pkl')
