import pickle
import torch
from sklearn.decomposition import IncrementalPCA

from sinn.model.combiner import Model_Combiner
from sinn.model.alignn import Alignn_Multihead
from sinn.train.transforms import APeriodicClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'Alignn_Multihead-k17-L8-int5-n7'
model_path = 'models/24-07-25/'
model_str = model_path + model_name

with open(f'{model_str}-pca.pkl', 'rb') as f:
    pca: IncrementalPCA = pickle.load(f)

base_model = torch.load(f'{model_str}.pkl', map_location=device)

k = int(model_name.split('-')[1].split('k')[1])
pre_eval_func = APeriodicClassification(k = k)

for i in range(2):
    pca_model = torch.tensor(pca.components_)
    print(pca_model[i].unsqueeze(0).shape)
    model = Model_Combiner(pre_eval_func=pre_eval_func, model=base_model, pca=pca_model[i].unsqueeze(0))
    model.eval()
    torch.save(model, f'{model_str}-combiner{i}.pkl')