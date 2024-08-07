import matplotlib.pyplot as plt
import torch
import pickle
from tqdm import tqdm

from MDAnalysis import Universe
from sklearn.decomposition import IncrementalPCA
from sinn.train.transforms_pyg import PeriodicKNN_PyG
from sinn.dataset.dataset import FilteredAtomsDataset, collate_multihead_noise
from sinn.train.train import test_model

n_atoms = 2
spg = list(range(195, 231))

categorical_filter = ([True],['spg_number'],[spg])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 17

pre_eval_func = PeriodicKNN_PyG(k = k)
target = 'international_number'

dataset = FilteredAtomsDataset(source = "dft_3d",
                        n_unique_atoms = (True,n_atoms),
                        categorical_filter = categorical_filter,
                        target = target,
                        transform=pre_eval_func,
                        collate = collate_multihead_noise,
                        ).dataset

fc2 = []

model_name = f'Alignn-k17-L4-int5-n7'
model_path = f'models/24-08-06/'

model = torch.load(model_path + model_name + '.pkl', map_location=device)

def hook_fn(module, input, output):
    fc2.append(output)
model.fc.register_forward_hook(hook_fn)

pca = IncrementalPCA(n_components=2)

test_model(model = model,
            dataset = dataset,
            device = device,
            )

for datapoint in tqdm(fc2):
    pca.partial_fit(datapoint.cpu().detach().numpy())

with open(model_path + model_name + '-pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
