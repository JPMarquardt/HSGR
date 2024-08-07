import matplotlib.pyplot as plt
import torch
import pickle

from nfflr.data.dataset import Atoms
from sklearn.decomposition import IncrementalPCA
from sinn.train.train import test_model
from sinn.train.transforms_pyg import PeriodicKNN_PyG
from sinn.dataset.dataset import FilteredAtomsDataset
from sinn.model.alignn_pyg import Alignn
from sinn.model.combiner import Model_Combiner
from MDAnalysis import Universe

def hook_fn(module, input, output):
    forward_hook.append(output)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
names = ['CsCl.gsd', 'Th3P4.gsd', 'aggr.gsd']
dates = ['24-08-06']
model_names = ['Alignn-k17-L4-int5-n7']
sparsity = 1000

for i, model_name in enumerate(model_names):
    for n, crystal_name in enumerate(names):
        data = Universe(f'test_traj/{crystal_name}')
        dataset = FilteredAtomsDataset(source = data,
                                       sparsity=sparsity,
                                        target='target'
                                        ).dataset
        

        transform = PeriodicKNN_PyG(k=17)
        model: Alignn = torch.load(f'models/{dates[i]}/{model_name}.pkl', map_location=device)
        with open(f'models/{dates[i]}/{model_name}-pca.pkl', 'rb') as f:
            pca: IncrementalPCA = pickle.load(f)
            pca = torch.tensor(pca.components_)

        model = Model_Combiner(pre_eval_func=transform, model=model, pca=pca)

        pca_eval = test_model(model, dataset, device)
        

        for j, datapoint in enumerate(dataset):
            for k in range(2):
                frame = j * sparsity
                atoms: Atoms = datapoint[0]
                positions = atoms.positions

                pca_j = pca_eval[j]
                print(torch.max(pca_j[:, k]), torch.min(pca_j[:, k]))
                
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=pca_j[:, k], cmap='viridis', marker='o', alpha=0.2)
                ax.set_title(f'{crystal_name} Frame {frame}')
                plt.savefig(f'models/{dates[i]}/{model_name}/{crystal_name}{k}-{frame}.png')
                plt.close(fig)
