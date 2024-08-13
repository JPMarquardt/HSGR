import matplotlib.pyplot as plt
import torch
import pickle

from sklearn.decomposition import IncrementalPCA
from sinn.train.train import test_model
from sinn.train.transforms_pyg import AperiodicKNN_PyG
from sinn.dataset.dataset import FilteredAtomsDataset
from sinn.model.alignn_pyg import Alignn
from sinn.model.combiner import ModelCombiner
from MDAnalysis import Universe



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
names = ['CsCl.gsd', 'Th3P4.gsd', 'aggr.gsd']
model_path = f'models/24-08-10/'
model_name = f"Alignn-k17-L4-spg22-n7"
indices = [-4, -7, -8]
sparsity = 1000

for n, crystal_name in enumerate(names):
    data = Universe(f'test_traj/{crystal_name}')
    dataset = FilteredAtomsDataset(source = data,
                                    sparsity=sparsity,
                                    target='target'
                                    )
    

    transform = AperiodicKNN_PyG(k=17)
    model: Alignn = torch.load(f'{model_path}{model_name}.pkl', map_location=device)
    """
    with open(f'models/{dates[i]}/{model_name}-pca.pkl', 'rb') as f:
        pca: IncrementalPCA = pickle.load(f)
        pca = torch.tensor(pca.components_)
        """
    

    for k in range(len(indices)): 
        model_c = ModelCombiner(pre_eval_func=transform, model=model, index=indices[k])
        pca_eval = test_model(model_c, dataset, device)

        

        for j, datapoint in enumerate(dataset):


            frame = j * sparsity
            atoms: dict = datapoint[0]
            positions = atoms['positions']

            print(pca_eval[j].shape)
            pca_j = pca_eval[j]
            print(torch.max(pca_j), torch.min(pca_j))
            print(torch.mean(pca_j), torch.std(pca_j))
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=pca_j, cmap='viridis', marker='o', alpha=0.2)
            ax.set_title(f'{crystal_name} Frame {frame}')
            plt.savefig(f'{model_path}{model_name}/{crystal_name}{k}-{frame}.png')
            plt.close(fig)
