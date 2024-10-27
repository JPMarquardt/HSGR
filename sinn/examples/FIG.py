import os
import argparse

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sinn.train.transforms_pyg import PeriodicKNN_PyG
from sinn.dataset.dataset import FilteredAtomsDataset

class DataPlotter():
    def __init__(self, path):
        if not path.endswith('.pkl'):
            raise ValueError('Invalid path')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.paths = path
        self.model = '-'.join(path.split('/')[-1].split('_')[0].split('-')[:-2])
        self.date = path.split('/')[-2]
        self.target = path.split('/')[-1].split('_')[0].split('-')[-1]
        self.data_type = path.split('/')[-1].split('_')[1].split('.')[0]

        self.data: torch.Tensor = torch.load(path, map_location=device)

        if self.data_type == 'fc2':
            self.data = self.data.squeeze()
            self.data = self.data.reshape(-1, self.data.shape[-1] * self.data.shape[-2])
            self.data = self.data.cpu().detach().numpy()

        self.idenfier = ( self.date, self.model, self.data_type)


    def set_figure(self, figure: plt.Figure):
        self.figure = figure

    def set_pca(self, pca: PCA):
        self.pca = pca



def main(paths):
    target_shape_map = {
        'CsCl.gsd': 'o',
        'Th3P4.gsd': 's',
        'aggr.gsd': 'x'
    }
    target_label_map = {
        'CsCl.gsd': 'CsCl',
        'Th3P4.gsd': 'Th3P4',
        'aggr.gsd': 'Aggregate'
    }

    new_paths = []
    for path in paths:
        if path.endswith('/'):
            folder_path = os.path.dirname(path)
            for file in os.listdir(folder_path):
                if file.endswith('fc2.pkl') or file.endswith('preds.pkl'):
                    paths.append(os.path.join(folder_path, file))
        elif path.endswith('.pkl'):
            new_paths.append(path)
        else:
            print('Invalid path')


    data = [DataPlotter(path) for path in new_paths]

    plotted: dict[tuple[str], DataPlotter] = {}

    for i, data in enumerate(data):
        if data.idenfier not in plotted:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            data.set_figure(fig)

            plotted[data.idenfier] = data

            if data.data_type == 'fc2':
                pca = PCA(n_components=2)
                data.data = pca.fit_transform(data.data)

                data.set_pca(pca)

            else:
                ax.set_xlabel('43m Character')
                ax.set_ylabel('m3m Character')
        else:
            fig = plotted[data.idenfier].figure
            ax = fig.get_axes()[0]

            if data.data_type == 'fc2':
                data.data = plotted[data.idenfier].pca.transform(data.data)

        colors = torch.linspace(0,1,data.data.shape[0])
        ax.scatter(data.data[:,0], data.data[:,-1], label=target_label_map[data.target], marker=target_shape_map[data.target], c=colors, alpha=0.5, cmap='viridis')

    for data in plotted.values():
        data.figure.legend(prop={'size': 20})
        data.figure.savefig(f'models/{data.date}/{data.model}-{data.data_type}-scatter.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset')
    parser.add_argument('paths', type=str, nargs='+', help='Paths to the model files')

    args = parser.parse_args()
    main(args.paths)