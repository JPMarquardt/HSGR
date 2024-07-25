import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sinn.train.transforms import PeriodicClassificationTrain
from sinn.dataset.dataset import FilteredAtomsDataset, collate_multihead_noise


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
names = ['CsCl.gsd', 'Th3P4.gsd', 'aggr.gsd']
shapes = ['o', 's', 'x']
dates = ['24-07-17']
model_names = ['Alignn_Multihead-k17-L8-Spg31-n6']

for i, model_name in enumerate(model_names):
    pca = PCA(n_components=2)
    plt.figure()
    for n, crystal_name in enumerate(names):
        data = torch.load(f'models/{dates[i]}/{model_name}-{crystal_name}_fc2.pkl', map_location=device).squeeze()
        print(data.shape)
        data = data.reshape(-1, data.shape[-1] * data.shape[-2])
        print(data[0])
        print(data[-1])
        print(data.shape)
        data = data.cpu().detach().numpy()
        if n == 0:
            for_plotting = pca.fit_transform(data)
        else:
            for_plotting = pca.transform(data)
        print(for_plotting.shape)
        colors = torch.linspace(0,1,for_plotting.shape[0])
        plt.scatter(for_plotting[:,0], for_plotting[:,1], label=crystal_name, marker=shapes[n], c=colors, alpha=0.5, cmap='viridis')
        plt.legend()
    plt.savefig(f'models/{dates[i]}/{model_name}-scatter.png')
    print(f'models/{dates[i]}/{model_name}-scatter.png')

for i, model_name in enumerate(model_names):
    plt.figure()
    for n, crystal_name in enumerate(names):
        data = torch.load(f'models/{dates[i]}/{model_name}-{crystal_name}_preds.pkl', map_location=device).squeeze()
        for_plotting = data.cpu().detach().numpy()
        print(for_plotting.shape)
        print(for_plotting[0])
        colors = torch.linspace(0,1,for_plotting.shape[0])
        plt.scatter(for_plotting[:,2], for_plotting[:,6], label=crystal_name, marker=shapes[n], c=colors, alpha=0.5, cmap='viridis')
        plt.legend()
    plt.savefig(f'models/{dates[i]}/{model_name}-scatter-preds.png')
    print(f'models/{dates[i]}/{model_name}-scatter-preds.png')