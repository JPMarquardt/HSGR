import matplotlib.pyplot as plt
import torch

from sinn.model.schnet import SchNet_Multihead

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
names = ['CsCl', 'Th3P4', 'aggr']
shapes = ['o', 's', 'x']
dates = ['24-06-24', '24-06-25']
model_names = ['SchNet-AtomNoise-Spg3-8L', 'SchNet_Multihead-k17-L8-Spg7-n2']
spg_ind = [1, 5]
for i, model_name in enumerate(model_names):
    plt.figure()
    for n, crystal_name in enumerate(names):
        for_plotting = torch.load(f'models/{dates[i]}/{model_name}-{crystal_name}_preds.pkl', map_location=device).squeeze()
        print(for_plotting.shape)
        colors = torch.linspace(0,1,for_plotting.shape[0])
        plt.scatter(for_plotting[:,-1], for_plotting[:,spg_ind[i]], label=crystal_name, marker=shapes[n], c=colors, alpha=0.5, cmap='viridis')
        plt.legend()
    plt.savefig(f'models/{dates[i]}/{model_name}-scatter.png')
