import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from MDAnalysis import Universe

from sinn.dataset.dataset import FilteredAtomsDataset
from sinn.model.model import SchNet, Alignn
from sinn.train.train import test_model
from sinn.train.transforms import SimulatedNoiseRegressionEval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 17
pre_eval_func = SimulatedNoiseRegressionEval(k = k)

dataset = Universe('./test_traj/trajectory_LS4_FP0.5_RN105_BL10_DL5.3_Th3P4.gsd', topology_format='GSD')

dataset = FilteredAtomsDataset(source = dataset,
                               transform=pre_eval_func,
                               target = 'target').dataset

fc2 = []
def hook_fn(module, input, output):
    fc2.append(output)

model_name = 'SchNet-AtomNoise-Spg225'
model_path = 'models/24-06-10/'
output_dir = f'{model_path}{model_name}.pkl'
model = torch.load(output_dir)
model.fc2.register_forward_hook(hook_fn)


loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

pred_list = test_model(model = model, 
                       dataset=dataset,
                       device=device,)


plt.plot(pred_list)
plt.savefig(f'{model_path}{model_name}_Th3P4.png')
