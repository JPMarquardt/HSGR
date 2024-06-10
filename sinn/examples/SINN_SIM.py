import torch.nn as nn
import torch

from MDAnalysis.coordinates.GSD import GSDReader
from MDAnalysis import Universe

from sinn.dataset.dataset import FilteredAtomsDataset
from sinn.model.model import SchNet, Alignn
from sinn.train.train import run_epoch
from sinn.train.transforms import SimulatedNoiseRegressionEval

n_atoms = 2
spg = ('225',)
categorical_filter = ([True],['spg_number'],[spg])

batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 50
pre_eval_func = SimulatedNoiseRegressionEval(k = k)

dataset = Universe('./test_traj/trajectory_LS4_FP0.5_RN105_BL10_DL5.3_Th3P4.gsd')

dataset = FilteredAtomsDataset(source = dataset,
                               transform=pre_eval_func,
                               target = 'target').dataset

model_name = 'SchNet-AtomNoise-Spg225'
model_path = 'models/24-05-29/'
output_dir = f'{model_path}{model_name}.pkl'
model = torch.load(output_dir)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

run_epoch(model = model, 
          loader = dataset, 
          loss_func=loss_func, 
          optimizer = optimizer, 
          device=device, 
          epoch = 0, 
          train=False,)
