import torch.nn as nn
import torch

from MDAnalysis.coordinates.GSD import GSDReader
from MDAnalysis import Universe

from sinn.dataset.dataset import FilteredAtomsDataset, collate_noise
from sinn.model.model import SchNet, Alignn
from sinn.train.train import train_model, SimulatedNoiseRegressionEval



n_atoms = 2
spg = ('225',)
categorical_filter = ([True],['spg_number'],[spg])

batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 50
pre_eval_func = SimulatedNoiseRegressionEval(k = k)

dataset = Universe('./test_traj/trajectory_LS4_FP0.5_RN105_BL10_DL5.3_Th3P4.gsd')

dataset = FilteredAtomsDataset(source = dataset).dataset

model_name = 'SchNet-AtomNoise-Spg225'
model_path = 'models/24-05-29/'
model = SchNet(num_classes=1, num_layers=3, hidden_features=64, radial_features=128)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model = model,
            dataset = dataset,
            loss_func = loss_func,
            optimizer = optimizer,
            n_epochs = 500,
            batch_size = 1,
            model_name=model_name,
            save_path = model_path,
            device = device)


"""
Scale invariant neural network (SINN) for predicting space group of a crystal structure.
data = load_data(path)
data = filter_data(data)

either
    data = melt/crystalize/gas(data)

    data = manual_onehot_target(data)
or
    data = add_noise(data)
    value = noise(data)

data = graph_construction(data)
train_data, test_data = split_data(data)

model = create_model()
model = train_model(model, train_data)
model = test_model(model, test_data)
"""