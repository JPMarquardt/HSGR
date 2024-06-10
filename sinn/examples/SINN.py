import torch.nn as nn
import torch

from sinn.dataset.dataset import FilteredAtomsDataset, collate_noise
from sinn.model.model import SchNet, Alignn
from sinn.train.train import train_model
from sinn.train.transforms import NoiseRegressionEval



n_atoms = 2
spg = ('225',)
categorical_filter = ([True],['spg_number'],[spg])

batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 17
noise = lambda x: 1 - torch.sqrt(1 - x**2)
pre_eval_func = NoiseRegressionEval(noise = noise, k = k)

dataset = FilteredAtomsDataset(source = "dft_3d",
                        n_unique_atoms = (True,n_atoms),
                        categorical_filter = categorical_filter,
                        transform=pre_eval_func,
                        collate = collate_noise,
                        ).dataset

model_name = 'SchNet-AtomNoise-Spg225'
model_path = 'models/24-06-10/'
model = SchNet(num_classes=1, num_layers=2, hidden_features=64, radial_features=256)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model = model,
            dataset = dataset,
            loss_func = loss_func,
            optimizer = optimizer,
            n_epochs = 100,
            batch_size = batch_size,
            model_name=model_name,
            save_path = model_path,
            device = device)

train_model(model = model,
            dataset = dataset,
            loss_func = loss_func,
            optimizer = optimizer,
            n_epochs = 100,
            batch_size = batch_size,
            model_name=model_name,
            save_path = model_path,
            device = device,
            swa=True)
        


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