import torch.nn as nn
import torch

from sinn.dataset.dataset import FilteredAtomsDataset, collate_multihead_noise
from sinn.model.model import SchNet, Alignn
from sinn.train.train import train_model
from sinn.train.transforms import NoiseRegressionEval

model_name = 'SchNet-AtomNoise-Spg225-1L'
model_path = 'models/24-06-16/'
model = torch.load(f'{model_path}{model_name}.pkl')

n_atoms = 2
spg = (225, 220)
categorical_filter = ([True],['spg_number'],[spg])

batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 17
noise = lambda x: 1 - torch.sqrt(1 - x**2)
pre_eval_func = NoiseRegressionEval(noise = noise, k = k)

def custom_loss_func(output, target):
    classification_pred = output[0]
    classification_target = target[0]

    regression_pred = output[1]
    regression_target = target[1]

    dataset_loss = nn.BCELoss()(classification_pred, classification_target)
    noise_loss = nn.MSELoss()(regression_pred, regression_target)

    penalty = 1 - regression_target
    return torch.mean(dataset_loss * penalty + noise_loss)

dataset = FilteredAtomsDataset(source = "dft_3d",
                        n_unique_atoms = (True,n_atoms),
                        categorical_filter = categorical_filter,
                        target = 'spg_number',
                        transform=pre_eval_func,
                        collate = collate_multihead_noise,
                        ).dataset





loss_func = custom_loss_func
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model = model,
            dataset = dataset,
            loss_func = loss_func,
            optimizer = optimizer,
            n_epochs = 20,
            batch_size = batch_size,
            model_name=model_name,
            save_path = model_path,
            device = device)

train_model(model = model,
            dataset = dataset,
            loss_func = loss_func,
            optimizer = optimizer,
            n_epochs = 5,
            batch_size = batch_size,
            model_name=model_name,
            save_path = model_path,
            device = device,
            swa=True)
        