import torch.nn as nn
import torch

from sinn.dataset.dataset import FilteredAtomsDataset, collate_multihead_noise
from sinn.model.alignn import Alignn_Multihead
from sinn.train.train import train_model
from sinn.train.transforms import NoiseRegressionTrain
from sinn.train.loss import RegressionClassificationLoss, find_class_weights

n_atoms = 2
spg = list(range(221,231))
categorical_filter = ([True],['spg_number'],[spg])

batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 17
noise = lambda: 1 - torch.sqrt(1 - torch.rand(1)**2)
pre_eval_func = NoiseRegressionTrain(noise = noise, k = k)

dataset = FilteredAtomsDataset(source = "dft_3d",
                        n_unique_atoms = (True,n_atoms),
                        categorical_filter = categorical_filter,
                        target = 'spg_number',
                        transform=pre_eval_func,
                        collate = collate_multihead_noise,
                        ).dataset

class_weights = find_class_weights(dataset, 'spg_number')
n_classes = class_weights.size(0)
print(class_weights)

num_layers = 2

model = Alignn_Multihead(num_classes = n_classes, num_layers = num_layers, hidden_features = 64, radial_features = 256)
model_type_name = type(model).__name__

model_name = f'{model_type_name}-k{k}-L{num_layers}-Spg{n_classes}'
model_path = 'models/24-06-16/'

loss_func = RegressionClassificationLoss(num_classes=n_classes, class_weights=class_weights, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=50)

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
            n_epochs = 20,
            batch_size = batch_size,
            model_name=model_name,
            save_path = model_path,
            device = device,
            swa=True)
        