import torch.nn as nn
import torch
from datetime import datetime
import os

from sinn.dataset.dataset import FilteredAtomsDataset
from sinn.model.alignn_pyg import Alignn
from sinn.train.train import train_model
from sinn.train.transforms_pyg import PeriodicKNN_PyG
from sinn.train.loss import RegressionClassificationLoss, find_class_weights


n_atoms = 2
spg = list(range(195,231))

categorical_filter = ([True],['spg_number'],[spg])

batch_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 17

pre_eval_func = PeriodicKNN_PyG(k = k)
target = 'spg_number'

dataset = FilteredAtomsDataset(source = "dft_3d",
                        n_unique_atoms = (True,n_atoms),
                        categorical_filter = categorical_filter,
                        target = target,
                        transform=pre_eval_func,
                        )


model_path = f'models/24-08-10/'
model_name = f'Alignn-k17-L4-spg22-n7'

model = torch.load(model_path + model_name + '.pkl', map_location=device)

model_name_new = f'Alignn-k17-L4-spg22-n8'

class_weights = find_class_weights(dataset, target)
num_classes = class_weights.size(0)

loss_func = RegressionClassificationLoss(num_classes=num_classes, class_weights=class_weights, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=500)


train_model(model = model,
            dataset = dataset,
            loss_func = loss_func,
            optimizer = optimizer,
            scheduler=scheduler,
            n_epochs = 500,
            batch_size = batch_size,
            model_name=model_name_new,
            save_path = model_path,
            device = device)