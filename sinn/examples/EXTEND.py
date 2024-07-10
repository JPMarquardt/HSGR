import torch.nn as nn
import torch
from datetime import datetime
import os

from sinn.dataset.dataset import FilteredAtomsDataset, collate_multihead_noise
from sinn.model.alignn import Alignn_Multihead
from sinn.train.train import train_model
from sinn.train.transforms import NoiseRegressionTrain, PeriodicClassificationTrain
from sinn.train.loss import RegressionClassificationLoss, find_class_weights


n_atoms = 2
spg = list(range(195,231))

categorical_filter = ([True],['spg_number'],[spg])

batch_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 17

pre_eval_func = PeriodicClassificationTrain(k = k)
target = 'international_number'

dataset = FilteredAtomsDataset(source = "dft_3d",
                        n_unique_atoms = (True,n_atoms),
                        categorical_filter = categorical_filter,
                        target = target,
                        transform=pre_eval_func,
                        collate = collate_multihead_noise,
                        ).dataset


model_path = f'models/24-07-09/'
model_name = f'Alignn_Multihead-k17-L8-Spg5-n6'

model = torch.load(model_path + model_name + '.pkl', map_location=device)

class_weights = find_class_weights(dataset, target)
num_classes = class_weights.size(0)

loss_func = RegressionClassificationLoss(num_classes=num_classes, class_weights=class_weights, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=1000)


train_model(model = model,
            dataset = dataset,
            loss_func = loss_func,
            optimizer = optimizer,
            scheduler=scheduler,
            n_epochs = 1000,
            batch_size = batch_size,
            model_name=model_name,
            save_path = model_path,
            device = device)