import torch.nn as nn
import torch
from datetime import datetime
import os

from sinn.dataset.dataset import FilteredAtomsDataset
from sinn.model.alignn_pyg import Alignn
from sinn.train.train import train_model
from sinn.train.transforms_pyg import PeriodicKNN_PyG
from sinn.train.loss import find_class_weights

date = datetime.now().strftime("%y-%m-%d")

n_atoms = 2
spg = list(range(195, 231))

categorical_filter = ([True],['spg_number'],[spg])

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

class_weights = find_class_weights(dataset, target)
class_weights = class_weights.to(device)
print(class_weights)

num_classes = class_weights.size(0)
num_layers = 4

model = Alignn(num_layers = num_layers, hidden_features = 128, radial_features = 256, out_feats=num_classes, classification=True).to(device)
model_type_name = type(model).__name__

model_path = f'models/{date}/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

model_num = 8
model_name = f'{model_type_name}-k{k}-L{num_layers}-{target[:3]}{num_classes}-n{model_num}'
print(model_name)

loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=10)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[10])

train_model(model = model,
            dataset = dataset,
            loss_func = loss_func,
            optimizer = optimizer,
            scheduler=scheduler,
            n_epochs = 100,
            batch_size=1,
            model_name=model_name,
            save_path = model_path,
            device = device)
        