import torch.nn as nn
import torch
from datetime import datetime
import os
import argparse
from typing import Union

from sinn.dataset.dataset import FilteredAtomsDataset
from sinn.model.alignn_pyg import Alignn
from sinn.train.train import train_model
from sinn.train.transforms_pyg import PeriodicKNN_PyG, AddNoise
from sinn.train.loss import find_class_weights

def main(model_path: Union[str, None] = None):
    n_atoms = 2
    spg = list(range(195, 231))

    categorical_filter = ([True],['spg_number'],[spg])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    k = 19

    pre_eval_func = PeriodicKNN_PyG(k = k)
    pre_eval_func = AddNoise(std=0.1, transform=pre_eval_func)
    
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
    num_layers = 2

    model = Alignn(num_layers = num_layers, hidden_features = 128, radial_features = 256, out_feats=num_classes, classification=True).to(device)
    model_type_name = type(model).__name__

    if model_path is None:
        date = datetime.now().strftime("%Y-%m-%d")
        model_path = f'models/{date}/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_num = 8
        model_name = f'{model_type_name}-k{k}-L{num_layers}-{target[:3]}{num_classes}-n{model_num}'

    else:
        model_name = model_path.split('/')[-1].split('.')[0]
        model_path = model_path.split(model_name)[0]

    print(model_name)

    loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=10000)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[20])

    train_model(model = model,
                dataset = dataset,
                loss_func = loss_func,
                optimizer = optimizer,
                scheduler=scheduler1,
                n_epochs = 10000,
                batch_size=1,
                model_name=model_name,
                save_path = model_path,
                device = device)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--model_path', type=str, help='Path to the model file', required=False)
    args = parser.parse_args()

    main(args.model_path)