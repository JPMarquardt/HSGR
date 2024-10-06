import torch.nn as nn
import torch
from datetime import datetime
import os
import argparse
from typing import Union
import yaml

from sinn.dataset.dataset import FilteredAtomsDataset
from sinn.model.alignn_pyg import Alignn
from sinn.train.train import train_model
from sinn.train.transforms_pyg import PeriodicKNN_PyG, AddNoise
from sinn.train.loss import find_class_weights

def main(yaml_path: Union[str, None] = None):
    with open(file=yaml_path, mode='r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    dataset_modifications = config['dataset_modifications']
    model_parameters = config['model_parameters']
    training_parameters = config['training_parameters']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset parameters

    if 'n_atoms' in dataset_modifications:
        n_atoms = dataset_modifications['n_atoms']
        n_unique_atoms = (True, n_atoms)
    else:
        n_unique_atoms = (False, None)

    if 'spg_range' in dataset_modifications:
        spg_range = dataset_modifications['spg_range']
        spg = list(range(spg_range[0], spg_range[1]))

        categorical_filter = ([True],['spg_number'],[spg])
    else:
        categorical_filter = None

    if 'k' in dataset_modifications:
        k = dataset_modifications['k']

        pre_eval_func = PeriodicKNN_PyG(k = k)
        pre_eval_func = AddNoise(std=0.1, transform=pre_eval_func)
    
    if 'target' in dataset_modifications:
        target = dataset_modifications['target']

    # Create model parameters

    if 'model_path' in model_parameters:
        model_path = model_parameters['model_path']

        model_name = model_path.split('/')[-1].split('.')[0]
        model_path = model_path.split(model_name)[0]
    else:
        date = datetime.now().strftime("%Y-%m-%d")
        model_path = f'models/{date}/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_num = 0
        model_name = f'{model_type_name}-k{k}-L{num_layers}-{target[:3]}{num_classes}-n{model_num}'

    if 'num_layers' in model_parameters:
        num_layers = model_parameters['num_layers']
    else:
        num_layers = 4

    if 'hidden_feats' in model_parameters:
        hidden_feats = model_parameters['hidden_feats']
    else:
        hidden_feats = 64

    if 'radial_feats' in model_parameters:
        radial_feats = model_parameters['radial_feats']
    else:
        radial_feats = 256

    # Create dataset and model

    dataset = FilteredAtomsDataset(source = "dft_3d",
                        n_unique_atoms = n_unique_atoms,
                        categorical_filter = categorical_filter,
                        target = target,
                        transform=pre_eval_func,
                        )

    class_weights = find_class_weights(dataset, target)
    class_weights = class_weights.to(device)
    print(class_weights)

    num_classes = class_weights.size(0)

    model = Alignn(num_layers = num_layers, hidden_feats = hidden_feats, radial_feats = radial_feats, out_feats=num_classes, classification=True).to(device)
    model_type_name = type(model).__name__

    # Create training parameters

    if 'n_epochs' in training_parameters:
        n_epochs = training_parameters['n_epochs']
    else:
        n_epochs = 10000

    if 'batch_size' in training_parameters:
        batch_size = training_parameters['batch_size']
    else:
        batch_size = 1

    if 'loss_func' in training_parameters:
        if training_parameters['loss_func'] == 'CrossEntropyLoss':
            loss_func = torch.nn.CrossEntropyLoss()
        elif training_parameters['loss_func'] == 'MSELoss':
            loss_func = torch.nn.MSELoss()
        else:
            raise ValueError('Invalid loss function')
    else:
        loss_func = torch.nn.CrossEntropyLoss()

    if 'optimizer' in training_parameters:
        assert 'optimizer_params' in training_parameters

        optimizer_params = training_parameters['optimizer_params']

        if training_parameters['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        elif training_parameters['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
        else:
            raise ValueError('Invalid optimizer')
    else:
        lr = 1e-3
        weight_decay = 0.1

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if 'scheduler' in training_parameters:
        assert 'scheduler_params' in training_parameters

        optimizer_params = training_parameters['scheduler_params']
        if training_parameters['scheduler'] == 'ConstantLR':
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, **optimizer_params)

        elif training_parameters['scheduler'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **optimizer_params)

        else:
            raise ValueError('Invalid scheduler')
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=n_epochs)

    # Train model

    train_model(model=model,
                dataset=dataset,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                n_epochs=n_epochs,
                batch_size=batch_size,
                model_name=model_name,
                save_path=model_path,
                device=device)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('configuration', type=str, help='Path to the model file')
    args = parser.parse_args()

    main(args.configuration)