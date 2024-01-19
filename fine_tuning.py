from nfflr.data.dataset import AtomsDataset
from nfflr.nn.transform import PeriodicAdaptiveRadiusGraph
from nfflr.nn.cutoff import XPLOR
from nfflr.data.atoms import Atoms as NFAtoms
from nfflr.models.gnn import alignn
from nfflr.models.utils import JP_Featurization
from nfflr.nn.transform import CustomPeriodicAdaptiveRadiusGraph

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from torch import nn

import torch
import dgl
import pickle
import jarvis
import sys
from HSGR_nfflr import (FilteredAtomsDataset,
                        collate_spg,
                        train_model)

if __name__ == "__main__":
    transform = CustomPeriodicAdaptiveRadiusGraph(cutoff = 8.0)
    n_atoms = 2
    spg = ('221','220','123','65','225')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_name = 'HSGR_nfflr'
    model_name_new = 'HSGR_nfflr_SWA'
    save_path = 'Models/23-01-16/'
    useAllSPG = True

    if useAllSPG:
        categorical_filter = None
    else:
        categorical_filter = ([True],['spg_number'],[spg])

    dataset = FilteredAtomsDataset(source = "dft_3d",
                            n_unique_atoms=(True, n_atoms),
                            categorical_filter=categorical_filter,
                            ).df
    
    spg = {}
    for i in dataset['spg_number']:
        spg[i] = True
    print(f'nspg = {len(spg)}')

    dataset = AtomsDataset(
        df = dataset,
        target = "spg_number",
        transform=transform,
        custom_collate_fn = collate_spg,
        n_train = 0.8,
        n_val = 0.1,
    )

    featurization = 'embedding'

    cfg = alignn.ALIGNNConfig(
        transform=transform,
        cutoff = XPLOR(7.5, 8),
        alignn_layers=4,
        norm="layernorm",
        atom_features=featurization,
        output_features=len(spg),
        classification = True,
        debug = False
    )

    batch_size = 8
    with open(f'{save_path}{model_name}.pkl', 'rb') as input_file:
        model = pickle.load(input_file)
    model.eval()
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    swa_model = AveragedModel(model)

    SWA_freq = round(len(dataset.split['train'])/batch_size)
    optimizer_cyclicLR = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=5e-4, step_size_up=SWA_freq, cycle_momentum=False)
    optimizer_SWA = SWALR(optimizer, swa_lr=1e-4)
    schedulers = (optimizer_SWA, optimizer_cyclicLR)

    train_model(model = model,
            dataset = dataset,
            device = device,
            model_name = model_name_new,
            save_path = save_path,
            epochs = 50,
            batch_size = batch_size,
            loss_func = criterion,
            optimizer = optimizer,
            scheduler=schedulers,
            use_arbitrary_feat=True,
            swa = swa_model,
            )
    
    optimizer_cyclicLR.swap_swa_sgd()
    output_dir = f'{save_path}{model_name}_final.pkl'
    with open(output_dir, 'wb') as output_file:
        torch.load(model, output_file)
