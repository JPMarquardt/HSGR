from nfflr.data.dataset import AtomsDataset
from nfflr.nn.transform import PeriodicRadiusGraph
from nfflr.nn.cutoff import XPLOR
from nfflr.data.atoms import Atoms as NFAtoms
from nfflr.models.gnn import alignn
from nfflr.nn.transform import PeriodicAdaptiveRadiusGraph
from nfflr.models.utils import JP_Featurization
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Any, Dict, List, Literal, Tuple, Union, Optional, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from torch import nn

import torch
import dgl
import pickle
import jarvis
from HSGR_nfflr import (FilteredAtomsDataset,
                        collate_spg,
                        train_model)

if __name__ == "__main__":
    transform = PeriodicAdaptiveRadiusGraph(cutoff = 8.0)
    n_atoms = 2
    spg = ('221','220','123','65','225')
    device = 'cuda'

    dataset = FilteredAtomsDataset(source = "dft_3d",
                            n_unique_atoms=(True, n_atoms),
                            categorical_filter=([True],['spg_number'],[spg])
                            ).df
    
    """
    spg = {}
    for i in dataset['spg_number']:
        spg[i] = True
    print(f'nspg = {len(spg)}')
    """


    dataset = AtomsDataset(
        df = dataset,
        target = "spg_number",
        transform=transform,
        custom_collate_fn = collate_spg,
        n_train = 0.8,
        n_val = 0.1,
    )

    #featurization = {'n_atoms': n_atoms, 'n_heads': len(spg), 'hidden_features': 128, 'use_atom_feat': False, 'eps': 1e-3}
    featurization = "embedding"

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

    model = alignn.ALIGNN(cfg)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    train_model(model = model,
                dataset = dataset,
                device = device,
                model_name = 'HSGR_NOBP_M1',
                save_path = 'NOBP_M1/',
                epochs = 300,
                batch_size = 8,
                loss_func = criterion,
                optimizer = optimizer,
                use_arbitrary_feat=False
                )
