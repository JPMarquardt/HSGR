from nfflr.data.dataset import AtomsDataset
from nfflr.nn.transform import PeriodicRadiusGraph
from nfflr.nn.cutoff import XPLOR
from nfflr.data.atoms import Atoms as NFAtoms
from nfflr.models.gnn import alignn
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


class FilteredAtomsDataset():
    def __init__(self,
                 source = "dft_3d",
                 n_unique_atoms: Tuple[bool, int] = None,
                 atom_types: Tuple[bool, str] = None,
                 categorical_filter: Tuple[Tuple[bool], Tuple[str], Tuple[Tuple[Any]]] = None
                 ):
        """
        source: which JARVIS dataset (or file) to take the datafrom
        arbitrary_feat: set atomic number to be numbers 1 to N where N is number of unique atoms. Recommended only use when n_unique_atoms is specified.

        each of the other filters then accepts a union of arguments: 
        the first is a bool, True indicates include listed, False indicates exclude
        the second and beyond indicate some criteria of the filter, specified in docs

        n_unique_atoms: allows filtering based on the number of unique atoms in a crystal to find/remove binary crystals etc.
        atom_types: allows filtering based on the element symbol
        categorical_filter: allows filtering of categorical variable such as spg etc. Inputs must be given as tuples to allow for multiple category filters

        """
        dataset = AtomsDataset(
            source
        )

        dataset = pd.DataFrame(dataset.df)
        print(f'Taking dataset with size {dataset.shape}')


        if atom_types:
            NotImplementedError()

        if n_unique_atoms:

            nAtomTypeList = []
            for search in dataset['search']:
                nAtomTypeList.append(len(search.split('-')))
            nAtomType = pd.DataFrame(nAtomTypeList)

            filt = nAtomType[0] == (n_unique_atoms[1] + 1)
            if not n_unique_atoms[0]:
                filt = ~filt

            dataset = dataset[filt]
            dataset.reset_index(inplace=True)

        if categorical_filter:
            for single_categorical_filter in zip(*categorical_filter):

                filt = dataset[single_categorical_filter[1]].isin(single_categorical_filter[2])
                if not single_categorical_filter[0]:
                    filt = ~filt

                dataset = dataset[filt]
                dataset.reset_index(inplace=True)

        print(f'Dataset reduced to size {dataset.shape}')
        self.df = dataset
        return None
    
def arbitrary_feat(dataset):
    for datapoint in dataset:
        graph_node_feat = datapoint[0].ndata['atomic_number']
        unique_atoms = torch.unique(graph_node_feat)

        mask_list = []
        for atom in unique_atoms:
            mask_list.append(graph_node_feat == atom)
        random.shuffle(mask_list)

        for i, mask in enumerate(mask_list):
            graph_node_feat[mask] = i

    return dataset

def collate_spg(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]], device='cpu'):
    """Dataloader helper to batch graphs cross `samples`.

    Forces get collated into a graph batch
    by concatenating along the atoms dimension

    energy and stress are global targets (properties of the whole graph)
    total energy is a scalar, stess is a rank 2 tensor

    FOR SPG or other categorization
    """
    if torch.cuda.is_available():
        device = 'cuda'

    graphs, targets = map(list, zip(*samples))
    for graph in graphs:
        graph = graph.to(device)
    target_block = torch.stack(targets, 0)
    return dgl.batch(graphs), target_block

def train_model(model,
                dataset,
                epochs,
                model_name,
                device = 'cpu',
                loss_func = nn.MSELoss(),
                optimizer = None,
                save_path = '',
                batch_size = 4,
                loss_graph = True,
                MAE_graph = True,
                use_arbitrary_feat = False
                ):
    
    t_device = torch.device('cuda:0')
    print(t_device)
    for i, datapoint in enumerate(dataset):
        graph, target = dataset.prepare_batch_default(datapoint, device = device)

    print(dataset[0][0].device)
    print(dataset[0][1].get_device())
    for param in model.parameters():
        param = param.to(device)
    print(f'Model and dataset sent to {device}')

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate,
        sampler=SubsetRandomSampler(dataset.split["train"]),
        drop_last=True
    )
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=dataset.collate,
        sampler=SubsetRandomSampler(dataset.split["test"]),
        drop_last=True
    )

    if optimizer == None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    ave_training_MAE = []
    ave_training_loss = []
    ave_test_loss = []
    ave_test_MAE = []

    for epoch in range(epochs):

        if use_arbitrary_feat:
            dataset = arbitrary_feat(dataset)

        #to keep all caluculations on the gpu we need a tensor on the gpu to keep track of the step
        gpu_step = torch.tensor([1], dtype = torch.long).to(device)
        ave_loss = torch.tensor([0], dtype = torch.long).to(device)
        ave_MAE = torch.tensor([0], dtype = torch.long).to(device)

        model.train()
        for step, (g, y) in enumerate(tqdm(train_loader)):
            if step == 1:
                print(g.device)
            pred = model(g)
            loss = loss_func(pred, y)
            MAE = torch.sum(torch.abs(pred - y))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            inv_step = 1/gpu_step
            inv_step_comp = 1 - inv_step
            ave_loss = ave_loss @ inv_step_comp + loss @ inv_step
            ave_MAE = ave_MAE @ inv_step_comp + MAE @ inv_step
            gpu_step += 1

        ave_training_loss.append(ave_loss)
        ave_training_MAE.append(ave_MAE)
        print(f'Epoch {epoch}-- Train Loss: {ave_training_loss} Train MAE: {ave_training_MAE}')


        gpu_step = torch.tensor([1], dtype = torch.long).to(device)
       	ave_loss = torch.tensor([0], dtype = torch.long).to(device)
       	ave_MAE = torch.tensor([0], dtype = torch.long).to(device)

        model.eval()
        with torch.no_grad():
            for (g, y) in tqdm(test_loader):
                pred = model(g)
                loss = loss_func(pred, y)
                MAE = torch.sum(torch.abs(pred - y))

                inv_step = 1/gpu_step
                inv_step_comp = 1 - inv_step
                ave_loss = ave_loss @ inv_step_comp + loss @ inv_step
       	        ave_MAE = ave_MAE @ inv_step_comp +	MAE @ inv_step
       	        gpu_step +=	1

        ave_test_loss_list.append(ave_loss)
        ave_test_MAE_list.append(ave_MAE)
        print(f'Epoch {epoch}-- Test Loss: {ave_test_loss} Test MAE: {ave_test_MAE}')


        output_dir = f'{save_path}{model_name}.pkl'
        with open(output_dir, 'wb') as output_file:
            pickle.dump(model, output_file)

        if loss_graph:
            plt.plot(ave_training_loss_list, label = 'train')
            plt.plot(ave_test_loss_list, label = 'test')
            plt.xlabel("training epoch")
            plt.ylabel("loss")
            plt.semilogy()
            plt.savefig(f'{save_path}{model_name}_loss.png')

        if MAE_graph:
            plt.plot(ave_training_MAE_list, label = 'train')
            plt.plot(ave_test_MAE_list, label = 'test')
            plt.xlabel("training epoch")
            plt.ylabel("loss")
            plt.semilogy()
            plt.savefig(f'{save_path}{model_name}_MAE.png')

if __name__ == '__main__':
    transform = PeriodicRadiusGraph(cutoff = 8.0)

    dataset = FilteredAtomsDataset(source = "dft_3d",
                            n_unique_atoms=(True,2),
                            categorical_filter=([True],['spg_number'],[('221','220','123','65','225')])
                            ).df
    
    dataset = AtomsDataset(
        df = dataset,
        target = "spg_number",
        transform=transform,
        custom_collate_fn = collate_spg,
        n_train = 0.8,
        n_val = 0.1,
    )

    cfg = alignn.ALIGNNConfig(
        transform=transform,
        cutoff = XPLOR(7.5, 8.0),
        alignn_layers=4,
        norm="layernorm",
        atom_features="embedding",
        output_features=5,
        classification = True
    )

    model = alignn.ALIGNN(cfg)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    train_model(model = model,
                dataset = dataset,
                device = 'cuda',
                model_name = 'HSGR_M1',
                save_path = '',
                epochs = 30,
                batch_size = 4,
                loss_func = criterion,
                optimizer = optimizer
                )


