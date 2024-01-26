from nfflr.data.dataset import AtomsDataset
from nfflr.nn.transform import PeriodicAdaptiveRadiusGraph
from nfflr.nn.cutoff import XPLOR
from nfflr.data.atoms import Atoms as NFAtoms
from nfflr.models.gnn import alignn
from nfflr.models.utils import JP_Featurization
from nfflr.nn.transform import CustomPeriodicAdaptiveRadiusGraph

from torch.optim.swa_utils import AveragedModel, SWALR

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
        graph_node_feat = datapoint[0][0].ndata['atomic_number']
        unique_atoms = torch.unique(graph_node_feat)

        mask_list = []
        for atom in unique_atoms:
            mask_list.append(graph_node_feat == atom)
        random.shuffle(mask_list)

        for i, mask in enumerate(mask_list):
            graph_node_feat[mask] = i

    return dataset

def collate_spg(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
    """Dataloader helper to batch graphs cross `samples`.

    Forces get collated into a graph batch
    by concatenating along the atoms dimension

    energy and stress are global targets (properties of the whole graph)
    total energy is a scalar, stess is a rank 2 tensor

    FOR SPG or other categorization
    """

    if isinstance(samples[0][0], tuple):
        graphs, targets = map(list, zip(*samples))
        g, lg = map(list, zip(*graphs))
        target_block = torch.stack(targets, 0)
        return (dgl.batch(g), dgl.batch(lg)), target_block
    
    graphs, targets = map(list, zip(*samples))
    target_block = torch.stack(targets, 0)
    return dgl.batch(graphs), target_block


def run_epoch(model, loader, loss_func, optimizer, device, epoch, scheduler = None, train=True, swa=False):
    """Runs one epoch of training or evaluation."""

    ave_MAE = 0
    ave_loss = 0

    if train:
        model.train()
        grad = torch.enable_grad()
        train_or_test = 'Train'
    else:
        model.eval()
        grad = torch.no_grad()
        train_or_test = 'Test'

    with grad:
        for step, (g, y) in enumerate(tqdm(loader)):

            if isinstance(g, tuple):
                g = tuple(graph_part.to(device) for graph_part in g)
            else:
                g = g.to(device)

            y = y.to(device)

            pred = model(g)
            loss = loss_func(pred, y)
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            MAE = torch.sum(torch.abs(y - torch.where(y == 1, pred, 0)))/y.shape[0]

            inv_step = 1/(step + 1)
            inv_step_comp = 1 - inv_step
            ave_loss = ave_loss * inv_step_comp + loss.item() * inv_step
            ave_MAE = ave_MAE * inv_step_comp + MAE.item() * inv_step

            torch.cuda.empty_cache()

    if swa:
        swa_model = swa
        swa_model.update_parameters(model)
    
    if scheduler:
        if type(scheduler) == tuple:
            for index in range(len(scheduler)):
                scheduler[index].step()
        else:
            scheduler.step()

    print(f'Epoch {epoch}-- {train_or_test} Loss: {ave_loss} {train_or_test} MAE: {ave_MAE}')

    return ave_loss, ave_MAE


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
                scheduler = None,
                use_arbitrary_feat = False,
                swa = False
                ):
    
    t_device = torch.device(device)

    if optimizer == None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    model = model.to(t_device)

    ave_training_MAE = []
    ave_training_loss = []
    ave_test_loss = []
    ave_test_MAE = []
    final_average_MAE = []
    epoch_saved = []

    for epoch in range(epochs):

        if use_arbitrary_feat:
            dataset = arbitrary_feat(dataset)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate,
            sampler=SubsetRandomSampler(dataset.split["train"]),
            drop_last=True
        )
        ave_loss, ave_MAE = run_epoch(model=model,
                                      loader=train_loader,
                                      loss_func=loss_func,
                                      optimizer=optimizer,
                                      device=t_device,
                                      epoch=epoch,
                                      scheduler=scheduler,
                                      train=True,
                                      swa=swa)

        ave_training_loss.append(ave_loss)
        ave_training_MAE.append(ave_MAE)

        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate,
            sampler=SubsetRandomSampler(dataset.split["test"]),
            drop_last=True
        )

        ave_loss, ave_MAE = run_epoch(model=model,
                                      loader=test_loader,
                                      loss_func=loss_func,
                                      optimizer=optimizer, 
                                      device=t_device,
                                      epoch=epoch,
                                      scheduler=None,
                                      train=False,
                                      swa=False)

        ave_test_loss.append(ave_loss)
        ave_test_MAE.append(ave_MAE)

        if ave_loss <= min(ave_test_loss):
            output_dir = f'{save_path}{model_name}.pkl'
            with open(output_dir, 'wb') as output_file:
                torch.save(model, output_file)
            final_average_MAE.append(ave_MAE)
            epoch_saved.append(epoch)

        if loss_graph:
            plt.figure()
            plt.plot(ave_training_loss, label = 'train')
            plt.plot(ave_test_loss, label = 'test')
            plt.xlabel("training epoch")
            plt.ylabel("loss")
            plt.semilogy()
            plt.legend(loc='upper right')
            plt.savefig(f'{save_path}{model_name}_loss.png')
            plt.close()

        if MAE_graph:
            plt.figure()
            plt.plot(ave_training_MAE, label = 'train')
            plt.plot(ave_test_MAE, label = 'test')
            plt.plot(epoch_saved, final_average_MAE, 'r.', label = 'saved')
            plt.xlabel("training epoch")
            plt.ylabel("loss")
            plt.semilogy()
            plt.legend(loc='upper right')
            plt.savefig(f'{save_path}{model_name}_MAE.png')
            plt.close()

if __name__ == '__main__':
    transform = CustomPeriodicAdaptiveRadiusGraph(cutoff = 8.0)
    n_atoms = 2
    spg = ('221','220','123','65','225')
    device = 'cuda'
    save_path = 'Models/23-01-21/'
    model_name = 'HSGR_trial'
    useAllSPG = True

    if useAllSPG:
        categorical_filter = None
    else:
        categorical_filter = ([True],['spg_number'],[spg])

    dataset = FilteredAtomsDataset(source = "dft_3d",
                            n_unique_atoms = (True,n_atoms),
                            categorical_filter = categorical_filter
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

    with open(f'{save_path}{model_name}_split.pkl', 'wb') as f:
        pickle.dump(dataset.split, f)
        print(f'Split saved to {save_path}{model_name}_split.pkl')

    featurization = {'n_atoms': n_atoms, 'n_heads': int(torch.sqrt(torch.tensor(len(spg))).item()), 'hidden_features': 128, 'eps': 1e-2}

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
    model = alignn.ALIGNN(cfg)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    train_model(model = model,
                dataset = dataset,
                device = device,
                model_name = model_name,
                save_path = save_path,
                epochs = 300,
                batch_size = batch_size,
                loss_func = criterion,
                optimizer = optimizer,
                use_arbitrary_feat=True
                )


    best_model = torch.load(f'{save_path}{model_name}.pkl') 
    swa_model = AveragedModel(best_model)
    swa_model.train()

    SWA_freq = round(len(dataset.split['train'])/batch_size)
    optimizer_cyclicLR = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=5e-4, step_size_up=SWA_freq, cycle_momentum=False)
    optimizer_SWA = SWALR(optimizer, swa_lr=1e-4)
    schedulers = (optimizer_SWA, optimizer_cyclicLR)

    train_model(model = best_model,
            dataset = dataset,
            device = device,
            model_name = f'{model_name}_SWA',
            save_path = save_path,
            epochs = 100,
            batch_size = batch_size,
            loss_func = criterion,
            optimizer = optimizer,
            scheduler=schedulers,
            use_arbitrary_feat=True,
            swa = swa_model,
            )
    
    optimizer_SWA.swap_swa_sgd()
    output_dir = f'{save_path}{model_name}_final.pkl'
    torch.save(model, output_dir)
