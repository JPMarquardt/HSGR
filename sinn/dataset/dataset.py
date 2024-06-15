from nfflr.data.dataset import AtomsDataset, Atoms #maybe remove this dependence for maybe just JARVIS
from jarvis.core.atoms import Atoms as jAtoms

import torch
import dgl
import pandas as pd
import random
import numpy as np

from MDAnalysis import Universe
from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable, Iterable)


class FilteredAtomsDataset():
    def __init__(self,
                 source = "dft_3d",
                 n_unique_atoms: Tuple[bool, int] = None,
                 atom_types: Tuple[bool, str] = None,
                 categorical_filter: Tuple[Tuple[bool], Tuple[str], Tuple[Tuple[Any]]] = None,
                 target: str = None,
                 transform: Callable = None,
                 collate: Callable = None,
                 **kwargs
                 ):
        """
        A wrapper on the nfflr AtomsDataset to allow for filtering of the dataset

        source: which JARVIS dataset (or file) to take the datafrom
        arbitrary_feat: set atomic number to be numbers 1 to N where N is number of unique atoms. Recommended only use when n_unique_atoms is specified.

        each of the other filters then accepts a union of arguments: 
        the first is a bool, True indicates include listed, False indicates exclude
        the second and beyond indicate some criteria of the filter, specified in docs

        n_unique_atoms: allows filtering based on the number of unique atoms in a crystal to find/remove binary crystals etc.

        atom_types: allows filtering based on the element symbol

        categorical_filter: allows filtering of categorical variable such as spg etc. Inputs must be given as tuples to allow for multiple category filters

        """
        self.transform = transform
        self.collate = collate
        
        if isinstance(source, str):
            dataset = AtomsDataset(source)
            dataset = pd.DataFrame(dataset.df)

        elif isinstance(source, pd.DataFrame):
            dataset = source

        else:
            dataset = universe2df(source)

        
        print(f'Taking dataset with size {dataset.shape}')
        if 'spg_number' in dataset.columns:
            dataset['spg_number'] = dataset['spg_number'].str.extract('(\d+)').astype(int)

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

        if target == 'spg_number':
            unique_spg = dataset['spg_number'].unique()
            unique_spg.sort()

            original_spg = dataset['spg_number']
            dataset['spg_number'] = dataset['spg_number'].astype(object)

            for i, spg in enumerate(unique_spg):
                spg_tensor = torch.zeros(len(unique_spg), dtype=torch.float32)
                spg_tensor[i] = 1.0

                index = original_spg == spg
                with_spg = dataset.loc[index , 'spg_number']

                spg_tensor_list = [spg_tensor] * len(with_spg)
                spg_tensor_list = pd.Series(spg_tensor_list)
                spg_tensor_list.index = with_spg.index

                dataset.loc[index, 'spg_number'] = spg_tensor_list

        print(f'Dataset reduced to size {dataset.shape}')
        self.dataset = AtomsDataset(df = dataset,
                                    target = target,
                                    transform = self.transform,
                                    custom_collate_fn = self.collate,
                                    **kwargs)
        return
    
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

def collate_general(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
    """Dataloader helper to batch graphs cross `samples`.

    Forces get collated into a graph batch
    by concatenating along the atoms dimension

    energy and stress are global targets (properties of the whole graph)
    total energy is a scalar, stess is a rank 2 tensor

    for SPG or other categorization
    """

    if isinstance(samples[0][0], tuple):
        graphs, targets = map(list, zip(*samples))
        g, lg = map(list, zip(*graphs))
        target_block = torch.stack(targets, 0)
        return (dgl.batch(g), dgl.batch(lg)), target_block
    
    graphs, targets = map(list, zip(*samples))
    target_block = torch.stack(targets, 0)
    return dgl.batch(graphs), target_block

def collate_multihead_noise(batch: Tuple[Tuple[torch.Tensor, dgl.DGLGraph]]):
    """
    Create a batch of DGLGraphs
    """
    batch, classification_target_list = zip(*batch)
    g_list, regression_target_list = map(list, zip(*batch))
    bg = dgl.batch(g_list)
    target = (torch.stack(classification_target_list), torch.cat(regression_target_list))
    return bg, target

def collate_noise(batch: Tuple[Tuple[torch.Tensor, dgl.DGLGraph]]):
    """
    Create a batch of DGLGraphs
    """
    batch, dataset_target_list = zip(*batch)
    g_list, noise_target_list = map(list, zip(*batch, dataset_target_list))
    bg = dgl.batch(g_list)
    target = torch.cat(noise_target_list)
    return bg, target

def universe2df(trajectory: Universe, **kwargs) -> pd.DataFrame:
    """
    Convert a GSD file to a pandas dataframe
    """
    if kwargs.get('target') is None:
        target = [0]*len(trajectory.trajectory)
    else:
        target = kwargs['target']
    
    atom_types = trajectory.atoms.types
    lattice_parameters = torch.tensor(trajectory.dimensions)
    abc = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    pi = torch.tensor(np.pi)
    alpha = angles[0] * (pi / 180)
    beta = angles[1] * (pi / 180)
    gamma = angles[2] * (pi / 180)

    cx = torch.cos(beta)
    cy = (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma)
    cz = torch.sqrt(1 - cx**2 - cy**2)

    a1 = abc[0] * torch.tensor([1, 0, 0])
    a2 = abc[1] * torch.tensor([torch.cos(gamma), torch.sin(gamma), 0])
    a3 = abc[2] * torch.tensor([cx, cy, cz])

    lattice_vectors = torch.stack([a1, a2, a3])

    atoms_list = []
    jid_list = []
    for id, frame in enumerate(trajectory.trajectory):
        coords = torch.tensor(frame.positions) + torch.sum(lattice_vectors, dim=0) / 2
        atoms = Atoms(jAtoms(lattice_mat=lattice_vectors, coords=coords, elements=atom_types, cartesian=True))
        atoms_list.append(atoms)
        jid_list.append(id)


    df = pd.DataFrame(list(zip(atoms_list,target,jid_list)), columns = ['atoms','target','jid'])
    return df