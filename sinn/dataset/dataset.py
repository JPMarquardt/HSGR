from nfflr.data.dataset import AtomsDataset
from jarvis.core.atoms import Atoms as jAtoms
from sinn.dataset.space_groups import spg_properties
from openeye.oechem import OEGetAtomicNum


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
                 target: str = 'target',
                 transform: Callable = None,
                 collate: Callable = None,
                 sparsity: int = None,
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
        if transform is None:
            transform = lambda x: x
        if collate is None:
            collate = lambda x: x

        self.transform = transform
        self.collate = collate
        self.target = target
        
        if isinstance(source, str):
            dataset = AtomsDataset(source)
            dataset = pd.DataFrame(dataset.df)

        elif isinstance(source, pd.DataFrame):
            dataset = source

        else:
            dataset = universe2df(source)

        
        print(f'Taking dataset with size {dataset.shape}')
        if 'spg_number' in dataset.columns and isinstance(dataset.loc[0, 'spg_number'], str):
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
            dataset = one_hot_encode(dataset, 'spg_number')

        elif target == 'international_number':
            spg = spg_properties()
            spg2int = spg.space_group2international_number
            dataset['international_number'] = dataset['spg_number'].map(spg2int)

            dataset = one_hot_encode(dataset, 'international_number')

        if sparsity:
            dataset = dataset[dataset.index % sparsity == 1]
            dataset.reset_index(inplace=True)

        print(f'Dataset reduced to size {dataset.shape}')

        self.split = {'train': dataset.index[:int(0.8*len(dataset))],
                      'val': dataset.index[int(0.8*len(dataset)):int(0.9*len(dataset))],
                      'test': dataset.index[int(0.9*len(dataset)):]}

        self.dataset = dataset
        self.dataset['atoms'] = self.dataset['atoms'].apply(convert_dict)
        self.nextpos = 0

        return
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return (self.transform(self.dataset.loc[idx, 'atoms']), self.dataset.loc[idx, self.target])
        else:
            atoms = self.dataset.loc[idx, 'atoms'].apply(self.transform)
            target = self.dataset.loc[idx, self.target]
            return list(zip(atoms, target))


    def __iter__(self):
        self.nextpos = 0
        return self
    
    def __next__(self):
        if self.nextpos >= len(self.dataset):
            raise StopIteration
        else:
            self.nextpos += 1
            return self[self.nextpos - 1]
        

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

def universe2df(trajectory: Universe, **kwargs) -> pd.DataFrame:
    """
    Convert a GSD file to a pandas dataframe
    """
    if kwargs.get('target') is None:
        target = [0]*len(trajectory.trajectory)
    else:
        target = kwargs['target']
    
    atom_types = trajectory.atoms.types
    
    if trajectory.dimensions is not None:
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

    else:
        lattice_vectors = torch.eye(3)

    atoms_list = []
    jid_list = []

    atom_types = [OEGetAtomicNum(atom) for atom in atom_types]
    atom_types = torch.tensor(atom_types)

    for id, frame in enumerate(trajectory.trajectory):
        coords = torch.tensor(frame.positions) + torch.sum(lattice_vectors, dim=0) / 2
        atoms = {'lattice_mat': lattice_vectors, 'positions': coords, 'numbers': atom_types, 'cartesian': True}
        atoms_list.append(atoms)
        jid_list.append(id)

    df = pd.DataFrame(list(zip(atoms_list,target,jid_list)), columns = ['atoms','target','jid'])
    return df

def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    One hot encode a column of a pandas dataframe
    """
    unique_spg = df[column].unique()
    unique_spg.sort()
    print(f'Unique {column}: {unique_spg}')

    original_spg = df[column]
    df[column] = df[column].astype(object)

    for i, spg in enumerate(unique_spg):
        spg_tensor = torch.zeros(len(unique_spg), dtype=torch.float32)
        spg_tensor[i] = 1.0

        index = original_spg == spg
        with_spg = df.loc[index , column]

        spg_tensor_list = [spg_tensor] * len(with_spg)
        spg_tensor_list = pd.Series(spg_tensor_list)
        spg_tensor_list.index = with_spg.index

        df.loc[index, column] = spg_tensor_list
    return df

def convert_dict(dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a dictionary to a dictionary with tensors
    """
    new_dict = {}
    if dict.get('positions') is None:
        new_dict['positions'] = torch.tensor(dict['coords'])
    else:
        new_dict['positions'] = dict['positions']
    
    if dict.get('elements') is not None:
        new_dict['numbers'] = torch.tensor([OEGetAtomicNum(atoms) for atoms in dict['elements']])
    elif dict.get('numbers') is not None:
        new_dict['numbers'] = dict['numbers']
    
    if dict.get('lattice_mat') is not None:
        if isinstance(dict['lattice_mat'], torch.Tensor):
            new_dict['cell'] = dict['lattice_mat']
        else:
            new_dict['cell'] = torch.tensor(dict['lattice_mat'])
    elif dict.get('cell') is not None:
        new_dict['cell'] = dict['cell']
    
    return new_dict

    
def big_box_sampler(datapoint: dict, target_n: str) -> list[dict]:
    """
    Reduce the size of a dataset
    """
    n_atoms = datapoint['positions'].shape[0]
    dim = datapoint['positions'].shape[1]
    ratio = target_n / n_atoms
    ratio = min(ratio, 1)
    
    side_ratio = torch.tensor([ratio ** (1/dim)])
    replicates = torch.floor(1/side_ratio).int()

    if replicates == 1:
        return [datapoint]
    
    datapoint['positions'] = datapoint['positions'] - torch.min(datapoint['positions'], dim=0).values
    max_pos = torch.max(datapoint['positions'], dim=0).values

    filt_list = [[] for i in range(dim)]

    for i in range(dim):
        for j in range(replicates):
            low_filt = datapoint['positions'][:, i] >= j / replicates * max_pos[i]
            high_filt = datapoint['positions'][:, i] < (j + 1) / replicates * max_pos[i]

            filt = torch.logical_and(low_filt, high_filt)
            filt_list[i].append(filt)

    new_datapoints = []
    for i in range(replicates):
        for j in range(replicates):
            for k in range(replicates):
                filt = torch.logical_and(filt_list[0][i], torch.logical_and(filt_list[1][j], filt_list[2][k]))
                new_datapoint = {}
                for key in ['positions', 'numbers']:
                    new_datapoint[key] = datapoint[key][filt]
                new_datapoints.append(new_datapoint)

    new_dict = {'atoms': new_datapoints, 'target': [0]*len(new_datapoints)}
    return pd.DataFrame(new_dict)


    