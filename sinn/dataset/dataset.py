from nfflr.data.dataset import AtomsDataset #maybe remove this dependence for maybe just JARVIS

import torch
import dgl
import pandas as pd
import random

from typing import (Any, Dict, List, Literal, Tuple, Union, Optional, Callable)


class FilteredAtomsDataset():
    def __init__(self,
                 source = "dft_3d",
                 n_unique_atoms: Tuple[bool, int] = None,
                 atom_types: Tuple[bool, str] = None,
                 categorical_filter: Tuple[Tuple[bool], Tuple[str], Tuple[Tuple[Any]]] = None,
                 transform: Callable = None,
                 collate: Callable = None
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

        dataset = AtomsDataset(source)
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
        self.dataset = AtomsDataset(df = dataset,
                                    transform = self.transform,
                                    custom_collate_fn = self.collate)
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

def collate_noise(batch: Tuple[Tuple[torch.Tensor, dgl.DGLGraph]]):
    """
    Create a batch of DGLGraphs
    """
    batch, _ = zip(*batch)
    g_list = [g for g, _ in batch]
    target_list = [target for _, target in batch]
    bg = dgl.batch(g_list)
    target = torch.cat(target_list)
    return bg, target