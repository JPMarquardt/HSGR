import os
import sys
import numpy as np
import argparse
import datetime
from scipy.stats import entropy
import math

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
#import tkinter

import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from jarvis.db.figshare import data as jdata
from nfflr.data.dataset import AtomsDataset
import tqdm

import MDAnalysis as mda
from MDAnalysis.transformations import boxdimensions

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from unit_cell_determination.RADFAC import RA_autocorrelation, autocorrelation, create_supercell, find_radf_peaks
from unit_cell_determination.MLP import *

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f'Using {device} device')
    data = AtomsDataset('dft_3d')

    for i in range(len(data)):
        kwargs = {'r_max_mult': torch.tensor(4.0), 
                  'n_r_bins': 200, 
                  'n_angle_bins': 19, 
                  'kernel': 'gaussian', 
                  'use_cutoff': True,
                  'verbose': True,
                  'extend_r_max': 1/4,
                  'n_samples': 25,
                  'significance': 0.1,}
        data_point = data[i][0]

        n = math.floor((7000/data_point.numbers.shape[0])**(1/3))
        coords = data_point.positions
        lattice = data_point.lattice
        coords = coords @ lattice
        print(lattice)
        std = 0.1
        coords = create_supercell(coords, lattice, n)
        coords += torch.rand(coords.shape)*((std/3)**0.5)
        plt.scatter(coords[:, 0], coords[:, 2])
        plt.savefig('coords.png')
        types = data_point.numbers.long()

        ohe_types = torch.nn.functional.one_hot(types,num_classes=-1)
        mask = ohe_types.sum(dim=0) > 0
        ohe_types = ohe_types[:, mask]
        ohe_types = ohe_types.repeat(n**3, 1)
        uncertainty = torch.ones(coords.shape[0])*std

        coords = coords.to(device)
        uncertainty = uncertainty.to(device)
        ohe_types = ohe_types.to(device)

        for j in range(3):
            auto_corr = autocorrelation(coords, data_uncertainty = uncertainty, atom_types = ohe_types, displacement=lattice[j])
            print(auto_corr)
        cart, auto_corr = RA_autocorrelation(coords, uncertainty = uncertainty, atom_types = ohe_types, **kwargs)

        print(f'Cartestian: {cart}')
        print(f'Lattice: {lattice}')
        loss = 0
        for j in range(3):
            temp_loss = torch.tensor([0., 0., 0., 0., 0., 0.])
            for k in range(3):
                temp_loss[k] = torch.mean(torch.abs(lattice[j] - cart[k]))
                temp_loss[k+3] = torch.mean(torch.abs(lattice[j] + cart[k]))
            loss += temp_loss.min()

        with open('losses.txt', 'a') as f:
            f.write(f'{i}: {loss.item()}\n')
