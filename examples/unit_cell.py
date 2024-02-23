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

from unit_cell_determination.RADFAC import RA_autocorrelation, autocorrelation, create_supercell, spherical2cart, cart2spherical
from unit_cell_determination.MLP import *

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    data = AtomsDataset('dft_3d')

    for i in range(len(data)):
        kwargs = {'r_max_mult': torch.tensor(4.0), 
                  'n_r_bins': 200, 
                  'n_theta_bins': 80, 'n_phi_bins': 40, 
                  'n_space_bins': 100, 
                  'kernel': 'gaussian', 
                  'use_cutoff': True}
        data_point = data[i][0]

        n = math.floor((7000/data_point.numbers.shape[0])**(1/3))
        coords = data_point.positions
        lattice = data_point.lattice
        coords = coords @ lattice
        coords = create_supercell(coords, lattice, n)
        types = data_point.numbers.long()

        ohe_types = torch.nn.functional.one_hot(types,num_classes=-1)
        mask = ohe_types.sum(dim=0) > 0
        ohe_types = ohe_types[:, mask].repeat(n**3, 1)
        uncertainty = torch.ones(coords.shape[0])/10

        coords = coords.to(device)
        uncertainty = uncertainty.to(device)
        ohe_types = ohe_types.to(device)

        spherical, auto_corr = RA_autocorrelation(coords, uncertainty = uncertainty, atom_types = ohe_types, **kwargs)
        cart = spherical2cart(spherical)

        print(f'Cartesian: {cart}')
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
