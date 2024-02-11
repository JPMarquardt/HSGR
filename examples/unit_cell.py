import os
import sys
import numpy as np
import argparse
import datetime
from scipy.stats import entropy

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
#import tkinter
matplotlib.use('Agg')

import torch
import torch.nn as nn
from sklearn.preprocessing import LabelBinarizer
from jarvis.db.figshare import data as jdata
from nfflr.data.dataset import AtomsDataset
import tqdm

import MDAnalysis as mda
from MDAnalysis.transformations import boxdimensions
from unit_cell_determination.RADFC import autocorrelation, RA_autocorrelation, create_supercell

if __name__ == '__main__':

    args = {'r_max_mult': 4, 'n_r_bins': 100, 'n_theta_bins': 20, 'n_phi_bins': 20, 'n_space_bins': 100, 'kernel': 'gaussian'}
    data = AtomsDataset('dft_3d')
    data_point = data[0][0]

    n = 3
    coords = torch.tensor(data_point.positions)
    lattice = torch.tensor(data_point.lattice)
    coords = coords @ lattice
    print(coords)
    coords = create_supercell(coords, lattice, n)
    types = data_point.numbers
    ohe = LabelBinarizer()
    ohe.fit(types)
    ohe_types = torch.from_numpy(ohe.transform(types)).repeat(n**3, 1)
    print(ohe_types.shape)

    uncertainty = torch.ones(coords.shape[0])/10
    r, angle = RA_autocorrelation(coords, uncertainty = uncertainty, atom_types = ohe_types)
    print(r, angle)
    print(data_point.lattice)
    print(autocorrelation(coords, uncertainty, ohe_types, lattice[0], 'gaussian'))
    print(autocorrelation(coords, uncertainty, ohe_types, lattice[1], 'gaussian'))
    print(autocorrelation(coords, uncertainty, ohe_types, lattice[2], 'gaussian'))
