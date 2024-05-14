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
from tqdm import tqdm

import MDAnalysis as mda
from MDAnalysis.transformations import boxdimensions
import gsd.hoomd

def spherical2cart(x):
    """
    Convert polar to cartesian coordinates
    """
    resqueeze = False

    if x.dim() == 1:
        x = x.unsqueeze(0)
        resqueeze = True

    r = x[:, 0]
    theta = x[:, 1]
    phi = x[:, 2]

    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    z = z.unsqueeze(-1)

    output = torch.cat((x, y, z), dim=-1)

    if resqueeze:
        output = output.squeeze()
    
    return output

def cart2spherical(r):
    """
    Convert cartesian to spherical coordinates
    """
    resqueeze = False
    if r.dim() == 1:
        r = r.unsqueeze(0)
        resqueeze = True

    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    r = r.unsqueeze(-1)
    theta = theta.unsqueeze(-1)
    phi = phi.unsqueeze(-1)

    output = torch.cat((r, theta, phi), dim=-1)
    if resqueeze:
        output = output.squeeze()

    return output


def phi_angle_matrix(data):
    """
    atoms x atoms angle matrix: from 1, 0, 0 in the xy plane
    """
    full_distance_matrix = distance_matrix(data)

    hypotenuse = distance_matrix(data[:,:2])

    adjacent = data[None, :, 0] - data[:, None, 0]

    sign_y = torch.sign(data[None, :, 1] - data[:, None, 1])
    
    cos_theta = adjacent / hypotenuse

    non_diagonal_nans = (hypotenuse == 0) & (full_distance_matrix != 0)

    cos_theta[non_diagonal_nans] = 1

    cos_theta = torch.clamp(cos_theta, -1, 1)

    return torch.acos(cos_theta) * sign_y

def theta_angle_matrix(data):
    """
    atoms x atoms angle matrix: angle above xy plane
    """
    adjacent = data[None, :, 2] - data[:, None, 2]

    hypotenuse = distance_matrix(data)

    cos_phi = adjacent / hypotenuse

    return torch.acos(cos_phi)

def distance_matrix(data: torch.tensor):
    """
    Compute the distance matrix
    """
    x0 = data[None, :]
    x1 = data[:, None]
    dx = (x0 - x1)
    square_distance_matrix = torch.sqrt(torch.sum(dx**2, dim = 2))

    return square_distance_matrix

def find_local_max(DF):
    """
    Find the peaks in the data
    """
    peak_matrices = []
    for i in range(DF.dim()):
        delta = DF - DF.roll(1, dims=i)
        delta_sign = delta.sign()
        delta_delta_sign = delta_sign - delta_sign.roll(1, dims=i)
        peaks = delta_delta_sign == -2
        peaks = peaks.roll(-1, dims=i)
        peak_matrices.append(peaks)

    and_matrix = peak_matrices[0]
    if len(peak_matrices) > 1:
        for i in range(len(peak_matrices)-1):
            peak_matrices = torch.logical_and(and_matrix, peak_matrices[i+1])
        
    output = and_matrix.nonzero()
    
    return output

def check_mask_zeros(mask):
    """
    Check if the mask has any zeros for any of the atom types
    """
    for i in range(mask.shape[-1]):
        if torch.sum(mask[:, :, i]) == 0:
            return True
    else:
        return False

def create_supercell(data: torch.tensor, lattice: torch.tensor, n: int):
    """
    Create a supercell
    """
    n_atoms = data.shape[0]
    supercell = torch.zeros((data.shape[0] * n**3, data.shape[1]))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                ind = (i*n**2 + j*n + k) * n_atoms
                displacement = torch.tensor([i, j, k], dtype=torch.float) @ lattice
                supercell[ind:ind + n_atoms] = data + displacement[None, :]

    return supercell