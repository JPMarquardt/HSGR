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

"""
Hyperparameters:
- n_r_bins: number of radial bins
- r_max_mult: number to multiply the smallest radius by to get maximum radial distance
- n_theta_bins: number of angular bins
- n_phi_bins: number of azimuthal bins
- n_space_bins: number of bins in space
- kernel: kernel to use for the autocorrelation
    - type: type of kernel
    - sigma: standard deviation of the kernelana
- n_angle_max: number of angles to consider in autocorrelation
- n_radial_max: number of radial distances to consider in autocorrelation
"""


def RA_autocorrelation(data,
                       r_max_mult: float = 4, 
                       n_r_bins: int = 100, 
                       n_theta_bins: int = 20, 
                       n_phi_bins: int = 20, 
                       kernel: str = 'gaussian',
                       use_cutoff: bool = False,
                       **kwargs):
    """
    Compute the autocorrelation spatial radial x angular function of the rdfs
    """
    if type(data) == mda.Universe:
        data = torch.mean(torch.from_numpy(data.coord.positions), dim=-1)
        uncertainty = torch.std(torch.from_numpy(data.coord.positions), dim =-1)
        atom_types = torch.from_numpy(data.atoms.types)
        #atom type needs to be changed to categorical for more than 2 atom types
    else:
        uncertainty = kwargs['uncertainty']
        atom_types = kwargs['atom_types']

    #atoms x atoms distance matrix
    distance_matrix = torch.sqrt(torch.sum((data[None, :, :] - data[:, None, :])**2, dim = -1))

    #r_min/max for bins
    r_min = torch.min(distance_matrix[distance_matrix != 0])
    r_max = r_min * r_max_mult
    r_bins = torch.linspace(r_min, r_max, n_r_bins)

    #theta and phi bins

    dth = np.pi / n_theta_bins
    dphi = np.pi / n_phi_bins
    th_min = 0
    phi_min = -np.pi/2
    th_max = np.pi - dth
    phi_max = np.pi/2 - dphi

    th_bins = torch.linspace(th_min, th_max, n_theta_bins)
    phi_bins = torch.linspace(phi_min, phi_max, n_phi_bins)

    #cutoff should only be use when the data is very large
    if use_cutoff:
        cutoff = r_max
    else: 
        cutoff = None

    auto_corr = torch.zeros((n_r_bins, n_theta_bins, n_phi_bins))
    for r_ind, r in enumerate(r_bins):
        for th_ind, theta in enumerate(th_bins):
            for phi_ind, phi in enumerate(phi_bins):
                displacement = polar2cart(torch.tensor((r, theta, phi)))
                auto_corr[r_ind, th_ind, phi_ind] = autocorrelation(data=data, 
                                                                    data_uncertainty=uncertainty, 
                                                                    atom_types=atom_types, 
                                                                    displacement=displacement, 
                                                                    kernel=kernel, 
                                                                    cutoff=cutoff)
                #stop multicounting spins that do nothing
                if (theta == 0) and phi_ind != 1:
                    auto_corr[r_ind, th_ind, phi_ind] = 0

    tot_ind = find_local_max(auto_corr)
    peak_val = auto_corr[tot_ind[:, 0], tot_ind[:, 1], tot_ind[:, 2]]
    print(peak_val)
    _, top3_peak_ind = torch.topk(peak_val, 3)
    print(_)
    print(top3_peak_ind)
    top_3_tot_ind = tot_ind[top3_peak_ind]
    print(top_3_tot_ind)

    r = r_bins[top_3_tot_ind[:, 0]]
    angle = torch.stack([th_bins[top_3_tot_ind[:, 1]], phi_bins[top_3_tot_ind[:, 2]]], dim = -1)
    plt.figure()
    sns.heatmap(auto_corr[90])
    plt.savefig('auto_corr_heatmap_far.png')
    print(auto_corr[90,0])

    return r, angle

def polar2cart(x):
    """
    Convert polar to cartesian coordinates
    """
    r = x[0]
    theta = x[1]
    phi = x[2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return torch.tensor((x, y, z))

def cart2polar(x):
    """
    Convert cartesian to polar coordinates
    """
    x = x[0]
    y = x[1]
    z = x[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    return torch.tensor((r, theta, phi))

def autocorrelation(data: torch.tensor,
                    data_uncertainty: torch.tensor,
                    atom_types: torch.tensor,
                    displacement: torch.tensor, 
                    kernel: str = 'gaussian',
                    cutoff: float = None):
    """
    Compute the autocorrelation of the data with a displacement
    Add math explanation below
    """
    sign_matrix = atom_types[None, :] * atom_types[:, None]
    sign_matrix = torch.sum(sign_matrix, dim = -1)
    sign_matrix[sign_matrix == 0] = -1

    sigma0 = data_uncertainty[None, :, None]
    sigma1 = data_uncertainty[:, None, None]
    k0 = 1 / (2 * sigma0 ** 2)
    k1 = 1 / (2 * sigma1 ** 2)
    x0 = data[None, :, :]
    x1 = data[:, None, :]
    d = displacement[None, None, :]

    if cutoff is not None:
        distance_matrix = torch.sqrt(torch.sum((x0 - x1 + d)**2, dim = -1))
        mask = distance_matrix < cutoff
        mask = mask[:, :, None]
        k0 = k0 * mask
        k1 = k1 * mask
        x0 = x0 * mask
        x1 = x1 * mask

    if kernel == 'gaussian':
        #compute the coefficients of the quadratic equation
        a = k0 + k1
        dx = (x0 + d - x1) 
        b = 2 * k0 * dx
        c = k0 * dx ** 2

        #factor out terms without x
        old_prefactor = 1/(2 * np.pi * sigma0 * sigma1)
        exponent = torch.sum(c - (b**2)/(4*a), dim = -1)
        exponential_prefactor = torch.exp(-exponent)
        new_integral = torch.sqrt(np.pi / a)

        #squeeze dataset for readout
        old_prefactor = torch.squeeze(old_prefactor)
        exponential_prefactor = torch.squeeze(exponential_prefactor)
        new_integral = torch.squeeze(new_integral)

        #compute the integral
        integral = old_prefactor * exponential_prefactor * new_integral * sign_matrix

    return torch.sum(integral)

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

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description='Compute the autocorrelation of the rdfs')
    parser.add_argument('--r_max_mult', type=float, default=4, help='Number to multiply the smallest radius by to get maximum radial distance')
    parser.add_argument('--n_r_bins', type=int, default=100, help='Number of radial bins')
    parser.add_argument('--n_theta_bins', type=int, default=20, help='Number of angular bins')
    parser.add_argument('--n_phi_bins', type=int, default=20, help='Number of azimuthal bins')
    parser.add_argument('--n_space_bins', type=int, default=100, help='Number of bins in space')
    parser.add_argument('--kernel', type=str, default='gaussian', help='Kernel to use for the autocorrelation')
    args = parser.parse_args()
    """

