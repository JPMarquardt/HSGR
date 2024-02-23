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
    device = data.device

    #r_min/max for bins
    r_min = torch.min(distance_matrix[distance_matrix != 0])
    r_max = r_min * r_max_mult
    r_bins = torch.linspace(r_min, r_max, n_r_bins)

    #theta and phi bins
    dth = np.pi / n_theta_bins
    dphi = np.pi / n_phi_bins
    th_min = 0
    phi_min = 0
    th_max = np.pi 
    phi_max = np.pi

    th_bins = torch.linspace(th_min, th_max, n_theta_bins)
    phi_bins = torch.linspace(phi_min, phi_max, n_phi_bins)

    #calculate 2D ADF
    n_types = atom_types.shape[1]
    type_isolation = []

    for i in range(n_types):
        mask = atom_types[:, i] > 0
        mask = mask.unsqueeze(-1).repeat(1, 3)
        type_isolation.append(torch.masked_select(data, mask).view(-1, 3))

    theta_matrix_list = [theta_angle_matrix(data_i) for data_i in type_isolation]
    phi_matrix_list = [phi_angle_matrix(data_i) for data_i in type_isolation]

    for i in range(n_types):
        theta_matrix_list[i][theta_matrix_list[i] < 1e-6] = np.pi
        phi_matrix_list[i][phi_matrix_list[i] < -np.pi/2 + 1e-6] = np.pi

    ADF = torch.ones((n_theta_bins, n_phi_bins))

    for i in range(n_types):
        for th_ind, theta in enumerate(th_bins):
            theta_mask = (theta_matrix_list[i] > theta) & (theta_matrix_list[i] <= theta + dth)
            for phi_ind, phi in enumerate(phi_bins):
                phi_mask = (phi_matrix_list[i] > phi) & (phi_matrix_list[i] <= phi + dphi)
                mask = theta_mask & phi_mask
                ADF[th_ind, phi_ind] *= torch.sum(mask) ** (1/n_types)

    #adf png
    plt.figure()
    sns.heatmap(ADF, cmap = 'viridis')
    plt.xlabel('phi')
    plt.ylabel('theta')
    plt.savefig('ADF.png')

    #cutoff should only be use when the data is very large
    if use_cutoff:
        cutoff = torch.max(uncertainty) * 3
    else: 
        cutoff = None

    auto_corr = torch.zeros((n_r_bins, n_theta_bins, n_phi_bins))
    for r_ind, r in tqdm(enumerate(r_bins), total = n_r_bins):
        for th_ind, theta in enumerate(th_bins):
            for phi_ind, phi in enumerate(phi_bins):
                displacement = spherical2cart(torch.tensor((r, theta, phi))).to(device)
                auto_corr[r_ind, th_ind, phi_ind] = autocorrelation(data=data, 
                                                                    data_uncertainty=uncertainty, 
                                                                    atom_types=atom_types, 
                                                                    displacement=displacement, 
                                                                    kernel=kernel, 
                                                                    cutoff=cutoff)
                #stop multicounting spins that do nothing
                if (theta == 0) and phi_ind != 0:
                    auto_corr[r_ind, th_ind, phi_ind] = 0
        

    tot_ind = find_local_max(auto_corr)
    peak_val = auto_corr[tot_ind[:, 0], tot_ind[:, 1], tot_ind[:, 2]]
    top3_ac, top3_peak_ind = torch.topk(peak_val, 3)
    top_3_tot_ind = tot_ind[top3_peak_ind]

    r = r_bins[top_3_tot_ind[:, 0]]
    theta = th_bins[top_3_tot_ind[:, 1]]
    phi = phi_bins[top_3_tot_ind[:, 2]]

    r = r.unsqueeze(-1)
    theta = theta.unsqueeze(-1)
    phi = phi.unsqueeze(-1)

    output = torch.cat((r, theta, phi), dim = -1)

    return output, top3_ac

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
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

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


def theta_angle_matrix(data):
    """
    atoms x atoms angle matrix: from 1, 0, 0 in the xy plane
    """
    x0_xy = data[None, :, :2] - data[:, None, :2]
    hypotenuse = torch.linalg.vector_norm(x0_xy, dim = -1)

    x0_x = data[None, :, 0] - data[:, None, 0]
    adjacent = x0_x

    sign_y = torch.sign(x0_xy[:, :, 1])
    
    cos_theta = adjacent / hypotenuse

    cos_theta = torch.clamp(cos_theta, -1, 1)

    return torch.acos(cos_theta)*sign_y

def phi_angle_matrix(data):
    """
    atoms x atoms angle matrix: angle above xy plane
    """
    x0_z = data[None, :, 2] - data[:, None, 2]
    adjacent = x0_z

    x0_xyz = data[None, :, :] - data[:, None, :]
    hypotenuse = torch.linalg.vector_norm(x0_xyz, dim = -1)

    sign_z = torch.sign(x0_xyz[:, :, 2])

    cos_phi = adjacent / hypotenuse

    cos_phi = cos_phi.fill_diagonal_(0)

    print(cos_phi.min(), cos_phi.max())

    cos_phi = torch.clamp(cos_phi, -1, 1)

    return torch.acos(cos_phi)

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
    n_atoms = data.shape[0]

    x0 = data[None, :, :].repeat(n_atoms, 1, 1)
    x1 = data[:, None, :].repeat(1, n_atoms, 1)
    d = displacement.repeat(n_atoms**2, 1).view(n_atoms, n_atoms, 3)
    dx = (x0 + d - x1)
    square_distance_matrix = torch.sum(dx**2, dim = -1)[:, :, None]

    if cutoff is not None:
        mask = square_distance_matrix < (cutoff ** 2)
        if mask.sum() == 0:
            return 0

    sigma0 = data_uncertainty[None, :, None].repeat(n_atoms, 1, 1)
    sigma1 = data_uncertainty[:, None, None].repeat(1, n_atoms, 1)

    if cutoff is not None:
        sigma0 = torch.masked_select(sigma0, mask)
        sigma1 = torch.masked_select(sigma1, mask)
        square_distance_matrix = torch.masked_select(square_distance_matrix, mask)

    k0 = 1 / (2 * sigma0 ** 2)
    k1 = 1 / (2 * sigma1 ** 2)

    if kernel == 'gaussian':
        #compute the coefficients of the quadratic equation
        a = k0 + k1
        c = k0 * square_distance_matrix
        b2 = 4 * k0 * c

        #factor out terms without x
        old_prefactor = 1/(2 * np.pi * sigma0 * sigma1)
        exponent = torch.sum(c - (b2)/(4*a), dim = -1)
        exponential_prefactor = torch.exp(-exponent)
        new_integral = torch.sqrt(np.pi / a)

        #squeeze dataset for readout
        old_prefactor = torch.squeeze(old_prefactor)
        exponential_prefactor = torch.squeeze(exponential_prefactor)
        new_integral = torch.squeeze(new_integral)

        #compute the integral
        integral = old_prefactor * exponential_prefactor * new_integral

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



