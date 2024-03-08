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

    #get types
    n_types = atom_types.shape[1]
    mask = atom_types.bool()

    #data is shape (n_atoms, 3) and mask is shape (n_atoms, n_types)
    mask_expand = mask[:, None, :].expand(-1, 3, -1)
    data_expand = data[:, :, None].expand(-1, -1, n_types)
    

    #dataXtype is shape (n_atoms, n_types, 3)
    dataXtype = torch.where(mask_expand, data_expand, torch.nan)
    
    #isolates the data for each atom type
    def dataXtype2data_i(dataXtype, i):
        data = dataXtype[:, i, :]
        data = data[~torch.isnan(data).any(dim = -1)]
        return data

    #get distance, theta, and phi matrices for each of the atom types
    r_matrix = distance_matrix(dataXtype)
    theta_matrix = theta_angle_matrix(dataXtype)
    phi_matrix = phi_angle_matrix(dataXtype)

    #device
    device = data.device

    #binning
    r_min = torch.min(r_matrix[r_matrix > 0])
    th_min = -np.pi
    phi_min = 0

    r_max = r_min * r_max_mult
    th_max = np.pi
    phi_max = 2*np.pi

    dr = (r_max - r_min) / n_r_bins
    dth = (th_max - th_min) / n_theta_bins
    dphi = (phi_max - phi_min) / n_phi_bins

    r_bins = torch.linspace(r_min, r_max, n_r_bins)
    th_bins = torch.linspace(th_min, th_max, n_theta_bins)
    phi_bins = torch.linspace(phi_min, phi_max, n_phi_bins)

    #masks
    maskDict = {}
    r_mask = torch.zeros((n_r_bins, r_matrix.shape[0], r_matrix.shape[1], n_types), dtype = torch.bool)
    theta_mask = torch.zeros((n_theta_bins, r_matrix.shape[0], r_matrix.shape[1], n_types), dtype = torch.bool)
    phi_mask = torch.zeros((n_phi_bins, r_matrix.shape[0], r_matrix.shape[1], n_types), dtype = torch.bool)

    #calculate masks for each of the atom types
    print('Calculating RDF and ADF')
    for r_ind, r in enumerate(r_bins):
        r_new = r - dr/2
        r_mat = (r_matrix > r_new) & (r_matrix <= r_new + dr)
        r_mask[r_ind] = r_mat

    for th_ind, th in enumerate(th_bins):
        th_new = th
        th_mat = (theta_matrix > th_new) & (theta_matrix <= th_new + dth)
        theta_mask[th_ind] = th_mat

    for phi_ind, phi in enumerate(phi_bins):
        phi_new = phi
        phi_mat = (phi_matrix > phi_new) & (phi_matrix <= phi_new + dphi)
        phi_mask[phi_ind] = phi_mat

    print('Calculating RADF')
    #caulcuate RADF
    for r_ind, r_mask_i in tqdm(enumerate(r_mask)):
        if check_mask_zeros(r_mask_i):
            continue

        for th_ind, th_mask_i in enumerate(theta_mask):
            r_th_mask = r_mask_i & th_mask_i

            if check_mask_zeros(r_th_mask):
                continue
            
            for phi_ind, phi_mask_i in enumerate(phi_mask):
                r_th_phi = r_th_mask & phi_mask_i
                
                if check_mask_zeros(r_th_phi):
                    continue

                hash_key = (r_ind, th_ind, phi_ind)
                maskDict[hash_key] = r_th_phi

    which_rtp = maskDict.keys()
    print(which_rtp)
    average_xyz = torch.zeros((len(which_rtp), 3))

    print('Calculating averaged peaks')
    for j, r_th_phi_ind in enumerate(which_rtp):
        r_ind = r_th_phi_ind[0]
        th_ind = r_th_phi_ind[1]
        phi_ind = r_th_phi_ind[2]

        lower_triangle_bool = torch.tril(torch.ones(r_matrix.shape), diagonal = -1).bool()

        r_mask_lower = r_mask[r_ind] & lower_triangle_bool
        theta_mask_lower = theta_mask[th_ind] & lower_triangle_bool
        phi_mask_lower = phi_mask[phi_ind] & lower_triangle_bool

        r = r_matrix[r_mask_lower]
        theta = theta_matrix[theta_mask_lower]
        phi = phi_matrix[phi_mask_lower]

        r = r.unsqueeze(-1)
        theta = theta.unsqueeze(-1)
        phi = phi.unsqueeze(-1)

        print(r.shape, theta.shape, phi.shape)

        rtp = torch.cat((r, theta, phi), dim = -1)

        average_xyz[j] = spherical2cart(rtp).mean(dim = 0)
        print(average_xyz[j])

    print(average_xyz)

    #initizlize cutoff (rarely used, but can be useful for large systems with high resolution)
    if use_cutoff:
        cutoff = torch.max(uncertainty) * 3
    else:
        cutoff = None

    print('Calculating kernel RAAC')
    RAAC = torch.zeros((n_r_bins, n_theta_bins, n_phi_bins))
    for rtp_ind, xyz in zip(which_rtp, average_xyz):
        r_ind = rtp_ind[0]
        th_ind = rtp_ind[1]
        phi_ind = rtp_ind[2]
        displacement = xyz
        print(rtp, displacement)
        RAAC[r_ind, th_ind, phi_ind] += autocorrelation(data, uncertainty, atom_types, displacement, kernel = kernel, cutoff = cutoff)
 

    #adf png
    candidates = find_local_max(RAAC)
    peak_val = RAAC[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
    top3_ac, top3_peak_ind = torch.topk(peak_val, 3)
    top_3_tot_ind = candidates[top3_peak_ind]

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
    full_distance_matrix = distance_matrix(data)

    hypotenuse = distance_matrix(data[:,:2])

    adjacent = data[None, :, 0] - data[:, None, 0]

    sign_y = torch.sign(data[:, None, 1] - data[None, :, 1])
    
    cos_theta = adjacent / hypotenuse

    non_diagonal_nans = (hypotenuse == 0) & (full_distance_matrix != 0)

    cos_theta[non_diagonal_nans] = 1

    cos_theta = torch.clamp(cos_theta, -1, 1)

    return torch.acos(cos_theta) * sign_y

def phi_angle_matrix(data):
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
        sigma0 = sigma0[mask]
        sigma1 = sigma1[mask]
        square_distance_matrix = square_distance_matrix[mask]

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



