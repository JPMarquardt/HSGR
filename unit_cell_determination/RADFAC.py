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
from unit_cell_determination.utils import *

import MDAnalysis as mda
from MDAnalysis.transformations import boxdimensions
import gsd.hoomd

"""
Hyperparameters:
- r_max_mult: number to multiply the smallest radius by to get maximum radial distance
- n_r_bins: number of radial bins
- n_theta_bins: number of angular bins
- n_phi_bins: number of azimuthal bins
- kernel: type of kernel
"""


def RA_autocorrelation(data,
                       r_max_mult: float = 4, 
                       n_r_bins: int = 100, 
                       n_angle_bins: int = 20, 
                       extend_r_max: bool = False,
                       kernel: str = 'gaussian',
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
        uncertainty = kwargs.pop('uncertainty')
        atom_types = kwargs.pop('atom_types')

    #get types
    n_types = atom_types.shape[1]
    mask = atom_types.bool()
    device = data.device


    #data is shape (n_atoms, 3) and mask is shape (n_atoms, n_types)
    mask_expand = mask[:, None, :].expand(-1, 3, -1)
    data_expand = data[:, :, None].expand(-1, -1, n_types)
    
    if 'verbose' in kwargs:
        print('Calculating RDF and ADF masks')

    #dataXtype is shape (n_atoms, n_types, 3)
    dataXtype = torch.where(mask_expand, data_expand, torch.nan)

    #get distance, theta, and phi matrices for each of the atom types
    r_matrix = distance_matrix(dataXtype)
    theta_matrix = theta_angle_matrix(dataXtype)
    phi_matrix = phi_angle_matrix(dataXtype)

    #binning
    r_min = torch.min(r_matrix[r_matrix > 0])
    th_min = 0
    phi_min = -np.pi

    r_max = r_min * r_max_mult
    th_max = 2*np.pi
    phi_max = np.pi

    r_bins = torch.linspace(r_min, r_max, n_r_bins)
    th_bins = torch.linspace(th_min, th_max, n_angle_bins)
    phi_bins = torch.linspace(phi_min, phi_max, n_angle_bins)

    #masks
    #maybe try onehot enconding to reduce space complexity
    #calculate masks for each of the atom types

    average_xyz, which_rtp = find_radf_peaks(r_matrix=r_matrix, 
                                             theta_matrix=theta_matrix, 
                                             phi_matrix=phi_matrix, 
                                             r_bins=r_bins, 
                                             th_bins=th_bins, 
                                             phi_bins=phi_bins, 
                                             uncertainty=uncertainty, 
                                             **kwargs)
    
    candidates, peak_val = find_auto_correlation_peaks(which_rtp, 
                                                       average_xyz, 
                                                       data, 
                                                       uncertainty, 
                                                       atom_types, 
                                                       n_r_bins, 
                                                       n_angle_bins, 
                                                       kernel, 
                                                       **kwargs)

    #get the top 3 peaks from the RAAC
    if candidates.shape[0] < 3 and extend_r_max:
        r_interval = r_max - r_min
        r_min = r_max
        r_max = r_max + (r_interval * extend_r_max)
        r_bins = torch.linspace(r_min, r_max, int(n_r_bins * extend_r_max))

        average_xyz_temp, which_rtp_temp = find_radf_peaks(r_matrix, 
                                                           theta_matrix, 
                                                           phi_matrix, 
                                                           r_bins, 
                                                           th_bins, 
                                                           phi_bins, 
                                                           uncertainty, 
                                                           **kwargs)
        
        candidates_temp, peak_val_temp = find_auto_correlation_peaks(which_rtp_temp, 
                                                                     average_xyz_temp, 
                                                                     data, 
                                                                     uncertainty, 
                                                                     atom_types, 
                                                                     n_r_bins, 
                                                                     n_angle_bins, 
                                                                     kernel, 
                                                                     **kwargs)
        
        average_xyz = torch.cat((average_xyz, average_xyz_temp), dim = 0)
        which_rtp = which_rtp + which_rtp_temp
        candidates = torch.cat((candidates, candidates_temp), dim = 0)
        peak_val = torch.cat((peak_val, peak_val_temp), dim = 0)

    top3_RAAC, top3_peak_ind = torch.topk(peak_val, 3)

    #get the xyz of those peaks
    top3_xyz = torch.zeros(3, 3)
    for i in range(3):
        rtp = candidates[top3_peak_ind[i]]
        key = tuple(rtp.tolist())
        which_rtp_ind = list(which_rtp).index(key)
        top3_xyz[i] = average_xyz[which_rtp_ind]

    return top3_xyz, top3_RAAC

def autocorrelation(data: torch.tensor,
                    data_uncertainty: torch.tensor,
                    atom_types: torch.tensor,
                    displacement: torch.tensor, 
                    kernel: str = 'gaussian',
                    cutoff: float = None,
                    beta: float = 1.0,
                    n_observations: int = 1):
    """
    Compute the autocorrelation of the data with a displacement
    Add math explanation below
    """
    n_atoms = data.shape[0]
    if cutoff is None:
        cutoff = torch.max(data_uncertainty) * 3

    #make the displacement the same shape as the data so they can be added
    x0 = data[None, :, :].repeat(n_atoms, 1, 1)
    x1 = data[:, None, :].repeat(1, n_atoms, 1)
    d = displacement.repeat(n_atoms**2, 1).view(n_atoms, n_atoms, 3)
    dx = (x0 + d - x1)
    square_distance_matrix = torch.sum(dx**2, dim = -1)[:, :, None]

    mask = square_distance_matrix < (cutoff ** 2)
    if mask.sum() == 0:
        return 0

    sigma0 = data_uncertainty[None, :, None].repeat(n_atoms, 1, 1)
    sigma1 = data_uncertainty[:, None, None].repeat(1, n_atoms, 1)

    sigma0 = sigma0[mask]
    sigma1 = sigma1[mask]
    square_distance_matrix = square_distance_matrix[mask]

    if kernel == 'gaussian':
        k0 = 1 / (2 * sigma0 ** 2)
        k1 = 1 / (2 * sigma1 ** 2)

        type0 = atom_types.bool()[None, :, :].repeat(n_atoms, 1, 1)
        type1 = atom_types.bool()[:, None, :].repeat(1, n_atoms, 1)
        type_mask = type0 & type1
        type_mask = type_mask.sum(dim = -1) > 0
        type_matrix = torch.where(type_mask, 1, -1)[mask.squeeze()]

        #compute the coefficients of the quadratic equation
        a = k0 + k1
        c = k0 * square_distance_matrix
        b2 = 4 * k0 * c

        #factor out terms without x
        old_prefactor = 1/(2 * np.pi * sigma0 * sigma1)
        exponent = torch.sum(c - (b2)/(4*a), dim = -1)
        exponential_prefactor = torch.exp(-beta * exponent)
        new_integral = torch.sqrt(np.pi / a)

        #squeeze dataset for readout
        old_prefactor = torch.squeeze(old_prefactor)
        exponential_prefactor = torch.squeeze(exponential_prefactor)
        new_integral = torch.squeeze(new_integral)

        #compute the integral
        integral = old_prefactor * exponential_prefactor * new_integral * type_matrix

    return torch.sum(integral)

def z_test(data: torch.tensor,
           data_uncertainty: torch.tensor,
           atom_types: torch.tensor,
           displacement: torch.tensor,
           significance: float = 0.05):
    """
    Compute the z-test for the distance matrix
    """
    norm = torch.distributions.Normal(0, 1)
    z_value_comparison = norm.icdf(1 - significance)
    cutoff = 2 * torch.max(data_uncertainty) * z_value_comparison

    x0 = data[None, :, :].repeat(n_atoms, 1, 1)
    x1 = data[:, None, :].repeat(1, n_atoms, 1)
    d = displacement.repeat(n_atoms**2, 1).view(n_atoms, n_atoms, 3)
    dx = (x0 + d - x1)
    square_distance_matrix = torch.sum(dx**2, dim = -1)[:, :, None]

    mask = square_distance_matrix < (cutoff ** 2)
    if mask.sum() == 0:
        return 0

    distance_matrix = torch.sqrt(square_distance_matrix[mask])

    n_atoms = distance_matrix.shape[0]
    sigma0 = data_uncertainty[None, :, None].repeat(n_atoms, 1, 1)
    sigma1 = data_uncertainty[:, None, None].repeat(1, n_atoms, 1)
    sigma0 = sigma0[mask]
    sigma1 = sigma1[mask]

    sigma_Xbar = torch.sqrt((sigma0**2 + sigma1**2))

    z_value = (distance_matrix / sigma_Xbar).squeeze()

    type0 = atom_types.bool()[None, :, :].repeat(n_atoms, 1, 1)
    type1 = atom_types.bool()[:, None, :].repeat(1, n_atoms, 1)
    type_mask = type0 & type1
    type_mask = type_mask.sum(dim = -1) > 0
    type_matrix = torch.where(type_mask, 1, -1)[mask.squeeze()]

    z_bool = z_value < z_value_comparison
    z_bool = z_bool * type_matrix
    z_sum = torch.sum(z_bool)

    return z_sum


def multi_mask(valueXtype: torch.tensor, bins: torch.tensor):
    """
    Compute the mask a value and 
    """
    n_bins = bins.shape[0]
    dx = bins[1] - bins[0]
    x_mask = torch.zeros((n_bins, valueXtype.shape[0], valueXtype.shape[1], valueXtype.shape[2]), dtype = torch.bool)
    for x_ind, x in enumerate(bins):
        x_new = x - dx/2
        x_mat = (valueXtype > x_new) & (valueXtype <= x_new + dx)
        x_mask[x_ind] = x_mat

    return x_mask

def masks2RADF(r_mask: torch.tensor, t_mask: torch.tensor, p_mask: torch.tensor):
    """
    Compute the Radial Angular Distribution Function (RADF) from the masks
    """

    maskDict = {}
    for r_ind, r_mask_i in tqdm(enumerate(r_mask)):
        if check_mask_zeros(r_mask_i):
            continue

        for th_ind, th_mask_i in enumerate(t_mask):
            r_th_mask = r_mask_i & th_mask_i

            if check_mask_zeros(r_th_mask):
                continue
            
            for phi_ind, p_mask_i in enumerate(p_mask):
                r_th_phi = r_th_mask & p_mask_i
                
                if check_mask_zeros(r_th_phi):
                    continue

                hash_key = (r_ind, th_ind, phi_ind)
                maskDict[hash_key] = r_th_phi

    return maskDict

def find_radf_peaks(r_matrix, theta_matrix, phi_matrix, r_bins, th_bins, phi_bins, uncertainty, **kwargs) -> tuple[torch.tensor, list]:
    r_mask = multi_mask(r_matrix, r_bins)
    t_mask = multi_mask(theta_matrix, th_bins)
    p_mask = multi_mask(phi_matrix, phi_bins)

    t_mask = t_mask & (r_matrix < r_bins[-1])
    p_mask = p_mask & (r_matrix < r_bins[-1])

    #caulcuate RADF
    if 'verbose' in kwargs:
        print('Calculating RADF')

    maskDict = masks2RADF(r_mask, t_mask, p_mask)

    which_rtp = list(maskDict.keys())
    n_rtp = len(which_rtp)
    average_xyz = torch.zeros((n_rtp, 3))
    
    if 'verbose' in kwargs:
        print(f'Found {n_rtp} candidate vectors')
        print('Calculating averaged peaks')

    for i, key in enumerate(which_rtp):
        lower_triangle_bool = torch.tril(torch.ones(r_matrix.shape), diagonal = -1).bool()

        rtp_mask = maskDict[key] & lower_triangle_bool

        r = r_matrix[rtp_mask].unsqueeze(-1)
        theta = theta_matrix[rtp_mask].unsqueeze(-1)
        phi = phi_matrix[rtp_mask].unsqueeze(-1)

        rtp = torch.cat((r, theta, phi), dim = -1)

        xyz = spherical2cart(rtp)
        average_xyz[i] = xyz.mean(dim = 0)
        
    #initizlize cutoff (rarely used, but can be useful for large systems with high resolution)
    cutoff = torch.max(uncertainty) * 3

    #filter out any vectors that are the negative of another vector
    for i in range(n_rtp):
        for j in range(i+1, n_rtp):
            if torch.sum(torch.abs(average_xyz[i] + average_xyz[j])) < cutoff:
                average_xyz[j] = torch.tensor([np.nan, np.nan, np.nan])

    return average_xyz, which_rtp
    

def find_auto_correlation_peaks(which_rtp, average_xyz, data, uncertainty, atom_types, n_r_bins, n_angle_bins, kernel, **kwargs):
    if 'verbose' in kwargs:
        print('Calculating kernel RAAC')

    #initizlize cutoff (rarely used, but can be useful for large systems with high resolution)
    comparison_func = z_test
    cutoff = torch.max(uncertainty) * 3

    raac = torch.zeros((n_r_bins, n_angle_bins, n_angle_bins))
    for rtp_ind, displacement in zip(which_rtp, average_xyz):
        r_ind = rtp_ind[0]
        th_ind = rtp_ind[1]
        phi_ind = rtp_ind[2]

        raac[r_ind, th_ind, phi_ind] = comparison_func(data, uncertainty, atom_types, displacement, **kwargs)
 
    #return a list of all maxima in the RAAC
    if 'verbose' in kwargs:
        print('Finding top 3 maxima of RAAC')
    candidates = find_local_max(raac)

    #get the values of RAAC at those peaks
    n_local_max = len(candidates)
    peak_val = torch.zeros(n_local_max)
    for i in range(n_local_max):
        peak_val[i] = raac[tuple(candidates[i])]

    return candidates, peak_val

def kz_test(data: torch.tensor,
            k: int = 50,
            **kwargs):
    """
    Compute ztest on knn data
    """
    if type(data) == mda.Universe:
        data = torch.mean(torch.from_numpy(data.coord.positions), dim=-1)
        uncertainty = torch.std(torch.from_numpy(data.coord.positions), dim =-1)
        atom_types = torch.from_numpy(data.atoms.types)
        #atom type needs to be changed to categorical for more than 2 atom types
    else:
        uncertainty = kwargs.pop('uncertainty')
        atom_types = kwargs.pop('atom_types')

    #get types
    n_types = atom_types.shape[1]
    mask = atom_types.bool()
    device = data.device


    #data is shape (n_atoms, 3) and mask is shape (n_atoms, n_types)
    mask_expand = mask[:, None, :].expand(-1, 3, -1)
    data_expand = data[:, :, None].expand(-1, -1, n_types)
    
    if 'verbose' in kwargs:
        print('Calculating RDF and ADF masks')

    #dataXtype is shape (n_atoms, n_types, 3)
    dataXtype = torch.where(mask_expand, data_expand, torch.nan)

    

    return 

def get_top_k_displacement(data, k):
    """
    Get the top k radii from the data
    """
    n_atoms = data.shape[0]
    x0 = data[None, :, :].repeat(n_atoms, 1, 1)
    x1 = data[:, None, :].repeat(1, n_atoms, 1)
    dx = x0 - x1
    square_distance_matrix = torch.sum(dx**2, dim = -1)
    torch.fill_diagonal_(square_distance_matrix, float('inf'))

    #get the top k radii
    top_k_radii, top_k_radii_ind = torch.topk(square_distance_matrix, k, largest=False, dim=0)

    for i in range(n_atoms):
        for j in range(square_distance_matrix.dim()):
            dx.index_select(j+1, top_k_radii_ind[j])
        dx[i][top_k_radii_ind[i]]
        top_k_radii_ind[i] = 

    return dx[top_k_radii_ind]