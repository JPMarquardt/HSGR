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
                       n_theta_bins: int = 20, 
                       n_phi_bins: int = 20, 
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

    #get distance, theta, and phi matrices for each of the atom types
    r_matrix = distance_matrix(dataXtype)
    theta_matrix = theta_angle_matrix(dataXtype)
    phi_matrix = phi_angle_matrix(dataXtype)

    #device
    device = data.device

    #binning
    r_min = torch.min(r_matrix[r_matrix > 0])
    th_min = 0
    phi_min = -np.pi

    r_max = r_min * r_max_mult
    th_max = 2*np.pi
    phi_max = np.pi

    dr = (r_max - r_min) / n_r_bins
    dth = (th_max - th_min) / n_theta_bins
    dphi = (phi_max - phi_min) / n_phi_bins

    r_bins = torch.linspace(r_min, r_max, n_r_bins)
    th_bins = torch.linspace(th_min, th_max, n_theta_bins)
    phi_bins = torch.linspace(phi_min, phi_max, n_phi_bins)

    #masks
    #maybe try onehot enconding to reduce space complexity
    #calculate masks for each of the atom types
    print('Calculating RDF and ADF masks')
    r_mask = multi_mask(r_matrix, r_bins, dr)
    t_mask = multi_mask(theta_matrix, th_bins, dth)
    p_mask = multi_mask(phi_matrix, phi_bins, dphi)

    print('Calculating RADF')
    #caulcuate RADF

    maskDict = masks2RADF(r_mask, t_mask, p_mask)

    which_rtp = maskDict.keys()
    n_rtp = len(which_rtp)
    average_xyz = torch.zeros((n_rtp, 3))
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

    print('Calculating kernel RAAC')
    RAAC = torch.zeros((n_r_bins, n_theta_bins, n_phi_bins))
    for rtp_ind, displacement in zip(which_rtp, average_xyz):
        r_ind = rtp_ind[0]
        th_ind = rtp_ind[1]
        phi_ind = rtp_ind[2]

        RAAC[r_ind, th_ind, phi_ind] = autocorrelation(data, uncertainty, atom_types, displacement, kernel = kernel, cutoff = cutoff)
 
    #return a list of all maxima in the RAAC
    print('Finding top 3 maxima of RAAC')
    candidates = find_local_max(RAAC)

    #get the values of RAAC at those peaks
    n_local_max = len(candidates)
    peak_val = torch.zeros(n_local_max)
    for i in range(n_local_max):
        peak_val[i] = RAAC[tuple(candidates[i])]

    #get the top 3 peaks from the RAAC
    top3_RAAC, top3_peak_ind = torch.topk(peak_val, 3)

    #get the xyz of those peaks
    top3_xyz = torch.zeros(3, 3)
    for i in range(3):
        rtp = candidates[top3_peak_ind[i]]
        key = tuple(rtp.tolist())
        which_rtp_ind = list(which_rtp).index(key)
        top3_xyz[i] = average_xyz[which_rtp_ind]

    output = top3_xyz

    return output, top3_RAAC

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

def multi_mask(valueXtype: torch.tensor, bins: torch.tensor, dx: float):
    """
    Compute the mask a value and 
    """
    n_bins = bins.shape[0]
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