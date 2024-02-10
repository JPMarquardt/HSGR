import os
import sys
import numpy as np
import argparse
from scipy.stats import entropy

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
#import tkinter
matplotlib.use('Agg')

import torch
import torch.nn as nn
from jarvis.db.figshare import data as jdata
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
                       n_space_bins: int = 100,
                       kernel: str = 'gaussian',
                       **kwargs):
    """
    Compute the autocorrelation spatial radial x angular function of the RDFs
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
    r_bins = torch.linspace(r_min, r_max, n_r_bins+1)

    #caluclate RDF
    for ind, (r_lo, r_hi) in enumerate(zip(r_bins[:-1], r_bins[1:])):
        mask = (distance_matrix >= r_lo) & (distance_matrix < r_hi)
        RDF[ind] = mask.sum().div(data.shape[0] * data.shape[1])

    #find the peaks
    RDF_peaks = find_local_max(RDF)
    print(RDF_peaks)

    #atoms x atoms angle matrices from 1, 0, 0
    tam = theta_angle_matrix(data)
    pam = phi_angle_matrix(data)

    #theta and phi bins
    th_min = 0
    th_max = np.pi
    phi_min = 0
    phi_max = np.pi

    th_bins = torch.linspace(th_min, th_max, n_theta_bins+1)
    phi_bins = torch.linspace(phi_min, phi_max, n_phi_bins+1)

    #calculate the ADF
    adf = torch.zeros((n_theta_bins, n_phi_bins))

    for th_ind, (th_lo, th_hi) in enumerate(zip(th_bins[:-1], th_bins[1:])):
        for phi_ind, (phi_lo, phi_hi) in enumerate(zip(phi_bins[:-1], phi_bins[1:])):
            theta_mask = (tam >= th_lo) & (tam < th_hi)
            phi_mask = (pam >= phi_lo) & (pam < phi_hi)
            mask = theta_mask & phi_mask
            adf[th_ind, phi_ind] = mask.sum().div(data.shape[0] * data.shape[1])

    ANG_peaks = find_local_max(adf)
    print(ANG_peaks)

    #compute the autocorrelation
    auto_corr = torch.zeros((RDF_peaks.shape[0], ANG_peaks.shape[0]))

    for r_ind, r in enumerate(RDF_peaks):
        for ang_ind, th_phi in enumerate(ANG_peaks):
            theta, phi = th_phi
            displacement = torch.tensor([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])                

            auto_corr[r_ind, ang_ind] = autocorrelation(data, uncertainty, atom_types, displacement, kernel)



    _, max_ac_ind = torch.topk(auto_corr, 3)
    r = max_ac_ind[0] * (r_max - r_min) / n_r_bins + r_min
    angle = max_ac_ind[1] * (np.pi) / n_theta_bins

    return r, angle

def theta_angle_matrix(data: torch.tensor):
    """
    atoms x atoms angle matrix: from 1, 0, 0 in the xy plane
    """ 
    x0 = data[None, :, :2] - data[:, None, :2]
    x0_norm = torch.norm(x0, dim = -1)

    x1 = torch.tensor([1, 0])
    x1 = x1[None, None, :]
    
    x0_x1 = torch.sum(x0 * x1, dim = -1)
    cos_theta = x0_x1 / x0_norm

    return torch.acos(cos_theta)

def phi_angle_matrix(data: torch.tensor):
    """
    atoms x atoms angle matrix: angle above xy plane
    """
    x0_xy = data[None, :, :2] - data[:, None, :2]
    adjacent = torch.linalg.vector_norm(x0_xy, dim = -1)

    x0_xyz = data[None, :, :] - data[:, None, :]
    hypotenuse = torch.linalg.vector_norm(x0_xyz, dim = -1)

    sign_z = torch.sign(x0_xyz[:, :, 2])

    cos_phi = adjacent / hypotenuse

    return torch.acos(cos_phi) * sign_z

def autocorrelation(data: torch.tensor,
                    data_uncertainty: torch.tensor,
                    atom_types: torch.tensor,
                    displacement: torch.tensor, 
                    kernel: str = 'gaussian'):
    """
    Compute the autocorrelation of the data with a displacement
    Add math explanation below
    """
    if kernel == 'gaussian':
        sign_matrix = torch.sign(atom_types[None, :] * atom_types[:, None])

        sigma0 = data_uncertainty[None, :, None]
        sigma1 = data_uncertainty[:, None, None]
        k0 = 1 / (2 * sigma0 ** 2)
        k1 = 1 / (2 * sigma1 ** 2)
        x0 = data[None, :, :]
        x1 = data[:, None, :]
        d = displacement[None, None, :]

        a = k0 + k1
        b = 2 * ((k0*x0) + (k0*d) + (k1*x1)) 
        c = (k0 * (x0*x0 + d*d + 2 * x0*d)) + (k1*x1*x1)

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
    if DF.dim() == 1:
        delta = DF.roll(1) - DF
        delta_sign = torch.sign(delta)
        delta_delta_sign = delta_sign.roll(1) - delta_sign
        peaks = delta_delta_sign == -2 #-2 is the sign of the second derivative
        peak_ind = peaks.nonzero()

    elif DF.dim() == 2:
        #this is pretty ugly, should be refactored in future to incorporate nD
        delta_x = DF - DF.roll(1, dims=0)
        delta_y = DF - DF.roll(1, dims=1)
        delta_sign_x = delta_x.sign()
        delta_sign_y = delta_y.sign()
        delta_delta_sign_x = delta_sign_x - delta_sign_x.roll(1, dims=0)
        delta_delta_sign_y = delta_sign_y - delta_sign_y.roll(1, dims=1)
        x_peaks = delta_delta_sign_x == -2
        y_peaks = delta_delta_sign_y == -2
        x_peaks = x_peaks.roll(-1, dims=0)
        y_peaks = y_peaks.roll(-1, dims=1)
        peaks = torch.logical_and(x_peaks, y_peaks)
        peak_ind = peaks.nonzero()

    else:
        raise ValueError('Distribution function must be 1D or 2D')
    
    return peak_ind

