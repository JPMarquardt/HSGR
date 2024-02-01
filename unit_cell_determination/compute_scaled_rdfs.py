import os
import sys
import numpy as np
import argparse
from scipy.stats import entropy
import matplotlib as plt
import torch
import torch.nn as nn
from jarvis.db.figshare import data as jdata

import MDAnalysis as mda
from MDAnalysis.transformations import boxdimensions


def RA_autocorrelation(data, r_max: float = None, n_r_bins: float = 100, n_theta_bins = 100):
    """
    Compute the autocorrelation spatial radial x angular function of the RDFs
    """
    if type(data) == mda.Universe:
        data = torch.from_numpy(data.coord.positions)



    return 

