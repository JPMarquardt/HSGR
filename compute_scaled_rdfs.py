import os
import sys
import numpy as np
import argparse
import MDAnalysis as mda
from MDAnalysis.transformations import boxdimensions
from scipy.stats import entropy
import matplotlib as plt
import torch
import torch.nn as nn
from jarvis.db.figshare import data as jdata


#use argparse instead of 
parser = argparse.ArgumentParser()
parser.add_argument('--files', metavar='f', type = list, nargs='+', help='Use this argument to specify file directory of .xyz files')
args = parser.parse_args()

if len(args.files) != 2:
    raise(Exception('Wrong number of .xyz files'))

def get_minimum_pairwise_distance(u):
    """Compute the minimum pairwise distance between atoms in a system.

    Parameters
    ----------
    u : MDAnalysis Universe
        The system to compute the average pairwise distance for.

    Returns
    -------
    min_dist : float
        The minimum pairwise distance between atoms in the system.
    """
    # Get the pairwise distances between all atoms in the system.
    dists = u.atoms.positions - u.atoms.positions[:, np.newaxis]
    dists = np.sqrt((dists**2).sum(axis=2))
    min_dist = np.min(dists[dists>0])
    return min_dist

def compute_scaled_rdf(u, scaled_maximum=4,n_rdf_bins=100, random_noise_scale=0):
    from MDAnalysis.analysis import rdf as compute_rdf
    # compute the minimum pairwise distance
    min_dist = get_minimum_pairwise_distance(u)

#    # for every frame in the trajectory, center
#    for ts in u.trajectory:
#        #print(ts.positions.mean(axis=0))
#        ts.positions -= ts.positions.mean(axis=0)
#        #print(ts.positions.mean(axis=0))
#        if random_noise_scale > 0:
#            #print(f'adding noise of size {random_noise_scale*min_dist} to positions')
#            ts.positions += np.random.normal(scale=random_noise_scale*min_dist,size=ts.positions.shape)
#            #print(ts.positions.mean(axis=0))

    u.atoms.positions -= u.atoms.positions.mean(axis=0)
    if random_noise_scale > 0:
        u.atoms.positions += np.random.normal(scale=random_noise_scale*min_dist,size=u.atoms.positions.shape)


    #get maximum distance from the center, ignoring pbc
    dr2 = u.atoms.positions**2
    maximum_distance = np.max(np.sqrt(dr2))

    dist_sel = u.select_atoms(f'point 0 0 0 {maximum_distance/3}')

    # using MDAnalysis, compute the RDF from 0.01 to some radial cutoff
    rdf = compute_rdf.InterRDF(dist_sel, dist_sel, range=(0.01, scaled_maximum*min_dist), nbins=n_rdf_bins)
    rdf.run()
    # get the rdf data
    rdf_data = rdf.results.rdf
    # get the rdf bins
    rdf_bins = rdf.results.bins

    # get the first peak position by finding the first zero crossing of rdf_data-1
    #peak_pos = rdf_bins[np.where(np.diff(np.sign(rdf_data-rdf_data.max()/10.)))[0][0]]
    peak_pos = rdf_bins[np.argmax(rdf_data)]

    # using MDAnalysis, compute the RDF from 0.01 to scaled_maximum times the peak distance
    rdf2 = compute_rdf.InterRDF(dist_sel, dist_sel, range=(0.01*peak_pos, scaled_maximum*peak_pos), nbins=n_rdf_bins)
    rdf2.run()
    # get the rdf data
    rdf_data2 = rdf2.results.rdf
    # get the rdf bins
    rdf_bins2 = rdf2.results.bins
    
    #new_peak_pos = rdf_bins2[np.where(np.diff(np.sign(rdf_data2-rdf_data2.max()/10.)))[0][0]]
    new_peak_pos = rdf_bins2[np.argmax(rdf_data2)]

    # using MDAnalysis, compute the RDF from 0.01 to scaled_maximum times the peak distance
    rdf3 = compute_rdf.InterRDF(dist_sel, dist_sel, range=(0.01*peak_pos, scaled_maximum*new_peak_pos), nbins=n_rdf_bins)
    rdf3.run()
    # get the rdf data
    rdf_data3 = rdf3.results.rdf
    # get the rdf bins
    rdf_bins3 = rdf3.results.bins

    # get the rdf bin centers
    return rdf_bins3, rdf_data3, new_peak_pos

class universeBox():
    def __init__(self, files, **kwargs):
        self.files = files
        self.filetail = os.path.basename(files[0])

        self.universes = {}
        for file in files:
            self.universes[file] = mda.Universe(file)

        self.size_check()
        self.add_trans()

        self.rdf_bins = {}
        self.rdf_data = {}
        self.peak_pos = {}
        
        self.rdf_bins[file], self.rdf_data[file], self.peak_pos[file] = compute_scaled_rdf(self.universes[file])
        print(entropy(self.rdf_data/self.rdf_data.max()+0.01,self.rdf_data/self.rdf_data.max()+0.01),self.files)

    def size_check(self):
        for file in self.universes:
            if len(self.universes[file].atoms.positions)>1e6:
                print(f"file {file} has too many lines")
                sys.exit(2)

    def add_trans(self):
        for file in self.universes:
            box_length = (self.universes[file].trajectory[0].positions.max(axis=0)-self.universes[file].trajectory[0].positions.min(axis=0))*1.5
            trans = boxdimensions.set_dimensions([box_length[0],box_length[1],box_length[2],90,90,90])
            self.universes[file].trajectory.add_transformations(trans)
            
    def plot(self):
        plt.plot(self.rdf_bins/self.peak_pos, self.rdf_data/self.rdf_data.max(),label='atomic')
        plt.plot(self.rdf_bins/self.peak_pos, self.rdf_data/self.rdf_data.max(),label='simulation')
        plt.xlabel('r/peak_pos')
        plt.ylabel('g(r)')
        plt.legend(loc=0)
        plt.show()

    def train_gnn(self):



        self.gnn = JPsGNN()

class JPsGNN(nn.Module):
    def __init__(self):
        super(JPsGNN, self).__init__()


    def forward(self, x):
        x
        


class kqvAttn(nn.Module):
    """
    Key Query Value Attention
    i2k = input 2 key
    i2q = input 2 query
    i2v = input 2 value

    This module takes in g by inputD tensors and outputs g by outputD tensors.
    g is the number of nodes in your graph.
    inputD is the number of node features in your input.
    outputD is the number of node features in your output.
    """
    def __init__(self, inputD, outputD, keyQueryEmbeddingD = 128):
        super(kqvAttn, self).__init__()
        #initalize key query value
        self.i2k = nn.Linear(inputD, keyQueryEmbeddingD)
        self.i2q = nn.Linear(inputD, keyQueryEmbeddingD)
        self.i2v = nn.Linear(inputD, outputD)

    def forward(self, x):
        key = self.i2k(x)
        query = self.i2q(x)
        value = self.i2v(x)

        attnCoef = torch.matmul(key, torch.transpose(query, 0, 1))
        attnCoefNorm = nn.functional.normalize(attnCoef, dim = -1)
        attnCoefSM = nn.Softmax(attnCoefNorm)

        attnSum = torch.matmul(attnCoefSM, value)
        output = attnSum + value

        return output






