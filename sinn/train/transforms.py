import nfflr
import torch
import torch.nn as nn

from sinn.graph.graph import create_supercell, create_labeled_supercell, create_knn_graph, lattice_plane_slicer, create_periodic_graph
from sinn.noise.gaussian_noise import noise_regression
from sinn.simulation.simulation import box_filter

from typing import Callable

def noise_regression_prep(a: nfflr.Atoms, n_target_atoms: int, noise: Callable = None, k: int = 9):
    coords = a.positions
    lattice = a.cell
    numbers = a.numbers

    data = torch.matmul(coords, torch.inverse(lattice))

    replicates = (n_target_atoms / data.size()[0]) ** (1/3)
    replicates = int(replicates)

    miller_index = torch.randint(0, 4, (3,))

    if noise is None:
        noise = lambda x: x

    supercell = create_supercell(data, replicates)
    sample_noise, supercell = noise_regression(supercell, noise)
    supercell = lattice_plane_slicer(supercell, miller_index, replicates)
    supercell = supercell @ lattice

    g = create_knn_graph(supercell, k=k, line_graph=False)
    numbers = numbers.repeat(replicates**3)
    g.ndata['z'] = numbers

    return g, sample_noise

def noise_regression_sim_prep(a: nfflr.Atoms, k: int = 9):
    data = a.positions
    lattice = a.cell
    numbers = a.numbers
    replicates = 3

    dx = 0.1 * torch.min(torch.norm(lattice, dim=1))
    supercell, atom_id, cell_id = create_labeled_supercell(data, n=replicates, lattice=lattice)
    numbers = numbers.repeat(replicates**3)
    filt = box_filter(supercell, lattice, dx)

    supercell = supercell[filt]
    atom_id = atom_id[filt]
    cell_id = cell_id[filt]
    numbers = numbers[filt]

    g = create_knn_graph(supercell, k=k, line_graph=False)

    g.ndata['z'] = numbers
    g.ndata['atom_id'] = atom_id
    g.ndata['cell_id'] = cell_id

    g = create_periodic_graph(g)

    return g

class NoiseRegressionEval(nn.Module):
    def __init__(self, noise, k):
        super(NoiseRegressionEval, self).__init__()
        self.noise = noise
        self.k = k

    def forward(self, datapoint):
        n_atoms = torch.randint(1, 5, (1,)) * 1000
        return noise_regression_prep(datapoint, n_atoms, self.noise, self.k)
    
class SimulatedNoiseRegressionEval(nn.Module):
    def __init__(self, k):
        super(SimulatedNoiseRegressionEval, self).__init__()
        self.k = k

    def forward(self, datapoint):
        return noise_regression_sim_prep(datapoint, self.k)