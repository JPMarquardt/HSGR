import torch
import torch.nn as nn
from math import ceil

from sinn.graph.graph import create_supercell, create_labeled_supercell, create_knn_graph, lattice_plane_slicer, create_periodic_graph, create_periodic_graph
from sinn.train.utils import gaussian_noise
from nfflr.data.dataset import Atoms

from typing import Callable, Union, Dict

def noise_regression_prep(a: Atoms, k: int, n_target_atoms: int, noise: float):
    coords = a.positions
    lattice = a.cell
    numbers = a.numbers

    data = torch.matmul(coords, torch.inverse(lattice))

    replicates = (n_target_atoms / data.size()[0]) ** (1/3)
    replicates = ceil(replicates)

    supercell = create_supercell(data, replicates)
    supercell = gaussian_noise(supercell, noise)
    for _ in range(torch.randint(0, 3, (1,)).item()):
        miller_index = torch.randint(0, 4, (3,))
        supercell = lattice_plane_slicer(supercell, miller_index, replicates)
    supercell = supercell @ lattice
    
    g = create_knn_graph(supercell, k=k)
    numbers = numbers.repeat(replicates**3)
    g.ndata['z'] = numbers

    return g

def aperiodic_classification_atoms(a: Dict, k: int = 9):

    data = a['positions']
    numbers = a['numbers']

    n = data.size()[0]
    #reduce the size of the dataset
    if n > 7000:
        target = 7000
        reduction_factor = (target / n) ** (1/3)
        for i in range(3):
            dim_width = torch.max(data[:,i]) - torch.min(data[:,i])
            filt = (data[:,i] < reduction_factor * dim_width) & (data[:,i] > 0)
            data = data[filt]
            numbers = numbers[filt]


    g = create_knn_graph(data, k=k)

    g.ndata['z'] = numbers

    return g

def aperiodic_classification_sim(a: Dict[str, torch.Tensor], k: int = 9):

    data = a['positions']
    numbers = a['numbers']

    n = data.size()[0]
    #reduce the size of the dataset
    if n > 7000:
        target = 7000
        reduction_factor = (target / n) ** (1/3)
        for i in range(3):
            dim_width = torch.max(data[:,i]) - torch.min(data[:,i])
            filt = (data[:,i] < reduction_factor * dim_width) & (data[:,i] > 0)
            data = data[filt]
            numbers = numbers[filt]


    g = create_knn_graph(data, k=k)

    g.ndata['z'] = numbers

    return g

def noise_regression_sim_prep(a: Atoms, k: int = 9):
    data = a.positions
    lattice = a.cell
    numbers = a.numbers
    replicates = 3

    dx = 0.1 * torch.min(torch.norm(lattice, dim=1))
    supercell, atom_id, cell_id = create_labeled_supercell(data, n=replicates, lattice=lattice)
    numbers = numbers.repeat(replicates**3)
    filt = big_box_filter(supercell, lattice, dx)

    supercell = supercell[filt]
    atom_id = atom_id[filt]
    cell_id = cell_id[filt]
    numbers = numbers[filt]

    g = create_knn_graph(supercell, k=k)

    g.ndata['z'] = numbers
    g.ndata['atom_id'] = atom_id
    g.ndata['cell_id'] = cell_id

    g = create_periodic_graph(g)

    return g



class NoiseRegressionTrain(nn.Module):
    """
    finite repeating crystal structure for training noise regression
    number of atoms in the crystal is randomly chosen between 1k and 5k
    """
    def __init__(self, k: int = 17, noise: Callable = None, crystal_size: Callable = None):
        super(NoiseRegressionTrain, self).__init__()
        self.k = k

        if noise is None:
            noise = lambda: torch.rand(1)
        if crystal_size is None:
            crystal_size = lambda: 1000 * torch.randint(1, 3, (1,))

        self.noise = noise
        self.crystal_size = crystal_size

    def forward(self, datapoint):
        n_atoms = self.crystal_size()
        noise = self.noise()
        return noise_regression_prep(datapoint, self.k, n_atoms, noise), noise
    
class PeriodicClassificationLarge(nn.Module):
    """
    infinite repeating simulation box for evaluation of noise regression
    """
    def __init__(self, k):
        super(PeriodicClassificationLarge, self).__init__()
        self.k = k

    def forward(self, datapoint):
        return noise_regression_sim_prep(datapoint, self.k)
    
class APeriodicClassificationAtoms(nn.Module):
    """
    infinite repeating simulation box for evaluation of noise regression
    """
    def __init__(self, k):
        super(APeriodicClassificationAtoms, self).__init__()
        self.k = k

    def forward(self, datapoint):
        return aperiodic_classification_atoms(datapoint, self.k)
    
class APeriodicClassification(nn.Module):
    """
    infinite repeating simulation box for evaluation of noise regression
    """
    def __init__(self, k):
        super(APeriodicClassification, self).__init__()
        self.k = k

    def forward(self, datapoint):
        return aperiodic_classification_sim(datapoint, self.k)
    
class PeriodicClassificationSmall(nn.Module):
    """
    infinite repeating simulation box for training periodic classification (no noise)
    """
    def __init__(self, k: int = 17):
        super(PeriodicClassificationSmall, self).__init__()
        self.k = k

    def forward(self, datapoint):
        return periodic_classification_prep(datapoint, self.k)
    

if __name__ == '__main__':
    from sinn.dataset.dataset import FilteredAtomsDataset
    data = FilteredAtomsDataset('dft_3d',
                                target='spg_number').dataset
    for i in range(10):
        print(periodic_classification(data[i][0], 9))