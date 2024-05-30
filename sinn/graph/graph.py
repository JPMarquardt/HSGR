import torch
import dgl
import numpy as np
import nfflr

from typing import Tuple
from sinn.graph.utils import compute_dx, compute_bond_cosines, copy_d, compute_max_d, compute_nd

def create_supercell(data: torch.tensor, n: int):
    """
    Create a supercell
    """
    n_atoms = data.shape[0]
    supercell = torch.zeros((data.shape[0] * n**3, data.shape[1]))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                ind = (i*n**2 + j*n + k) * n_atoms
                displacement = torch.tensor([i, j, k], dtype=torch.float)
                supercell[ind:ind + n_atoms] = data + displacement[None, :]

    return supercell

def lattice_plane_slicer(data: torch.Tensor, miller_index: torch.Tensor, n: int):
    """
    Slice the unit cell on lattice plane
    """
    xeq0 = miller_index[0] != 0
    yeq0 = miller_index[1] != 0
    zeq0 = miller_index[2] != 0

    if xeq0.int() + yeq0.int() + zeq0.int() < 2:
        return data

    #prep miller index for normalization
    miller_index = miller_index.float()
    nz_miller_index = miller_index[miller_index != 0]

    #normalize miller index so that the smallest component is 1
    mi_max = torch.max(nz_miller_index)
    miller_index = miller_index / mi_max
    miller_index = miller_index.unsqueeze(0)

    belowplane = torch.sum(data * miller_index, dim=0) < n
    
    #propagate in the direction of the vector with 1 because it lines up
    max_index = torch.argmax(nz_miller_index)
    propagation_vector = torch.zeros(3)
    propagation_vector[max_index] = n

    belowplane = belowplane.unsqueeze(0)
    new_cell = torch.where(belowplane, data + propagation_vector, data)

    return new_cell

def create_knn_graph(data: torch.Tensor, k: int, line_graph: bool = False):
    """
    Create a k-nearest neighbor graph with necessary edge features
    """
    g = dgl.knn_graph(data, k)
    g.ndata['x'] = data
    g.apply_edges(compute_dx)
    g.edata['d'] = torch.norm(g.edata['dx'], dim=1)
    g.update_all(copy_d, compute_max_d)
    g.apply_edges(compute_nd)

    if line_graph:
        tfm = dgl.LineGraph()
        h = tfm(g)
        h.apply_edges(compute_bond_cosines)
        return (g, h)
    
    else:
        return g
    
if __name__ == "__main__":
    #verify the functions

    import matplotlib.pyplot as plt
    data = torch.rand((5, 3))
    n = 2
    k = 3
    xyz = torch.tensor([1, 2, 1], dtype=torch.float)
    xyz = xyz / torch.max(xyz)
    supercell = create_supercell(data, n)
    sliced_cell = lattice_plane_slicer(supercell, xyz, 2)
    knn_graph = create_knn_graph(supercell, k)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(supercell[:, 0], supercell[:, 1], supercell[:,2], alpha=0.25)
    ax.scatter(sliced_cell[:, 0], sliced_cell[:, 1], sliced_cell[:,2], alpha=0.25)
    ax.plot([n/xyz[0], 0, 0, n/xyz[0]], [0, n/xyz[1], 0, 0], [0, 0, n/xyz[2], 0], color='b')
    ax.plot([n/xyz[0], 0, 0, n/xyz[0]], [n, n+n/xyz[1], n, n], [0, 0, n/xyz[2], 0], color='r')

    plt.show()
    
    print(knn_graph)
