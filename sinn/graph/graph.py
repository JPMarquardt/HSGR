import torch
import dgl
import numpy as np
import nfflr

from typing import Tuple
from sinn.graph.utils import compute_bond_cosines, check_in_center

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

def create_labeled_supercell(data: torch.tensor, n: int, lattice: torch.tensor = None):
    """
    Create a supercell
    """
    if lattice is None:
        lattice = torch.eye(3)
    
    n_atoms = data.shape[0]
    d = data.shape[1]
    supercell = torch.zeros((n_atoms * n**3, d))
    atom_id = torch.linspace(0, n_atoms-1, n_atoms, dtype=torch.int)
    atom_id = atom_id.repeat(n**3)
    cell_id = torch.zeros((n_atoms * n **3, d), dtype=torch.int)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                ind = (i*n**2 + j*n + k) * n_atoms
                displacement = torch.tensor([i, j, k], dtype=torch.float)
                supercell[ind:ind + n_atoms] = data + displacement[None, :] @ lattice
                cell_id[ind:ind + n_atoms] = displacement[None, :]

    return supercell, atom_id, cell_id

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
    
def create_linegraph(g: dgl.DGLGraph):
    """
    Create a line graph from a graph
    """
    h = dgl.transforms.line_graph(g)
    h.ndata['dr'] = g.edata['dr']
    h.apply_edges(compute_bond_cosines)
    return h

def create_knn_graph(data: torch.Tensor, k: int, line_graph: bool = False):
    """
    Create a k-nearest neighbor graph with necessary edge features
    """
    g = dgl.knn_graph(data, k)
    g.ndata['r'] = data

    compute_dx = dgl.function.v_sub_u('r', 'r', 'dr')
    copy_d = dgl.function.copy_e('r', 'm')
    compute_max_d = dgl.function.max('m', 'max_r')
    compute_nd = dgl.function.e_div_v('r', 'max_r', 'nr')

    g.apply_edges(compute_dx)
    g.edata['r'] = torch.norm(g.edata['dr'], dim=1)

    g.update_all(copy_d, compute_max_d)
    g.apply_edges(compute_nd)
    g.edata['r'] = g.edata.pop('nr')

    if line_graph:
        h = create_linegraph(g)
    
    g.ndata.pop('r')
    g.ndata.pop('max_r')

    if line_graph: return (g, h)
    return g

def create_periodic_graph(g):
    """
    Create a periodic graph from a graph
    """
    cell_id = g.ndata.pop('cell_id')
    g.ndata['atom_id'] = g.ndata['atom_id'].float()

    in_center = torch.all(torch.eq(cell_id, 1), dim=1)

    g.ndata['in_center'] = in_center.float()
    gr = g.reverse(copy_edata=True)

    copy_in_center_src = dgl.function.copy_u('in_center', 'in_center_src')
    copy_in_center_dst = dgl.function.copy_u('in_center', 'in_center_dst')

    g.apply_edges(copy_in_center_src)
    gr.apply_edges(copy_in_center_dst)

    in_center_src = g.edata.pop('in_center_src')
    in_center_dst = gr.edata.pop('in_center_dst')

    center_ids = g.filter_nodes(check_in_center)

    g.ndata.pop('in_center')

    out_in_filt = (~in_center_src.bool()) & in_center_dst.bool()

    copy_atom_id_src = dgl.function.copy_u('atom_id', 'atom_id_src')
    copy_atom_id_dst = dgl.function.copy_u('atom_id', 'atom_id_dst')

    g.apply_edges(copy_atom_id_src)
    gr.apply_edges(copy_atom_id_dst)

    g.ndata.pop('atom_id')

    src_out_in = g.edata.pop('atom_id_src')[out_in_filt].to(torch.int64)
    dst_out_in = gr.edata.pop('atom_id_dst')[out_in_filt].to(torch.int64)

    g = g.subgraph(center_ids)

    g.add_edges(src_out_in, dst_out_in)

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
