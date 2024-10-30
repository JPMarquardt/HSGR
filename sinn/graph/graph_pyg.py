import torch
from math import ceil
from typing import Union

class Graph():
    def __init__(self, dictionary: dict[str, torch.Tensor], nodes: int, edges: int, device: str = 'cpu'):
        self.dictionary = dictionary
        self.n_nodes = nodes
        self.k = edges
        self.device = device

        self.to(device)

    def to(self, device: str):
        self.device = device
        for key, value in self.dictionary.items():
            if isinstance(value, torch.Tensor):
                self.dictionary[key] = value.to(device)
        return self

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.dictionary[key]
    
    def __setitem__(self, key: str, value: torch.Tensor):
        self.dictionary[key] = value

    def pop(self, key: str) -> torch.Tensor:
        return self.dictionary.pop(key)
    
    def keys(self):
        return self.dictionary.keys()
    
    def values(self):
        return self.dictionary.values()

    def debug(self):        
        for key, value in self.dictionary.items():
            print(key, torch.any(torch.isnan(value)))

def box_filter(data: torch.Tensor, dx: float, lattice: torch.Tensor):
    """
    Filter the data based on the box
    """
    device = data.device
    # get the lattice position
    inverse_lattice = torch.inverse(lattice)
    lattice_position = data @ inverse_lattice

    # initialize the filter
    full_filter = torch.ones(data.shape[0], dtype=torch.bool, device=device)

    # filter the data based on the center box
    for i in range(data.shape[1]):
        high_filt = lattice_position[:, i] < 2 + dx
        low_filt = lattice_position[:, i] >= 1 - dx

        filt = torch.logical_and(high_filt, low_filt)
        full_filter = torch.logical_and(full_filter, filt)

    return full_filter

def create_labeled_supercell(data: torch.Tensor, n: int, lattice: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a supercell
    """
    device = data.device
    
    if lattice is None:
        lattice = torch.eye(3)
    
    # get the necessary constants
    n_atoms = data.shape[0]
    d = data.shape[1]
    
    # not used right now: for non cubic lattices
    lattice_vector_norm = torch.norm(lattice, dim=1)
    lattice_vector_norm_prop = lattice_vector_norm / lattice_vector_norm.mean()
    norm_props = torch.ceil(lattice_vector_norm_prop * n).int()

    # create the atom ids
    atom_id = torch.arange(n_atoms, dtype=torch.float, device=device)
    atom_id = atom_id.repeat(n**3).int()

    # create the cell ids
    cell_id_1 = torch.arange(n, dtype=torch.int, device=device).repeat_interleave(n_atoms * n**2)
    cell_id_2 = torch.arange(n, dtype=torch.int, device=device).repeat_interleave(n_atoms * n).repeat(n)
    cell_id_3 = torch.arange(n, dtype=torch.int, device=device).repeat_interleave(n_atoms).repeat(n**2)
    cell_id = torch.stack((cell_id_1, cell_id_2, cell_id_3), dim=1)

    # create the supercell
    supercell = data.repeat(n**3, 1) + cell_id.float() @ lattice

    return supercell, atom_id, cell_id

def aperiodic_knn_graph_from_supercell(data: torch.Tensor, k: int):
    """
    Create a k-nearest neighbor graph with necessary edge features
    """

    n_nodes = data.size()[0]

    # get the distances and make our knn graph
    dx = data[: , None] - data[None, :]
    r = torch.norm(dx, dim=2)
    knn = torch.topk(r, k=k, dim=1, largest=False)

    values = knn.values

    # necessary things to add to the graph
    maxes = torch.max(values, dim=1).values.unsqueeze(1)
    knn_normalized = values / torch.clamp(maxes, min=1e-6)

    indices = knn.indices[:, :-1]
    values = values[:, :-1]
    knn_normalized = knn_normalized[:, :-1]

    knn_dx = torch.gather(dx, 1, indices.unsqueeze(2).expand(-1, -1, dx.size(2)))

    # KNN graph: dx, r, knn, n_nodes
    g = {'dx': knn_dx, 'r': knn_normalized, 'knn': indices}
    
    return Graph(g, n_nodes, k-1)

def periodic_graph_from_labeled_supercell(g: Graph, center: int = 1):
    """
    Create a periodic graph from a supercell and add the necessary edge features
    """
    # get which atoms are in the center
    in_center = torch.all(torch.eq(g.pop('cell_id'), center), dim=1).bool()
    in_center_ids = torch.nonzero(in_center).squeeze()

    # some useful constants of the graph
    total_nodes = g.n_nodes
    g.n_nodes = torch.sum(in_center)
    reduced_nodes = g.n_nodes
    
    # filter the graph to only include the in-center atoms
    g['dst_z'] = g['z'][in_center_ids]
    g['knn'] = g['knn'][in_center_ids]
    g['dx'] = g['dx'][in_center_ids]
    g['r'] = g['r'][in_center_ids]

    # this is how to add edge source properties to the graph
    edge_feat = torch.zeros((reduced_nodes, total_nodes), dtype=torch.int)
    
    # expand the edge features to the full graph
    edge_feat.scatter_(1, g['knn'], True)

    # update edge features to be relative to the in-center atoms
    g['src_z'] = edge_feat * g.pop('z')[None, :]
    g['src_z'] = g['src_z'].gather(1, g['knn'])

    # update knn indices to be relative to the in-center atoms
    knn = edge_feat * g.pop('atom_id')[None, :]
    g['knn'] = knn.gather(1, g['knn'])
    
    return g

def create_periodic_knn_graph(a: dict[str, torch.Tensor], k: int = 9):
    """
    Create a periodic k-nearest neighbor graph
    """
    device = a['positions'].device

    # get the stuff
    data = a['positions']
    lattice = a['cell']
    atomic_numbers = a['numbers']

    # math to get the number of replicates and if we need to filter cuz the box has a lot of atoms
    n_atoms = data.shape[0]
    dx = ((k + 1) / n_atoms) ** (1/3) / 2
    center = ceil(dx)
    replicates = center * 2 + 1
    
    # create the supercell
    supercell, atom_id, cell_id = create_labeled_supercell(data, n=replicates, lattice=lattice)
    atomic_numbers = atomic_numbers.repeat(replicates**3)

    # filter the supercell if we need to
    if dx < 0.5:
        filt = box_filter(supercell, 2*dx, lattice)
        supercell = supercell[filt]
        atomic_numbers = atomic_numbers[filt]
        atom_id = atom_id[filt]
        cell_id = cell_id[filt]

    # create the aperiodic knn graph
    g = aperiodic_knn_graph_from_supercell(supercell, k=k)

    # KNN graph: dx, r, knn, n_nodes
    # Additional: z, atom_id, cell_id

    # dx: src (per_node) -> dst (ind) (3)
    # r: src (per_node) -> dst (ind)
    # knn: src (per_node) -> dst (ind)
    # n_nodes: int

    # add the features needed for the periodic graph and z
    g['z'] = atomic_numbers
    g['atom_id'] = atom_id
    g['cell_id'] = cell_id

    # create the periodic graph
    g = periodic_graph_from_labeled_supercell(g, center=center)

    return g

def create_aperiodic_knn_graph(a: dict[str, torch.Tensor], k: int = 9):
    """
    Create a periodic k-nearest neighbor graph
    """
    device = str(a['positions'].device)

    # get the stuff
    data = a['positions']
    atomic_numbers = a['numbers']

    # create the aperiodic knn graph
    g = aperiodic_knn_graph_from_supercell(data, k=k)
    g.to(device)

    # KNN graph: dx, r, knn, n_nodes
    # Additional: z

    # dx: src (per_node) -> dst (ind) (3)
    # r: src (per_node) -> dst (ind)
    # knn: src (per_node) -> dst (ind)
    # n_nodes: int

    # add the features needed for the periodic graph and z
    g['dst_z'] = atomic_numbers

    # this is how to add edge source properties to the graph
    edge_feat = torch.zeros((g.n_nodes, g.n_nodes), dtype=torch.int, device=device)
    
    # expand the edge features to the full graph
    edge_feat.scatter_(1, g['knn'], True)

    # update edge features to be relative to the in-center atoms
    g['src_z'] = edge_feat * g['dst_z'][None, :]
    g['src_z'] = g['src_z'].gather(1, g['knn'])

    return g

def create_linegraph(g: Graph):
    """
    Create a line graph from a graph
    """
    h = {}
    device = g.device

    # put in the cosines
    cosine_denomenator = torch.sqrt(torch.sum(g['dx'][:, :, None] ** 2, dim=-1))
    cosine_denomenator = cosine_denomenator * cosine_denomenator.transpose(1, 2)

    cosine_numerator = torch.sum(g['dx'][:, :, None] * g['dx'][:, None, :], dim= -1)

    h['r'] = cosine_numerator / cosine_denomenator
    h['r'][h['r'].isnan()] = 0

    # put in the knn features
    h['knn'] = g['knn']

    # right now this is not generalizable to more than 2 dimensions, but it is possible using cat instead
    h['dst_z'] = g['dst_z']

    # put in the z features
    h['src_z'] = g['src_z'][:, :, None].expand(-1, -1, g.k)

    # if we want to go to higher dim we need to create dx for h
    
    return Graph(h, g.n_nodes, g.k, device=device)