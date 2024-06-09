import torch

def box_filter(data, lattice, dx):
    d = data.size()[1]
    full_filter = torch.ones(data.size()[0], dtype=torch.bool)

    for i in range(d):
        lattice_vector = lattice[i]
        lattice_vector_norm = torch.norm(lattice_vector)
        dp = torch.sum(data * lattice_vector/lattice_vector_norm, dim=1)
        high_filt = dp < 2 * lattice_vector_norm + dx
        low_filt = dp > lattice_vector_norm - dx
        filt = torch.logical_and(high_filt, low_filt)
        full_filter = torch.logical_and(full_filter, filt)

    return full_filter