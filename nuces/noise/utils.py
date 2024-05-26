import torch

def dist_matrix(tensor):
    x1 = torch.unsqueeze(tensor, 0)
    x2 = torch.unsqueeze(tensor, 1)
    dists = torch.norm(x1 - x2, dim=-1)
    return dists

def find_min_dist(tensor):
    dists = dist_matrix(tensor)
    min_dist = torch.min(dists[dists != 0])
    return min_dist