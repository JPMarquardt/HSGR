import torch

def compute_dx(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # 
    r1 = -edges.src["x"]
    r2 = edges.dst["x"]

    return {"dx": r1 + r2}

def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["dx"]
    r2 = edges.dst["dx"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))

    return {"angle": bond_cosine}

def copy_d(edges):
    """Copy the bond distance from the original graph to the line graph."""
    return {"d": edges.data["d"]}

def compute_max_d(nodes):
    """Compute the maximum distance between a node and its neighbors."""
    return {"max_d": torch.max(nodes.mailbox["d"], dim=1)[0]}

def compute_nd(edges):
    """Compute the normalized distance between two atoms."""
    return {"d": edges.data["d"] / edges.dst["max_d"]}