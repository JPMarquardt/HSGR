import torch

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

    return {"d": bond_cosine}

def check_in_center(nodes):
    """Get the projection of the edge onto the plane."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # get the projection of the edge onto the plane
    # the projection is the displacement vector of the edge
    # projected onto the plane defined by the normal vector
    # of the edge
    return nodes.data["in_center"]