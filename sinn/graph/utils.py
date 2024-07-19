import torch
import dgl.function as fn

def compute_bond_cosines(h):
    h.ndata['dr_norm'] = h.ndata['dr'].norm(dim=1)

    h.apply_edges(fn.u_dot_v('dr', 'dr', 'dr_dot_dr'))
    h.apply_edges(fn.u_mul_v('dr_norm', 'dr_norm', 'dr_norm_sq'))

    r = h.edata.pop('dr_dot_dr').squeeze() / h.edata.pop('dr_norm_sq')
    r = torch.nan_to_num(r, nan=0.0)
    h.edata['r'] = r
    

def check_in_center(nodes):
    """Get the projection of the edge onto the plane."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # get the projection of the edge onto the plane
    # the projection is the displacement vector of the edge
    # projected onto the plane defined by the normal vector
    # of the edge
    return nodes.data.pop("in_center")

def compute_bond_cross_product(h):
    h.ndata['dr-1'] = h.ndata['dr'].roll(1, dims=1)
    h.ndata['dr+1'] = h.ndata['dr'].roll(-1, dims=1)

    h.apply_edges(fn.u_mul_v('dr-1', 'dr', 'cross-'))
    h.apply_edges(fn.u_mul_v('dr', 'dr+1', 'cross+'))

    cross = h.edata.pop('cross+') - h.edata.pop('cross-')
    return cross
