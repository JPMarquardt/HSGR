import dgl
import numpy as np
from simtk.openmm import XmlSerializer


def create_dgl_graph(trajectory_file):
    # Load the trajectory from OpenMM XML file
    with open(trajectory_file, 'r') as f:
        serialized_system = f.read()
    system = XmlSerializer.deserialize(serialized_system)

    # Extract atom positions from the trajectory
    positions = []
    for state in system.context.getState(getPositions=True):
        positions.append(state.getPositions(asNumpy=True).value_in_unit(u.angstrom))

    # Create DGL graph
    num_frames = len(positions)
    num_atoms = positions[0].shape[0]
    graph = dgl.DGLGraph()
    graph.add_nodes(num_atoms * num_frames)

    # Add edges between consecutive frames
    for i in range(num_frames - 1):
        src_nodes = np.arange(i * num_atoms, (i + 1) * num_atoms)
        dst_nodes = np.arange((i + 1) * num_atoms, (i + 2) * num_atoms)
        graph.add_edges(src_nodes, dst_nodes)

    # Set node features as atom positions
    graph.ndata['pos'] = np.concatenate(positions)

    return graph
