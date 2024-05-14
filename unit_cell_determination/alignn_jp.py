from plum import dispatch

import logging
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Literal, Callable

import dgl
import torch
from torch import nn

from dgl.nn import AvgPooling, SumPooling

from nfflr.models.utils import (
    autograd_forces,
    RBFExpansion,
    MLPLayer,
    ALIGNNConv,
    EdgeGatedGraphConv,
    JP_Featurization,
)
from nfflr.data.graph import compute_bond_cosines
from nfflr.nn.transform import PeriodicRadiusGraph
from nfflr.nn.cutoff import XPLOR
from nfflr.data.atoms import _get_attribute_lookup, Atoms

class ALIGNN_JP(nn.Module):
    """Atomistic Line graph network.
    
    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig()):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.config = config
        self.transform = config.transform
        logging.debug(f"{config=}")

        self.atom_features = config.atom_features
        if config.atom_features == "embedding":
            self.atom_embedding = nn.Embedding(108, config.hidden_features)
        elif isinstance(config.atom_features, dict):
            self.atom_embedding = nn.Sequential(
                JP_Featurization(**config.atom_features, output_features = config.hidden_features),
                MLPLayer(config.hidden_features, config.hidden_features, norm=config.norm)
            )
            
        else:
            f = _get_attribute_lookup(atom_features=config.atom_features)
            self.atom_embedding = nn.Sequential(
                f, MLPLayer(f.embedding_dim, config.hidden_features, norm=config.norm)
            )

        self.reference_energy = None
        if config.reference_energies is not None:
            self.reference_energy = nn.Embedding(
                108, embedding_dim=1, _weight=config.reference_energies.view(-1, 1)
            )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(
                config.edge_input_features, config.embedding_features, norm=config.norm
            ),
            MLPLayer(
                config.embedding_features, config.hidden_features, norm=config.norm
            ),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(
                config.triplet_input_features,
                config.embedding_features,
                norm=config.norm,
            ),
            MLPLayer(
                config.embedding_features, config.hidden_features, norm=config.norm
            ),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features, config.hidden_features, norm=config.norm
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features, norm=config.norm
                )
                for idx in range(config.gcn_layers)
            ]
        )

        if config.energy_units == "eV/atom":
            self.readout = AvgPooling()
        else:
            self.readout = SumPooling()

        self.fc = nn.Linear(config.hidden_features, config.output_features)

        if config.classification:
            self.sm = nn.Softmax(dim=-1)

    @dispatch
    def forward(self, x):
        print("convert")
        return self.forward(Atoms(x))

    @dispatch
    def forward(self, x: Atoms):
        device = next(self.parameters()).device
        return self.forward(self.transform(x).to(device))

    @dispatch
    def forward(self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        config = self.config

        if isinstance(g, dgl.DGLGraph):
            lg = None
        else:
            g, lg = g

        g = g.local_var()

        # to compute forces, take gradient wrt g.edata["r"]
        # need to add bond vectors to autograd graph
        if config.compute_forces:
            g.edata["r"].requires_grad_(True)

        # initial bond features
        bondlength = torch.norm(g.edata["nr"], dim=1)
        y = self.edge_embedding(bondlength)

        if config.cutoff is not None:
            # save cutoff function value for application in EdgeGatedGraphconv
            g.edata["cutoff_value"] = self.config.cutoff(bondlength)

        # initial triplet features
        if len(self.alignn_layers) > 0:
            if lg is None:
                lg = g.line_graph(shared=True)
                lg.apply_edges(compute_bond_cosines)

            z = self.angle_embedding(lg.edata["h"])

        # initial node features: atom feature network...
        if not isinstance(self.atom_features, dict):
            atomic_number = g.ndata.pop("atomic_number").int()
            x = self.atom_embedding(atomic_number)
        else:
            x = self.atom_embedding((g, lg))

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        output = torch.squeeze(self.fc(h))

        #classification
        if self.config.classification:
            output = self.sm(output)

        if self.config.debug:
            print(output)

        if config.compute_forces:
            forces, stress = autograd_forces(
                output,
                g.edata["r"],
                g,
                energy_units=config.energy_units,
                compute_stress=True,
            )

            return dict(total_energy=output, forces=forces, stress=stress)
        return output


"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from plum import dispatch

import logging
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Literal, Callable

import dgl
import torch
from torch import nn

from dgl.nn import AvgPooling, SumPooling

from nfflr.models.utils import (
    autograd_forces,
    RBFExpansion,
    MLPLayer,
    ALIGNNConv,
    EdgeGatedGraphConv,
    JP_Featurization,
)
from nfflr.data.graph import compute_bond_cosines
from nfflr.nn.transform import PeriodicRadiusGraph
from nfflr.nn.cutoff import XPLOR
from nfflr.data.atoms import _get_attribute_lookup, Atoms


@dataclass
class ALIGNNConfig:
    """Hyperparameter schema for nfflr.models.gnn.alignn."""

    transform: Callable = PeriodicRadiusGraph(cutoff=8.0)
    # cutoff: Optional[tuple[float]] = (7.5, 8.0)
    cutoff: torch.nn.Module = XPLOR(7.5, 8.0)
    alignn_layers: int = 4
    gcn_layers: int = 4
    norm: Literal["batchnorm", "layernorm"] = "batchnorm"
    atom_features: str = "cgcnn"
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    output_features: int = 1
    classification: bool = False
    compute_forces: bool = False
    energy_units: Literal["eV", "eV/atom"] = "eV/atom"
    reference_energies: Optional[torch.Tensor] = None
    debug: bool = False


class ALIGNN(nn.Module):
    """Atomistic Line graph network.
    
    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig()):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.config = config
        self.transform = config.transform
        logging.debug(f"{config=}")

        self.atom_features = config.atom_features
        if config.atom_features == "embedding":
            self.atom_embedding = nn.Embedding(108, config.hidden_features)
        elif isinstance(config.atom_features, dict):
            self.atom_embedding = nn.Sequential(
                JP_Featurization(**config.atom_features, output_features = config.hidden_features),
                MLPLayer(config.hidden_features, config.hidden_features, norm=config.norm)
            )
            
        else:
            f = _get_attribute_lookup(atom_features=config.atom_features)
            self.atom_embedding = nn.Sequential(
                f, MLPLayer(f.embedding_dim, config.hidden_features, norm=config.norm)
            )

        self.reference_energy = None
        if config.reference_energies is not None:
            self.reference_energy = nn.Embedding(
                108, embedding_dim=1, _weight=config.reference_energies.view(-1, 1)
            )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(
                config.edge_input_features, config.embedding_features, norm=config.norm
            ),
            MLPLayer(
                config.embedding_features, config.hidden_features, norm=config.norm
            ),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(
                config.triplet_input_features,
                config.embedding_features,
                norm=config.norm,
            ),
            MLPLayer(
                config.embedding_features, config.hidden_features, norm=config.norm
            ),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features, config.hidden_features, norm=config.norm
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features, norm=config.norm
                )
                for idx in range(config.gcn_layers)
            ]
        )

        if config.energy_units == "eV/atom":
            self.readout = AvgPooling()
        else:
            self.readout = SumPooling()

        self.fc = nn.Linear(config.hidden_features, config.output_features)

        if config.classification:
            self.sm = nn.Softmax(dim=-1)

    @dispatch
    def forward(self, x):
        print("convert")
        return self.forward(Atoms(x))

    @dispatch
    def forward(self, x: Atoms):
        device = next(self.parameters()).device
        return self.forward(self.transform(x).to(device))

    @dispatch
    def forward(self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        config = self.config

        if isinstance(g, dgl.DGLGraph):
            lg = None
        else:
            g, lg = g

        g = g.local_var()

        # to compute forces, take gradient wrt g.edata["r"]
        # need to add bond vectors to autograd graph
        if config.compute_forces:
            g.edata["r"].requires_grad_(True)

        # initial bond features
        bondlength = torch.norm(g.edata["nr"], dim=1)
        y = self.edge_embedding(bondlength)

        if config.cutoff is not None:
            # save cutoff function value for application in EdgeGatedGraphconv
            g.edata["cutoff_value"] = self.config.cutoff(bondlength)

        # initial triplet features
        if len(self.alignn_layers) > 0:
            if lg is None:
                lg = g.line_graph(shared=True)
                lg.apply_edges(compute_bond_cosines)

            z = self.angle_embedding(lg.edata["h"])

        # initial node features: atom feature network...
        if not isinstance(self.atom_features, dict):
            atomic_number = g.ndata.pop("atomic_number").int()
            x = self.atom_embedding(atomic_number)
        else:
            x = self.atom_embedding((g, lg))

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        output = torch.squeeze(self.fc(h))

        #classification
        if self.config.classification:
            output = self.sm(output)

        if self.config.debug:
            print(output)

        if config.compute_forces:
            forces, stress = autograd_forces(
                output,
                g.edata["r"],
                g,
                energy_units=config.energy_units,
                compute_stress=True,
            )

            return dict(total_energy=output, forces=forces, stress=stress)
        return output

class JP_Featurization(nn.Module):
    """
    bond symmetry metric
    \sum_{l=1}^N g_l * \exp[-a_l*(R_ij-Rjk)^2] * [\cos(\theta_{ijk} * b_l + c_l)]^d_l * f_i \cdot f_k
    )
    where N is the number of space groups to discriminate between
    """

    def __init__(
        self,
        n_heads: int,
        n_atom_types: Optional[int] = None,
        hidden_features: int = 128,
        output_features: int = 256,
        eps: float = 1e-3
    ):
        """Initialize parameters for Atom Featurization based on local symmetry.
        
        Parameters
        ----------
        n_heads : int
            Number of heads for the attention like mechanism.
            
            n_atom_types : int"""
        super().__init__()

        self.hidden_features = hidden_features
        self.output_features = output_features
        self.n_heads = n_heads
        self.eps = eps
        self.n_atom_types = n_atom_types

        #check if the atoms are masked or not
        if n_atom_types is None:
        
            #four is the number of symmetries of 3 atoms with 2 degenerate atoms
            n_atom_types = 4
            self.compute_symmetry_key = compute_symmetry_key_masked
        else:
            key_embedding = nn.Embedding(n_atom_types, hidden_features * n_heads)
            self.compute_symmetry_key = compute_symmetry_key

        pi = torch.tensor((np.pi), requires_grad =False)
        self.register_buffer('pi', pi)

        self.a = nn.Parameter(torch.rand((n_heads),requires_grad= True),requires_grad= True)
        self.b = nn.Parameter(torch.rand((n_heads),requires_grad= True),requires_grad= True)
        self.c = nn.Parameter(torch.rand((n_heads),requires_grad= True),requires_grad= True)
        self.d = nn.Parameter(torch.rand((n_heads),requires_grad= True),requires_grad= True)

        self.value_function = nn.Embedding(n_atom_types, output_features * n_heads)

    def forward(
        self,
        g: dgl.DGLGraph,
    ) -> torch.Tensor:
        """
        Generate featurization based on graph and line graph.
        """
        if isinstance(g, tuple):
            g, lg = g

        g = g.local_var()
        pi = self.pi

        # send the atomic numbers as the keys
        g.apply_edges(send_k_src)

        # if the atoms aren't masked use their embedding
        if self.n_atom_types is not None:
            n_edges = lg.edata["k_src"].size()[0]
            lg.ndata["k_src"] = self.key_embedding(g.edata["k_src"]).reshape((n_edges, self.hidden_features, self.n_heads))
            lg.ndata["k_dst"] = self.key_embedding(g.edata["k_dst"]).reshape((n_edges, self.hidden_features, self.n_heads))
            lg.ndata["v_src"] = self.value_funtion(g.edata["v_src"]).reshape((n_edges, self.output_features, self.n_heads))

        # if they are masked use the atomic numbers
        else:
            lg.ndata["k_dst"] = g.edata["k_dst"]
            lg.ndata["k_src"] = g.edata["k_src"]


        # compute atomic symmetry value based on the atomic symmetry function
        lg.apply_edges(self.compute_symmetry_key)
        atomic_symmetry_key = lg.edata.pop("symmetry_key")

        # if the atoms are masked the value has to be computed using the symmetry key instead of being taken from the central atom atomic number
        if self.n_atom_types is None:
            atomic_symmetry_value = self.value_function(atomic_symmetry_key)
            atomic_symmetry_value = atomic_symmetry_value.reshape((atomic_symmetry_key.size()[0], self.output_features, self.n_heads))

        # scale computed cosine to make it safer (gradient of arccos at 1/-1 is inf)
        costheta = lg.edata.pop("h")
        costheta = torch.clamp(costheta, -self.eps, self.eps)

        # compute all graph properties needed for the symmetry functions
        theta = torch.arccos(costheta)
        theta = theta.unsqueeze(1)
        dnr = lg.edata.pop("dnr").unsqueeze(1)

        # ensure b is not too big
        A = self.a
        B = self.b % pi
        C = self.c
        D = self.d

        # compute the symmetry functions
        angular_sym = ((torch.cos(A[None, :] * theta + B[None, :]) + 1)/2) ** C[None, :]
        radial_sym = torch.exp(-D[None, :] * (dnr ** 2))

        # initialize both reverse graphs
        rlg = dgl.reverse(lg)
        rg = dgl.reverse(g)

        # compute total symmetry function
        spatial_sym = radial_sym * angular_sym
        symmetry_value = atomic_symmetry_value * spatial_sym[:, None, :]

        # send product of atom and spatial
        rlg.edata["w_ij"] = symmetry_value

        # update the reverse graphs. mean is used instead of sum for stability with sm
        rlg.update_all(
            fn.copy_e("w_ij","w"),
            fn.mean("w", "w_ij")
        )
        rg.edata["w_ij"] = rlg.ndata["w_ij"]
        rg.update_all(
            fn.copy_e("w_ij","w"),
            fn.mean("w", "w_ij")
        )

        #sum the weights over each of the heads leaving us with an output of size (n_atoms, output_features)
        v_ij = rg.ndata.pop("w_ij")
        out = torch.sum(v_ij, dim=-1)

        return out
    

def send_k_src(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # this sends neighbor information to each edge
    k_src = edges.src["atomic_number"]
    k_dst = edges.dst["atomic_number"]

    return {"k_src": k_src, "k_dst": k_dst}

def compute_symmetry_key_masked(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # this checks if the atomic number of b is in (a, c) and if c = a
    # then it assigns a a symmetry value from 0 to 3
    ka = edges.src["k_src"]
    kc = edges.dst["k_dst"]
    kb = edges.src["k_dst"]

    peripheral_symmetry = ka == kc
    central_symmetry = (kb == ka) * (kb == kc)

    symmetry = peripheral_symmetry * 2 + central_symmetry #there must be a more beautiful way to do this

    return {"symmetry_key": symmetry}

def compute_symmetry_key(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # this checks if the atomic number of b is in (a, c) and if c = a
    # then it assigns a a symmetry value from 0 to 3
    ka = edges.src["k_src"]
    kc = edges.dst["k_dst"]
    vb = edges.src["v_dst"]

    weight = torch.sum(ka * kc, dim=1)
    symmetry = vb * weight[:, None, :]

    return {"symmetry_key": symmetry}

def compute_delta_radius(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # 
    r1 = -edges.src["nr"]
    r2 = edges.dst["nr"]
    r1 = torch.norm(r1, dim=1)
    r2 = torch.norm(r2, dim=1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))

    delta_radius = r2-r1

    return {"dnr": delta_radius}

def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))

    return {"h": bond_cosine}
