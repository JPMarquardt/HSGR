import argparse
import numpy as np
import numpy.typing as npt
import pickle as pkl
import openmm
import yaml
import pathlib

from openmm import app
from openmm.app.xtcreporter import XTCReporter
from openmm import unit
from openmmtorch import TorchForce
from colloids import ColloidPotentialsAlgebraic, ColloidPotentialsParameters, ColloidPotentialsTabulated, ShiftedLennardJonesWalls
from colloids.gsd_reporter import GSDReporter
from colloids.helper_functions import read_xyz_file, write_gsd_file, write_xyz_file
from colloids.run_parameters import RunParameters
from colloids.status_reporter import StatusReporter

import torch
from torchmdnet.models.model import load_model

class ExampleAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # TODO ADD OPTION FOR PLATFORM PROPERTIES?
        # TODO PUT EQUILIBRATION STEPS?
        default_parameters = RunParameters()
        default_parameters.to_yaml("example.yaml")
        parser.exit()

class CVModule(torch.nn.Module):
    def __init__(self, mlp_name, ptypes):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.mlp = torch.load(mlp_name, map_location=self.device)
        self.ptypes = torch.tensor(ptypes, dtype=torch.long, device=self.device)

        self.mlp.eval()
        print(self.ptypes.device)

    def forward(self, positions):
        datapoint = {'positions': positions, 'numbers': self.ptypes}
        return torch.mean(self.mlp(datapoint))


class CVReporter(object):
    def __init__(self, file, reportInterval, metad_obj):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self._meta = metad_obj

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        cv0_val, cv1_val = self._meta.getCollectiveVariables(simulation)
        self._out.write('%d %.6f %.6f\n' % (simulation.currentStep,
                                            cv0_val, cv1_val))
        self._out.flush()
    
def set_up_simulation(parameters: RunParameters, types: npt.NDArray[str],
                      cell: npt.NDArray[float], bias: dict[str, tuple[float]]) -> app.Simulation:
    topology = app.topology.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("res1", chain)

    for t in types:
        topology.addAtom(t, None, residue)

    topology.setPeriodicBoxVectors(cell)

    system = openmm.System()

    include_walls = any(parameters.wall_directions)
    all_walls = all(parameters.wall_directions)
    if include_walls:
        box_vector_one = cell[0]
        box_vector_two = cell[1]
        box_vector_three = cell[2]
        if not (box_vector_one[1] == 0.0 and box_vector_one[2] == 0.0 and
                box_vector_two[0] == 0.0 and box_vector_two[2] == 0.0 and
                box_vector_three[0] == 0.0 and box_vector_three[1] == 0.0):
            raise ValueError("If any wall is included, the box vectors must be parallel to the coordinate axes.")
        wall_distances = (box_vector_one[0] * (unit.nano * unit.meter) if parameters.wall_directions[0] else None,
                          box_vector_two[1] * (unit.nano * unit.meter)if parameters.wall_directions[1] else None,
                          box_vector_three[2] * (unit.nano * unit.meter) if parameters.wall_directions[2] else None)
        final_cell = cell.copy()
        if not all_walls:
            for index, wall_direction in enumerate(parameters.wall_directions):
                if wall_direction:
                    # The shifted Lennard Jones walls diverge at distance r = radius - 1 from the location of the wall,
                    # where radius is the radius of the particle. The minimum distance between periodic images through
                    # a wall is thus 2 * radius_min - 2, where radius_min is the smallest radius in the system.
                    # The maximum cutoff of the electrostatic interactions is
                    # 2 * radius_max + cutoff_factor * debye_length. In order to prevent particles from interacting
                    # through the walls, we thus increase the length of the periodic box vectors (not the wall) by
                    # 2 * (radius_max - radius_min) + 2 + cutoff_factor * debye_length.
                    final_cell[index][index] += \
                        (2.0 * (max(parameters.radii.values()) - min(parameters.radii.values()))
                         + 2.0 * (unit.nano * unit.meter)
                         + parameters.cutoff_factor * parameters.debye_length).value_in_unit(unit.nano * unit.meter)
    else:
        wall_distances = None
        final_cell = cell

    if not all_walls:
        topology.setPeriodicBoxVectors(final_cell)
        system.setDefaultPeriodicBoxVectors(openmm.Vec3(*final_cell[0]), openmm.Vec3(*final_cell[1]),
                                            openmm.Vec3(*final_cell[2]))

    # Prevent printing the traceback when the platform is not existing.
    platform = openmm.Platform.getPlatformByName(parameters.platform_name)

    # TODO: ALLOW FOR DIFFERENT INTEGRATORS?
    integrator = openmm.LangevinMiddleIntegrator(parameters.temperature,
                                                 parameters.collision_rate,
                                                 parameters.timestep)
    if parameters.integrator_seed is not None:
        integrator.setRandomNumberSeed(parameters.integrator_seed)

    potentials_parameters = ColloidPotentialsParameters(
        brush_density=parameters.brush_density, brush_length=parameters.brush_length,
        debye_length=parameters.debye_length, temperature=parameters.temperature,
        dielectric_constant=parameters.dielectric_constant
    )

    if parameters.use_tabulated:
        # TODO: Maybe generalize tabulated potentials to more than two types.
        # Use a dictionary instead of a set to preserve the order of the types.
        # See https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
        # Works since Python 3.7.
        set_of_types = list(dict.fromkeys(types))
        if not len(set_of_types) == 2:
            raise ValueError("Tabulated potentials only supports two types.")
        first_type = set_of_types.pop()
        second_type = set_of_types.pop()
        colloid_potentials = ColloidPotentialsTabulated(
            radius_one=parameters.radii[first_type], radius_two=parameters.radii[second_type],
            surface_potential_one=parameters.surface_potentials[first_type],
            surface_potential_two=parameters.surface_potentials[second_type],
            colloid_potentials_parameters=potentials_parameters, use_log=parameters.use_log,
            cutoff_factor=parameters.cutoff_factor, periodic_boundary_conditions=not all_walls)
    else:
        colloid_potentials = ColloidPotentialsAlgebraic(
            colloid_potentials_parameters=potentials_parameters, use_log=parameters.use_log,
            cutoff_factor=parameters.cutoff_factor, periodic_boundary_conditions=not all_walls)
        
    for t in types:
        system.addParticle(parameters.masses[t])
        colloid_potentials.add_particle(radius=parameters.radii[t],
                                        surface_potential=parameters.surface_potentials[t])

    if include_walls:
        slj_walls = ShiftedLennardJonesWalls(wall_distances, parameters.epsilon, parameters.alpha,
                                             parameters.wall_directions)
        for i, t in enumerate(types):
            # noinspection PyTypeChecker
            slj_walls.add_particle(index=i, radius=parameters.radii[t])
        for force in slj_walls.yield_potentials():
            system.addForce(force)
        
    for force in colloid_potentials.yield_potentials():
        system.addForce(force)

    # Setting up ML-CVs (2D), hard-coding things for now...
    types_ml = []
    for itype in types:
        if itype == 'P':
            types_ml.append(55)
        elif itype == 'N':
            types_ml.append(17)
    types_ml = np.array(types_ml, dtype=np.int32)
    
    bias_factor = bias['bias_factor']
    height_factor = bias['height_factor']
    minmax = bias['minmax']
    model_name = bias['model_name']
    frequency = bias['frequency']

    cv0module = torch.jit.script(CVModule(f'{model_name}-combiner0.pkl', types_ml))
    cv1module = torch.jit.script(CVModule(f'{model_name}-combiner1.pkl', types_ml))
    cv0 = TorchForce(cv0module)
    cv1 = TorchForce(cv1module)
    # cv0.setUsesPeriodicBoundaryConditions(True)
    # cv1.setUsesPeriodicBoundaryConditions(True)
    
    cvforce0 = openmm.CustomCVForce('cv0')
    cvforce0.addCollectiveVariable('cv0', cv0)
    bv0 = app.metadynamics.BiasVariable(cvforce0, minmax[0][0], minmax[0][1], (minmax[0][1] - minmax[0][0]) / 40)

    cvforce1 = openmm.CustomCVForce('cv1')
    cvforce1.addCollectiveVariable('cv1', cv1)
    bv1 = app.metadynamics.BiasVariable(cvforce1, minmax[1][0], minmax[1][1], (minmax[1][1] - minmax[1][0]) / 40)

    path = pathlib.Path(__file__).parent

    meta = app.metadynamics.Metadynamics(system, [bv0, bv1], parameters.temperature,
                                         biasFactor=bias_factor, height=height_factor*unit.kilojoules_per_mole, frequency=frequency,
                                         saveFrequency=100, biasDir=f'{path}/../biases')
    
    if parameters.platform_name == "CUDA" or parameters.platform_name == "OpenCL":
        simulation = app.Simulation(topology, system, integrator, platform,
                                    platformProperties={"Precision": "mixed"})
    else:
        simulation = app.Simulation(topology, system, integrator, platform)

    # Would make more sense to put in set_up_reporters()...
    simulation.reporters.append(CVReporter('cv-values.txt',parameters.state_data_interval,meta))
        
    return simulation, meta


def set_up_reporters(parameters: RunParameters, simulation: app.Simulation, append_file: bool,
                     total_number_steps: int) -> None:
    simulation.reporters.append(GSDReporter(parameters.trajectory_filename, parameters.trajectory_interval,parameters.radii, parameters.surface_potentials, simulation,append_file=append_file))
    simulation.reporters.append(StatusReporter(max(1, total_number_steps // 100), total_number_steps))
    simulation.reporters.append(app.StateDataReporter(parameters.state_data_filename,parameters.state_data_interval, time=True,kineticEnergy=True, potentialEnergy=True, temperature=True,speed=True, append=append_file))
    simulation.reporters.append(app.CheckpointReporter(parameters.checkpoint_filename,parameters.checkpoint_interval))


def main():
    parser = argparse.ArgumentParser(description="Run OpenMM for a colloids system.")
    parser.add_argument("yaml_file", help="YAML file with simulation parameters", type=str)
    parser.add_argument("bias", help="bias parameters", type=str)
    parser.add_argument("--example", help="write an example YAML file and exit", action=ExampleAction)
    args = parser.parse_args()

    if not args.yaml_file.endswith(".yaml"):
        raise ValueError("The YAML file must have the .yaml extension.")
    
    if not args.bias.endswith(".yaml"):
        raise ValueError("The bias file must have the .yaml extension.")
    
    with open(args.bias, 'r') as f:
        bias = yaml.load(f, Loader=yaml.FullLoader)

    parameters = RunParameters.from_yaml(args.yaml_file)
    parameters.check_types_of_initial_configuration()



    types, positions, cell = read_xyz_file(parameters.initial_configuration)
    
    simulation, meta = set_up_simulation(parameters, types, cell, bias)
    
    simulation.context.setPositions(positions)
    if parameters.velocity_seed is not None:
        simulation.context.setVelocitiesToTemperature(parameters.temperature,
                                                      parameters.velocity_seed)
    else:
        simulation.context.setVelocitiesToTemperature(parameters.temperature)

    if parameters.minimize_energy_initially:
        # TODO: Do we want this?
        # Add reporter during minimization?
        # See https://openmm.github.io/openmm-cookbook/dev/notebooks/cookbook/report_minimization.html
        simulation.minimizeEnergy()

    set_up_reporters(parameters, simulation, False, parameters.run_steps)

    print("Running simulation...")
    meta.step(simulation, parameters.run_steps)
    # TODO: Automatically plot energies etc.
    # TODO: CHECK ALL SURFACE SEPARATIONS

    if parameters.final_configuration_gsd_filename is not None:
        write_gsd_file(parameters.final_configuration_gsd_filename, simulation, parameters.radii,
                       parameters.surface_potentials)

    if parameters.final_configuration_xyz_filename is not None:
        write_xyz_file(parameters.final_configuration_xyz_filename, simulation)

    return simulation, meta


if __name__ == '__main__':
    sim_object, metad_object = main()
    pkl.dump(np.array(metad_object.getFreeEnergy()), open('fes.p', 'wb'))
