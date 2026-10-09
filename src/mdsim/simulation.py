"""Running a simulation: setup, soft start, the integration loop and output."""
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .forcefield import ForceField
from .geometry import wrap_molecules
from .integrators import METHODS, euler, initial_state, step
from .io import write_itp, write_xyz
from .observables import RDF, PhaseSpace
from .plotting import (computation_piechart, energy_plot, phase_space_plots,
                       rdf_plots, temperature_plot)
from .progress import ProgressBar
from .system import build_system, initial_velocities
from .vmd import visualize_xyz_with_vmd


def output_folder(config):
    """trajectories/<system>/<integration settings>, named after the
    parameters of the run."""
    name = '_'.join(f"{float(w)}{m}" for m, w in config.weights.items()
                    if w > 0)
    system = (f"boxsize={config.boxsize:.2f}_"
              f"roff={config.random_grid_offset}__{name}")
    run = f"dt={config.dt}_T={config.end_time}_steps={config.save_steps}_"
    if config.f_threshold is not None:
        run += f"fthr={config.f_threshold}_"
    else:
        run += f"rcut-sclr={config.r_cutoff_scalar}_"
    run += f"temp={config.initial_temp}_control={config.temp_control}"
    return Path(config.output_dir) / system / run


def soft_start(forcefield, x, masses, method, dt, r_threshold_scalar,
               max_iterations=100_000):
    """Integrates from rest, with LJ forces capped at r_threshold_scalar *
    sigma, until no LJ pair is closer than that. Removes overlaps of the
    initial grid that would otherwise blow up the simulation."""
    def compute(x):
        return forcefield.compute(x, cap_scalar=r_threshold_scalar)

    sigma = forcefield.sigma_LJ
    threshold = r_threshold_scalar * sigma
    state, result = initial_state(x, np.zeros_like(x), compute, masses)
    smallest_dist = np.min(result.D_LJ / sigma, initial=np.inf)
    total = np.sum(result.D_LJ < threshold)
    progress = ProgressBar(total)

    state, result = step(euler, state, compute, masses, dt)
    iteration = 0
    while np.any(result.D_LJ < threshold):
        iteration += 1
        if iteration > max_iterations:
            raise RuntimeError(
                f"Equilibration did not separate all LJ pairs within "
                f"{max_iterations} steps. Lower r_threshold_scalar or check "
                f"the LJ cutoff.")
        state, result = step(method, state, compute, masses, dt)
        progress.update(total - np.sum(result.D_LJ < threshold))
    progress.finish()
    print(f"  Equilibrium configuration initialized. Smallest distance"
          f"  increased from {smallest_dist:.3f} to "
          f"{np.min(result.D_LJ / sigma, initial=np.inf):.3f}...")
    return state.x


def run(config):
    """Runs the simulation described by a validated Config. Returns the
    folder the results were written to."""
    rng = np.random.default_rng(config.seed)
    method = METHODS[config.method]
    dt, save_steps = config.dt, config.save_steps

    print("Creating initial configuration...")
    topology, x = build_system(config.boxsize, config.random_grid_offset,
                               config.weights, config.molecule_dir, rng)
    masses = topology.masses

    print("Computing simulation parameters...")
    forcefield = ForceField(topology, config.boxsize, config.r_cutoff_scalar,
                            config.f_threshold)
    if config.r_threshold_scalar is not None:
        print("  Computing equilibrium configuration...")
        x = soft_start(forcefield, x, masses, method, dt,
                       config.r_threshold_scalar)

    print("Saving initial configuration data to files...")
    folder = output_folder(config)
    folder.mkdir(parents=True, exist_ok=True)
    write_itp(folder / "INITIAL.itp", topology)
    with open(folder / "INITIAL.xyz", "w") as f:
        write_xyz(f, wrap_molecules(x, topology.molec_index, config.boxsize),
                  topology.atoms, 0)

    print("Initializing data arrays...")
    timesteps = int(np.ceil(config.end_time / dt))
    nr_saves = 1 + timesteps // save_steps
    save_interval = dt * save_steps
    if config.energy_graph:
        energies = {name: np.empty(nr_saves)
                    for name in ("Bond", "Angular", "LJ", "Dihedral")}
    if config.energy_graph or config.temp_graph:
        kinetic = np.empty(nr_saves)

    # Energies are computed alongside the forces, on save steps only
    def compute_with_energy(x):
        return forcefield.compute(x, compute_energy=config.energy_graph)
    if config.phase_space_diagram:
        print("  Initializing phase space...")
        phase_space = PhaseSpace(topology, config.bonds_of_interest, nr_saves)
    if config.RDF_vis:
        print("  Initializing RDF...")
        rdf = RDF(topology, config.pairs_of_interest, config.boxsize,
                  config.RDF_thin, config.RDF_dr, nr_saves, rng)

    timings = {"Writing XYZ": 0., "RDF": 0., "PS diag": 0.}

    @contextmanager
    def timed(label):
        t0 = time.perf_counter()
        yield
        timings[label] += time.perf_counter() - t0

    def save(save_idx, state, result, trajectory, t):
        """Store the measurements for later visualization."""
        if config.RDF_vis:
            with timed("RDF"):
                rdf.record(save_idx, state.x)
        if config.energy_graph:
            for name, value in result.energies.items():
                energies[name][save_idx] = value
        if config.energy_graph or config.temp_graph:
            kinetic[save_idx] = state.KE
        # Always save the positions, after projecting them into the box.
        with timed("Writing XYZ"):
            write_xyz(trajectory, wrap_molecules(state.x, topology.molec_index,
                                                 config.boxsize),
                      topology.atoms, t)
        if config.phase_space_diagram:
            with timed("PS diag"):
                phase_space.record(save_idx, result, state.v)

    print("Starting integration...")
    forcefield.timings = dict.fromkeys(forcefield.timings, 0.)
    start = time.perf_counter()
    v = initial_velocities(masses, config.initial_temp, rng)
    temp = config.initial_temp if config.temp_control else None
    state, result = initial_state(x, v, compute_with_energy, masses)
    progress = ProgressBar(timesteps)
    with open(folder / "INTEGRATED.xyz", "w") as trajectory:
        save(0, state, result, trajectory, 0)
        for i in range(1, timesteps + 1):
            saving = i % save_steps == 0
            # The first step has no history yet, so it is always Euler
            state, result = step(euler if i == 1 else method, state,
                                 compute_with_energy if saving
                                 else forcefield.compute, masses, dt, temp)
            if saving:
                save(i // save_steps, state, result, trajectory, i * dt)
            progress.update(i)
    progress.finish()
    total_time = time.perf_counter() - start
    print(f"  Integration took {total_time:.4f} seconds")

    show = config.show_plots
    if config.RDF_vis:
        print("Generating RDF figures.")
        rdf_plots(rdf, save_interval, folder / "RDF", topology.r0)
        print("Finished RDF figures.")
    if config.energy_graph:
        energy_plot({**energies, "Kinetic": kinetic}, save_interval, folder,
                    show)
    if config.temp_graph:
        temperature_plot(kinetic, 3 * topology.nr_atoms, save_interval,
                         folder, show)
    if config.phase_space_diagram:
        ps_folder = folder / "ps_diagram"
        ps_folder.mkdir(exist_ok=True)
        phase_space_plots(phase_space, ps_folder, show)
    if config.comptime_vis:
        computation_piechart({**forcefield.timings, **timings}, total_time,
                             timesteps, folder, show)
    if config.VMD_vis:
        print("Opening VMD")
        d_max = 1.05 * np.max(result.D_bonds)  # Used for dynamic bonds in VMD
        visualize_xyz_with_vmd(folder / "INTEGRATED.xyz", 10 * config.boxsize,
                               10 * d_max, config.vmd_path)
    return folder
