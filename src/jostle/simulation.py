"""Running a simulation: setup, the integration loop and output."""
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .constants import BUILTIN_MOLECULE_DIR, TRAJECTORY_FILE
from .forcefield import POTENTIAL_TERMS, ForceField
from .geometry import wrap_molecules
from .integrators import (
    INTEGRATORS,
    initial_state,
    integration_step,
    startup_integrator,
)
from .minimise import minimise_energy, thermal_force
from .molecule_files import write_xyz
from .observables import RDF, PhaseSpace
from .results import Results
from .system import build_initial_box, initial_velocities


def config_hash(config):
    """8 hex digits that identify all settings except output_dir: equal
    configs give equal hashes, any difference (seed and recording settings
    included) another one. Ints count as floats (box_length=2 is
    box_length=2.0), and the built-in molecule_dir is left out, so the hash
    does not depend on where the package is installed."""
    def canonical(value):
        if isinstance(value, dict):
            return {k: canonical(v) for k, v in value.items()}
        if isinstance(value, list):
            return [canonical(v) for v in value]
        if isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        return value

    d = config.to_dict()
    del d["output_dir"]
    if config.system.molecule_dir == BUILTIN_MOLECULE_DIR:
        del d["system"]["molecule_dir"]
    text = json.dumps(canonical(d), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def run_folder(config):
    """<output_dir>/<mixture>_<box_length>nm_<end_time>ps_<hash>: a readable
    prefix, and the hash of all settings (config.json holds them), so runs
    with the same settings overwrite each other and all others don't."""
    system = config.system
    mixture = '_'.join(f"{w:g}{m}" for m, w in system.weights.items()
                       if w > 0)
    name = (f"{mixture}_{system.box_length:g}nm_{config.end_time:g}ps_"
            f"{config_hash(config)}")
    return Path(config.output_dir) / name


def first_phase_space_step(config):
    """The phase space is recorded every step from this one on: the last
    phase_space_window ps of the run (None: all of it)."""
    nr_steps, ps_time = config.nr_steps, config.recording.phase_space_window
    if ps_time is None:
        return 0
    return max(0, nr_steps - round(ps_time / config.numerics.dt))


def run(config):
    """Runs the simulation described by a Config, writes the trajectory and
    the recorded data (results.npz, see Results.save) to a run folder, and
    returns the Results. Figures are made on request, from the Results."""
    from .checks import check_config

    config.validate()
    check_config(config)
    system, recording = config.system, config.recording
    rng = np.random.default_rng(config.seed)
    integrator = INTEGRATORS[config.numerics.integrator]
    dt = config.numerics.dt
    box_length = system.box_length

    print("Creating initial configuration...")
    topology, x = build_initial_box(box_length, system.weights,
                               system.molecule_dir, rng)
    # Its own force field, so the timings below cover the integration only
    start = time.perf_counter()
    x, iterations = minimise_energy(
        x, ForceField(topology, box_length, config.numerics.lj_cutoff),
        system.temperature)
    print(f"  Minimised the energy in {iterations} iterations "
          f"({time.perf_counter() - start:.1f} s): largest force below "
          f"{thermal_force(topology, system.temperature):.0f} kJ/mol/nm, "
          f"the thermal force of the stiffest bond.")
    masses = topology.masses
    forcefield = ForceField(topology, box_length, config.numerics.lj_cutoff)
    folder = run_folder(config)
    folder.mkdir(parents=True, exist_ok=True)

    # ---- Recording: each measurement on its own steps ----
    nr_steps = config.nr_steps
    energy_interval = recording.energy_interval
    rdf_interval = recording.rdf_interval

    def frame_times(interval):
        return dt * interval * np.arange(1 + nr_steps // interval)
    times = frame_times(energy_interval)
    potential_energies = lj_correction = phase_space = rdf = None
    if recording.record_potential_energies:
        potential_energies = {name: np.empty(len(times))
                              for name in POTENTIAL_TERMS}
        lj_correction = np.empty(len(times))
    kinetic_energies = np.empty(len(times))

    # Energies are computed alongside the forces, on energy steps only
    def compute_with_energy(x):
        return forcefield.compute_forces(
            x, compute_energy=recording.record_potential_energies)
    phase_space_start = first_phase_space_step(config)
    if recording.phase_space_bonds:
        phase_space = PhaseSpace(
            topology, recording.phase_space_bonds,
            dt * np.arange(phase_space_start, nr_steps + 1))
    if recording.rdf_pairs:
        # A spawned generator: the pairs it samples do not shift the random
        # numbers of the dynamics, so the trajectory is the same whatever
        # the RDF settings.
        rdf = RDF(topology, recording.rdf_pairs, box_length,
                  recording.rdf_pair_fraction, recording.rdf_bin_width,
                  frame_times(rdf_interval), rng.spawn(1)[0])

    timings = {"Trajectory output": 0., "RDF": 0., "Phase space": 0.}

    @contextmanager
    def timed(label):
        t0 = time.perf_counter()
        yield
        timings[label] += time.perf_counter() - t0

    def record_step(i, state, result, trajectory_file):
        """Stores the measurements due at step i."""
        if i % rdf_interval == 0 and rdf is not None:
            with timed("RDF"):
                rdf.record(i // rdf_interval, state.x)
        if i % energy_interval == 0:
            if potential_energies is not None:
                for name, value in result.potential_energies.items():
                    potential_energies[name][i // energy_interval] = value
                lj_correction[i // energy_interval] = result.lj_correction
            kinetic_energies[i // energy_interval] = state.kinetic_energy
        # The positions are saved after projecting them into the box.
        if i % recording.trajectory_interval == 0:
            with timed("Trajectory output"):
                write_xyz(trajectory_file, topology.elements,
                          wrap_molecules(state.x, topology.molecule_index,
                                         box_length),
                          i * dt)
        if i >= phase_space_start and phase_space is not None:
            with timed("Phase space"):
                phase_space.record(i - phase_space_start, result, state.v)

    print("Integrating...")
    start = time.perf_counter()
    v = initial_velocities(masses, system.temperature, rng)
    target_temperature = system.temperature if system.thermostat else None
    state, result = initial_state(x, v, compute_with_energy, masses)
    with open(folder / TRAJECTORY_FILE, "w") as trajectory_file:
        record_step(0, state, result, trajectory_file)
        for i in tqdm(range(1, nr_steps + 1)):
            # The first step has no history yet, so it must be self-starting
            step_integrator = (startup_integrator(integrator) if i == 1
                               else integrator)
            compute_forces = (compute_with_energy if i % energy_interval == 0
                              else forcefield.compute_forces)
            state, result = integration_step(
                step_integrator, state, compute_forces, masses, dt,
                target_temperature)
            record_step(i, state, result, trajectory_file)
    wall_time = time.perf_counter() - start
    print(f"  Integration took {wall_time:.4f} seconds "
          f"({forcefield.neighbour_list.nr_builds} neighbour list builds)")

    results = Results(
        config=config, folder=folder, topology=topology,
        energy_times=times, potential_energies=potential_energies,
        lj_correction=lj_correction,
        kinetic_energies=kinetic_energies, phase_space=phase_space, rdf=rdf,
        timings={**forcefield.timings, **timings}, wall_time=wall_time)
    results.save()
    print(f"Results written to {folder}")
    return results
