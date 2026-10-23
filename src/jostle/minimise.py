"""Energy minimisation of the initial box.

Molecules start on grid sites with random orientations, where LJ atoms of
neighbouring molecules can nearly coincide. Steepest descent on the
potential energy of the run's force field removes these overlaps before the
dynamics starts. Every atom moves along its own force, so the bonded forces
carry whole molecules along: they translate and rotate, and deform only
slightly.
"""
import numpy as np

from .constants import BOLTZMANN_CONSTANT


def thermal_force(topology, temperature):
    """The rms force of the stiffest harmonic bond at the temperature
    (kJ/mol/nm). For E = 1/2 k x^2, Boltzmann statistics give <x^2> = kT/k,
    so the force F = -k x has <F^2> = kT k. The stiffest bonds have the
    largest thermal forces in the liquid (water O-H: 1116 at 298 K)."""
    return np.sqrt(BOLTZMANN_CONSTANT * temperature
                   * np.max(topology.k_bond, initial=0.))


def minimise_energy(positions, forcefield, temperature):
    """
    Steepest descent on the potential energy until the largest force on any
    atom is below thermal_force. The forces are then like those of thermal
    bond vibrations at the temperature: stopping earlier leaves overlaps
    that heat the box, minimising further removes the potential energy the
    liquid has at that temperature, which then cools it. Returns the
    positions and the number of iterations used.
    """
    def energy_and_forces(x):
        result = forcefield.compute_forces(x, compute_energy=True)
        return sum(result.potential_energies.values()), result.forces

    tolerance = thermal_force(forcefield.topology, temperature)
    energy, forces = energy_and_forces(positions)
    max_displacement = 0.01  # largest atom displacement per iteration (nm)
    iteration = 0
    while (max_force := np.max(np.linalg.norm(forces, axis=1))) > tolerance:
        iteration += 1
        # ---- Trial move along the forces; accept only if the energy drops ----
        trial_positions = positions + max_displacement / max_force * forces
        if np.array_equal(trial_positions, positions):
            raise RuntimeError("Energy minimisation stalled: the step became "
                               "too small to move any atom.")
        trial_energy, trial_forces = energy_and_forces(trial_positions)
        if trial_energy < energy:
            positions, energy, forces = (trial_positions, trial_energy,
                                         trial_forces)
            max_displacement = min(1.2 * max_displacement, 0.05)
        else:
            max_displacement *= 0.5
    return positions, iteration
