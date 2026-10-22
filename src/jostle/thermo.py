"""Kinetic energy, degrees of freedom and temperature scaling."""
import numpy as np

from .constants import BOLTZMANN_CONSTANT


def kinetic_energy(masses, velocities):
    """Total kinetic energy (kJ/mol)."""
    return 0.5 * np.dot(masses, np.sum(velocities**2, axis=1))


def degrees_of_freedom(nr_atoms):
    """Degrees of freedom: 3 per atom, minus the 3 of the centre-of-mass
    momentum, which initial_velocities removes and the forces conserve."""
    return 3 * nr_atoms - 3


def velocity_scale_factor(kinetic_energy, nr_atoms, temperature):
    """Factor by which to scale velocities with the given kinetic energy such
    that the system reaches the given temperature."""
    nr_dof = degrees_of_freedom(nr_atoms)
    return np.sqrt(temperature * nr_dof * BOLTZMANN_CONSTANT
                   / (2 * kinetic_energy))
