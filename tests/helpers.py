"""Shared fixtures for the tests: small topologies and helpers."""
import numpy as np

from jostle.constants import BUILTIN_MOLECULE_DIR
from jostle.forcefield import ForceField
from jostle.system import Topology, load_molecule


def make_topology(n, **arrays):
    """A single-molecule topology of n atoms; missing data is empty/zero."""
    defaults = dict(
        elements=np.full(n, "H"), masses=np.ones(n),
        lj_epsilon=np.zeros(n), lj_sigma=np.zeros(n),
        bonds=np.zeros((0, 2), int), k_bond=np.zeros(0),
        bond_r0=np.zeros(0), angles=np.zeros((0, 3), int),
        k_angle=np.zeros(0), theta0=np.zeros(0),
        dihedrals=np.zeros((0, 4), int), rb_coefficients=np.zeros((0, 4)),
        molecule_index=np.zeros(n, int),
        molecule_names=np.full(n, "Test"))
    defaults.update(arrays)
    return Topology(**defaults)


def forcefield(topology, box_length=0., lj_cutoff=np.inf):
    """ForceField without periodic boundaries (box_length=0) by default."""
    return ForceField(topology, box_length, lj_cutoff=lj_cutoff)


def forces_and_energy(ff, x):
    """Total force and potential energy at positions x."""
    result = ff.compute_forces(x, compute_energy=True)
    return result.forces, sum(result.potential_energies.values())


# ------------------------------ Real molecules -------------------------------
def water_ethanol_pair():
    """One water and one ethanol molecule, close enough to interact via LJ."""
    x_w, water = load_molecule("Water", BUILTIN_MOLECULE_DIR)
    x_e, ethanol = load_molecule("Ethanol", BUILTIN_MOLECULE_DIR)
    x = np.vstack([x_w, x_e + np.array([0.45, 0.1, -0.05])]).astype(float)
    return x, Topology.concatenate([water, ethanol])
