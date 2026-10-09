"""The simulated system: its topology and how the initial box is built."""
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from .constants import AVOGADRO, KBOLTZ, MOLAR_VOLUMES
from .io import read_itp, read_xyz


@dataclass
class Topology:
    """Per-atom, per-bond, per-angle and per-dihedral data of a system.

    Bonds are stored sorted (i < j). molec_index numbers the molecules
    0..M-1 in order; molec_names holds the molecule name of every atom.
    """
    atoms: np.ndarray        # (N,) element symbols
    masses: np.ndarray       # (N,) amu
    epsilon: np.ndarray      # (N,) LJ well depth, kJ/mol
    sigma: np.ndarray        # (N,) LJ diameter, nm
    bonds: np.ndarray        # (B, 2) atom indices
    kb: np.ndarray           # (B,) kJ/mol/nm^2
    r0: np.ndarray           # (B,) nm
    angles: np.ndarray       # (A, 3) atom indices, middle atom in the centre
    ktheta: np.ndarray       # (A,) kJ/mol/rad^2
    theta0: np.ndarray       # (A,) rad
    dihedrals: np.ndarray    # (D, 4) atom indices
    C: np.ndarray            # (D, 4) Ryckaert-Bellemans constants, kJ/mol
    molec_index: np.ndarray  # (N,)
    molec_names: np.ndarray  # (N,)

    @property
    def nr_atoms(self):
        return len(self.masses)

    def replicate(self, count):
        """A topology of `count` copies of this (single molecule) topology."""
        n = self.nr_atoms
        offsets = n * np.arange(count)

        def shift(idx):
            return (idx[None] + offsets[:, None, None]).reshape(-1,
                                                                idx.shape[1])
        replicated = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("bonds", "angles", "dihedrals"):
                replicated[f.name] = shift(value)
            elif f.name == "molec_index":
                replicated[f.name] = np.repeat(np.arange(count), n)
            elif f.name == "C":
                replicated[f.name] = np.tile(value, (count, 1))
            else:
                replicated[f.name] = np.tile(value, count)
        return Topology(**replicated)

    @staticmethod
    def concatenate(topologies):
        """Combines topologies into one system, renumbering atoms and
        molecules."""
        atom_offsets = np.cumsum([0] + [t.nr_atoms for t in topologies])
        molec_offsets = np.cumsum(
            [0] + [len(np.unique(t.molec_index)) for t in topologies])
        combined = {}
        for f in fields(Topology):
            parts = []
            for t, a0, m0 in zip(topologies, atom_offsets, molec_offsets):
                value = getattr(t, f.name)
                if f.name in ("bonds", "angles", "dihedrals"):
                    value = value + a0
                elif f.name == "molec_index":
                    value = value + m0
                parts.append(value)
            combined[f.name] = np.concatenate(parts)
        return Topology(**combined)


def load_molecule(name, molecule_dir):
    """Reads <name>.xyz and <name>.itp into positions and a topology."""
    atoms, positions = read_xyz(Path(molecule_dir) / f"{name}.xyz")
    itp = read_itp(Path(molecule_dir) / f"{name}.itp")
    topology = Topology(atoms=atoms,
                        molec_index=np.zeros(len(atoms), dtype=int),
                        molec_names=np.full(len(atoms), name), **itp)
    return positions, topology


# ------------------------- Initial positions ---------------------------------
def rotate_molecule(array, theta, phi):
    """Rotates an xyz array around the z axis by theta, and then around the y
    axis by phi."""
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])
    return array @ (Rz @ Ry).T


def orient_molecule(positions, rng):
    """Centres a molecule at the origin and rotates it such that it fits in
    the smallest box possible (minimising the largest box dimension)."""
    positions = positions - np.mean(positions, axis=0)

    def largest_dimension(angles):
        X = rotate_molecule(positions, *angles)
        return np.max(np.max(X, axis=0) - np.min(X, axis=0))

    bounds = [(-np.pi, np.pi), (-np.pi, np.pi)]
    result = differential_evolution(largest_dimension, bounds, rng=rng)
    return rotate_molecule(positions, *result.x)


def build_system(boxsize, random_grid_offset, weights, molecule_dir, rng):
    """
    Creates the positions and topology of a grid of given box size, where
    each molecule in the weights dict is present with its associated weight.
    The random_grid_offset parameter slightly randomizes the offset away from
    the exact grid points and should be set between 0 and 0.5.
    """
    # Identify which molecules to include in the system and normalize weights
    molecules = [name for name, w in weights.items() if w > 0]
    w = np.array([weights[name] for name in molecules], dtype=float)
    w /= np.sum(w)

    # Compute the total nr of molecules that should be present in the system
    inverse_molar_volumes = np.array([1 / MOLAR_VOLUMES[m] for m in molecules])
    nr_molecules = int(
        boxsize**3 * 1e-21 * AVOGADRO * np.dot(w, inverse_molar_volumes))
    # Find nr of molecules per type
    nr_per_type = (w * nr_molecules).astype(int)
    total_molecules = int(np.sum(nr_per_type))
    if total_molecules == 0:
        raise ValueError(f"Boxsize={boxsize} is too small to accommodate "
                         f"required density. Please increase it.")

    # Compute the number of molecules along a single dimension of the box
    N = int(np.ceil(total_molecules ** (1 / 3)))

    # Build the topology and template positions, one block per molecule type
    templates = [load_molecule(m, molecule_dir) for m in molecules]
    topology = Topology.concatenate(
        [t.replicate(count) for (_, t), count in zip(templates, nr_per_type)])
    template_positions = np.vstack(
        [np.tile(orient_molecule(x, rng), (count, 1))
         for (x, _), count in zip(templates, nr_per_type)]).astype('f')

    # Choose random grid positions for the molecules, with a random offset
    points = np.arange(N)
    grid = np.array(np.meshgrid(points, points, points)).T.reshape(-1, 3)
    sites = grid[rng.choice(N**3, total_molecules, replace=False)]
    offsets = rng.uniform(low=-random_grid_offset, high=random_grid_offset,
                          size=(total_molecules, 3))
    shifts = boxsize / N * (sites + offsets)
    positions = template_positions + shifts[topology.molec_index]

    print(f"  Initialized {total_molecules} molecules "
          f"({topology.nr_atoms} atoms),")
    print(f"  in a grid of {N}x{N}x{N} positions. Effective weights:")
    for name, count in zip(molecules, nr_per_type):
        print(f"    {name}: {count / total_molecules:.3f}")
    return topology, positions


# ------------------------- Initial velocities --------------------------------
def kinetic_energy(masses, v):
    """Total kinetic energy (kJ/mol)."""
    return 0.5 * np.dot(masses, np.sum(v**2, axis=1))


def temperature_scale(KE, nr_atoms, temp):
    """Factor by which to scale velocities with kinetic energy KE such that
    the system reaches temperature temp."""
    Nf = 3 * nr_atoms
    return np.sqrt(temp * Nf * KBOLTZ / (2 * KE))


def initial_velocities(masses, temp, rng):
    """
    Returns initial velocities in random directions, normalized to give the
    system a chosen temperature.
    """
    v = rng.normal(0, 1, (len(masses), 3))
    return v * temperature_scale(kinetic_energy(masses, v), len(masses), temp)
