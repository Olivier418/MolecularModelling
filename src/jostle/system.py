"""The simulated system: its topology and how the initial box is built."""
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .constants import AVOGADRO, BOLTZMANN_CONSTANT, MOLAR_VOLUMES
from .molecule_files import read_itp, read_xyz
from .packing import PACKING_TARGET, pack_molecules
from .thermo import kinetic_energy, velocity_scale_factor


@dataclass
class Topology:
    """Per-atom, per-bond, per-angle and per-dihedral data of a system.

    Bonds are stored sorted (i < j). molecule_index numbers the molecules
    0..M-1 in order; molecule_names holds the molecule name of every atom.
    """
    elements: np.ndarray         # (N,) element symbols
    masses: np.ndarray           # (N,) amu
    lj_epsilon: np.ndarray       # (N,) LJ well depth, kJ/mol
    lj_sigma: np.ndarray         # (N,) LJ diameter, nm
    bonds: np.ndarray            # (B, 2) atom indices
    k_bond: np.ndarray           # (B,) kJ/mol/nm^2
    bond_r0: np.ndarray          # (B,) nm
    angles: np.ndarray           # (A, 3) atom indices, middle atom centred
    k_angle: np.ndarray          # (A,) kJ/mol/rad^2
    theta0: np.ndarray           # (A,) rad
    dihedrals: np.ndarray        # (D, 4) atom indices
    rb_coefficients: np.ndarray  # (D, 4) Ryckaert-Bellemans, kJ/mol
    molecule_index: np.ndarray   # (N,)
    molecule_names: np.ndarray   # (N,)

    @property
    def nr_atoms(self):
        return len(self.masses)

    @staticmethod
    def concatenate(topologies):
        """Combines topologies into one system, renumbering atoms and
        molecules."""
        atom_offsets = np.cumsum([0] + [top.nr_atoms for top in topologies])
        molecule_offsets = np.cumsum(
            [0] + [len(np.unique(top.molecule_index)) for top in topologies])
        combined = {}
        for f in fields(Topology):
            parts = []
            for top, a0, m0 in zip(topologies, atom_offsets, molecule_offsets,
                                 strict=False):
                value = getattr(top, f.name)
                if f.name in ("bonds", "angles", "dihedrals"):
                    value = value + a0
                elif f.name == "molecule_index":
                    value = value + m0
                parts.append(value)
            combined[f.name] = np.concatenate(parts)
        return Topology(**combined)


def load_molecule(name, molecule_dir):
    """Reads <name>.xyz and <name>.itp into positions and a topology, and
    checks that the two files describe the same molecule."""
    elements, positions = read_xyz(Path(molecule_dir) / f"{name}.xyz")
    itp = read_itp(Path(molecule_dir) / f"{name}.itp")
    topology = Topology(elements=elements,
                        molecule_index=np.zeros(len(elements), dtype=int),
                        molecule_names=np.full(len(elements), name), **itp)
    check_molecule(name, positions, topology)
    return positions, topology


# ---- xyz/itp consistency ----
# The xyz geometry only has to be near the itp equilibrium, not at it: the
# shipped files are off by up to 7% (bonds) and 3.4 deg (angles), about the
# size of thermal vibrations. Typical file errors are far larger: an atom
# order mismatch puts a "bond" between non-bonded atoms (>= ~50% off), a unit
# mix-up scales all bonds by 10. Dihedrals are not checked: any torsion is a
# valid conformer.
BOND_TOLERANCE = 0.2                # max relative deviation |D / r0 - 1|
ANGLE_TOLERANCE = np.radians(20.)   # max deviation |theta - theta0|, rad


def check_molecule(name, positions, topology):
    """Raises a ValueError if the xyz positions and the itp topology of a
    molecule do not match: different atom counts, bonded indices out of
    range, or bond lengths/angles far from their equilibrium values."""
    top = topology
    n_xyz, n_itp = len(positions), top.nr_atoms
    if n_xyz != n_itp:
        raise ValueError(f"{name}: the xyz file has {n_xyz} atoms, "
                         f"the itp file {n_itp}")
    for section in ("bonds", "angles", "dihedrals"):
        idx = getattr(top, section)
        if idx.size and (idx.min() < 0 or idx.max() >= n_itp):
            raise ValueError(f"{name}: {section} refer to atoms outside "
                             f"0..{n_itp - 1}")

    # ---- Bond lengths ----
    lengths = np.linalg.norm(
        positions[top.bonds[:, 0]] - positions[top.bonds[:, 1]], axis=1)
    for k in np.flatnonzero(np.abs(lengths / top.bond_r0 - 1) > BOND_TOLERANCE):
        i, j = top.bonds[k]
        raise ValueError(
            f"{name}: bond {i}-{j} is {10 * lengths[k]:.3f} A in the xyz file "
            f"but r0 = {10 * top.bond_r0[k]:.3f} A in the itp file (more than "
            f"{BOND_TOLERANCE:.0%} apart; atom order or units?)")

    # ---- Angles (a-b-c, b in the centre) ----
    vec_ba = positions[top.angles[:, 0]] - positions[top.angles[:, 1]]
    vec_bc = positions[top.angles[:, 2]] - positions[top.angles[:, 1]]
    cos_theta = np.sum(vec_ba * vec_bc, axis=1) / (
        np.linalg.norm(vec_ba, axis=1) * np.linalg.norm(vec_bc, axis=1))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    for k in np.flatnonzero(np.abs(theta - top.theta0) > ANGLE_TOLERANCE):
        a, b, c = top.angles[k]
        raise ValueError(
            f"{name}: angle {a}-{b}-{c} is {np.degrees(theta[k]):.1f} deg in "
            f"the xyz file but theta0 = {np.degrees(top.theta0[k]):.1f} deg in "
            f"the itp file (more than {np.degrees(ANGLE_TOLERANCE):.0f} deg "
            f"apart; atom order?)")


# ------------------------- Initial positions ---------------------------------
def molecule_counts(box_length, weights):
    """Number of molecules of each type (weight > 0) in a box of the given
    size: the weights, normalised, fill the box at the molar volumes."""
    # Identify which molecules to include in the system and normalize weights
    molecules = [name for name, w in weights.items() if w > 0]
    fractions = np.array([weights[name] for name in molecules], dtype=float)
    fractions /= np.sum(fractions)

    # Compute the total nr of molecules that should be present in the system
    inverse_molar_volumes = np.array([1 / MOLAR_VOLUMES[m] for m in molecules])
    nr_molecules = int(box_length**3 * 1e-21 * AVOGADRO
                       * np.dot(fractions, inverse_molar_volumes))
    # Find nr of molecules per type
    nr_per_type = (fractions * nr_molecules).astype(int)
    if np.sum(nr_per_type) == 0:
        raise ValueError(f"box_length={box_length} is too small to accommodate "
                         f"required density. Please increase it.")
    return dict(zip(molecules, nr_per_type.tolist(), strict=True))


def build_initial_box(box_length, weights, molecule_dir, rng):
    """
    Creates the positions and topology of a box of given size, where each
    molecule in the weights dict is present with its associated weight.
    Molecules start on random sites of a cubic grid with random orientations,
    and are then packed (moved and rotated as rigid bodies) until no LJ atoms
    of different molecules are closer than PACKING_TARGET * sigma_ij.
    """
    counts = molecule_counts(box_length, weights)
    molecules, nr_per_type = list(counts), np.array(list(counts.values()))
    nr_molecules = int(np.sum(nr_per_type))

    # Compute the number of molecules along a single dimension of the box
    grid_size = int(np.ceil(nr_molecules ** (1 / 3)))

    # Build the topology and the template positions (centred on each
    # molecule's mean), one block per molecule type
    templates = [load_molecule(m, molecule_dir) for m in molecules]
    topology = Topology.concatenate(
        [top for (_, top), count in zip(templates, nr_per_type, strict=True)
         for _ in range(count)])
    local_positions = np.vstack(
        [np.tile(x - np.mean(x, axis=0), (count, 1))
         for (x, _), count in zip(templates, nr_per_type, strict=True)])

    # Start from random grid sites and random orientations
    coords = np.arange(grid_size)
    grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    sites = grid[rng.choice(grid_size**3, nr_molecules, replace=False)]
    print(f"  Initialized {nr_molecules} molecules "
          f"({topology.nr_atoms} atoms),")
    print(f"  on a grid of {grid_size}x{grid_size}x{grid_size} positions. "
          f"Effective weights:")
    for name, count in zip(molecules, nr_per_type, strict=True):
        print(f"    {name}: {count / nr_molecules:.3f}")

    # Remove the overlaps between LJ atoms of different molecules
    start = time.perf_counter()
    positions, iterations = pack_molecules(
        local_positions, topology, box_length / grid_size * sites, box_length,
        rng)
    print(f"  Packed in {iterations} iterations "
          f"({time.perf_counter() - start:.1f} s): no LJ atoms of different "
          f"molecules closer than {PACKING_TARGET} sigma.")
    return topology, positions


# ------------------------- Initial velocities --------------------------------
def initial_velocities(masses, temperature, rng):
    """
    Returns Maxwell-Boltzmann velocities (each component ~ N(0, kT/m)) with
    zero centre-of-mass momentum, scaled to give exactly the temperature.
    """
    # ---- Maxwell-Boltzmann: sigma = sqrt(kT/m) per atom (nm/ps) ----
    v = rng.normal(0, 1, (len(masses), 3))
    v *= np.sqrt(BOLTZMANN_CONSTANT * temperature / masses)[:, None]
    # ---- Remove centre-of-mass momentum (a scalar rescale keeps it 0) ----
    v -= (masses @ v) / masses.sum()
    return v * velocity_scale_factor(kinetic_energy(masses, v), len(masses),
                                     temperature)
