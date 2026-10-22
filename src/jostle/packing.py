"""Packing molecules into the periodic box without LJ overlaps.

Molecules are moved as rigid bodies (a centre and a rotation each) to remove
every overlap between Lennard-Jones atoms of different molecules, in the
style of Packmol. Atoms without LJ interactions (epsilon or sigma = 0) are
ignored, so they may end up close to other atoms.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import periodic_pairs

# Closest allowed distance between LJ atoms of different molecules, as a
# multiple of sigma_ij. A packed box at 0.85 starts calmly (without a
# thermostat, the default box heats from 298 to ~305 K), while at 0.9 liquid
# mixtures start to jam (default box: 5 s to pack at 0.85, 320 s at 0.9).
PACKING_TARGET = 0.85
# The penalty acts up to this factor beyond the target. Its push fades to 0
# at the penalty distance, so without a margin the last pairs only creep
# towards the target and never reach it. 1.02-1.08 all work; much larger
# values jam (the effective target becomes too dense).
PACKING_MARGIN = 1.04
# Packing gives up (the box is too dense) after this many iterations
PACKING_MAX_ITERATIONS = 20_000


def overlap_penalty(positions, molecule_of_atom, sigma, box_length,
                    contact_factor):
    """Penalty P = sum (d0^2 - D^2)^2 over pairs of atoms of different
    molecules closer than d0 = contact_factor * (sigma_i + sigma_j) / 2,
    using the minimum image. Returns P, the force -dP/dX on every atom and
    the smallest D / sigma_ij of these pairs (inf if there are none)."""
    a, b = periodic_pairs(positions, box_length,
                          contact_factor * np.max(sigma))
    keep = molecule_of_atom[a] != molecule_of_atom[b]
    a, b = a[keep], b[keep]

    vectors = positions[a] - positions[b]
    vectors -= box_length * np.floor(vectors / box_length + 0.5)
    sq_distances = np.sum(vectors**2, axis=1)
    sigma_ab = 0.5 * (sigma[a] + sigma[b])
    gap = (contact_factor * sigma_ab)**2 - sq_distances
    overlapping = gap > 0
    a, b = a[overlapping], b[overlapping]
    vectors, gap = vectors[overlapping], gap[overlapping]
    min_distance_ratio = (
        np.sqrt(np.min(sq_distances[overlapping]
                       / sigma_ab[overlapping]**2))
        if len(gap) else np.inf)

    # -dP/dX_a = 4 (d0^2 - D^2) (X_a - X_b): pushes the pair apart
    pair_forces = (4 * gap)[:, None] * vectors
    nr_atoms = len(positions)
    forces = np.zeros_like(positions)
    for k in range(3):
        forces[:, k] = (
            np.bincount(a, weights=pair_forces[:, k], minlength=nr_atoms)
            - np.bincount(b, weights=pair_forces[:, k], minlength=nr_atoms))
    return np.sum(gap**2), forces, min_distance_ratio


def pack_molecules(local_positions, topology, centres, box_length, rng):
    """
    Gives the molecules of the topology random orientations, then moves and
    rotates them as rigid bodies until no two LJ atoms of different
    molecules are closer than PACKING_TARGET * sigma_ij. Every atom k sits
    at rotation[m].apply(local_positions[k]) + centres[m] for its molecule m.
    Returns the packed positions of all atoms and the number of iterations
    used.

    The penalty acts up to PACKING_MARGIN * PACKING_TARGET * sigma_ij, so
    that pairs at the target still get pushed.
    """
    nr_molecules = len(centres)
    # Uniformly distributed rotations (normalised Gaussian quaternions)
    rotations = Rotation.from_quat(rng.normal(size=(nr_molecules, 4)))
    lj_atoms = np.nonzero((topology.lj_epsilon > 0)
                          & (topology.lj_sigma > 0))[0]
    molecule_index = topology.molecule_index
    lj_local = local_positions[lj_atoms]
    lj_molecule = molecule_index[lj_atoms]
    lj_sigma = topology.lj_sigma[lj_atoms]

    # Size of each molecule (largest distance of an LJ atom to its centre),
    # so rotations move atoms by about the same amount as translations
    radius = np.zeros(nr_molecules)
    np.maximum.at(radius, lj_molecule, np.linalg.norm(lj_local, axis=1))
    radius = np.maximum(radius, 0.01)

    def lj_positions(centres, rotations):
        return rotations[lj_molecule].apply(lj_local) + centres[lj_molecule]

    positions = lj_positions(centres, rotations)
    contact_factor = PACKING_MARGIN * PACKING_TARGET
    penalty, forces, min_ratio = overlap_penalty(
        positions, lj_molecule, lj_sigma, box_length, contact_factor)
    max_displacement = 0.01  # largest atom displacement per iteration (nm)
    iteration = 0
    while min_ratio < PACKING_TARGET:
        iteration += 1
        if iteration > PACKING_MAX_ITERATIONS:
            raise RuntimeError(
                f"Packing did not remove all LJ overlaps within "
                f"{PACKING_MAX_ITERATIONS} iterations. Use a larger box or a "
                f"lower PACKING_TARGET.")

        # ---- Net force and torque per molecule ----
        torque = np.cross(positions - centres[lj_molecule], forces)
        molecule_force = np.zeros((nr_molecules, 3))
        molecule_torque = np.zeros((nr_molecules, 3))
        for k in range(3):
            molecule_force[:, k] = np.bincount(
                lj_molecule, weights=forces[:, k], minlength=nr_molecules)
            molecule_torque[:, k] = np.bincount(
                lj_molecule, weights=torque[:, k], minlength=nr_molecules)
        # Torque / radius is the force-like push on the outer atoms
        push = np.maximum(np.linalg.norm(molecule_force, axis=1),
                          np.linalg.norm(molecule_torque, axis=1) / radius)
        scale = max_displacement / np.max(push)

        # ---- Trial move; accept only if the penalty drops ----
        trial_centres = centres + scale * molecule_force
        trial_rotations = (Rotation.from_rotvec(
            scale * molecule_torque / radius[:, None]**2) * rotations)
        trial_positions = lj_positions(trial_centres, trial_rotations)
        trial_penalty, trial_forces, trial_min_ratio = overlap_penalty(
            trial_positions, lj_molecule, lj_sigma, box_length,
            contact_factor)
        if trial_penalty < penalty:
            centres, rotations = trial_centres, trial_rotations
            positions, penalty = trial_positions, trial_penalty
            forces, min_ratio = trial_forces, trial_min_ratio
            max_displacement = min(1.2 * max_displacement, 0.05)
        else:
            max_displacement *= 0.5

    positions = (rotations[molecule_index].apply(local_positions)
                 + centres[molecule_index])
    return positions, iteration
