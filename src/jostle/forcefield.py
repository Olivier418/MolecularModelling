"""Force field setup (LJ mixing rules and cutoffs) and evaluation of forces
and energies. Each bonded term computes its geometry once and returns its
force contributions per atom, plus its energy when asked. LJ forces are
evaluated by a compiled kernel over the pairs of a neighbour list."""
import time
from dataclasses import dataclass

import numpy as np
from numba import njit

from .geometry import (
    angle_geometry,
    bond_index_map,
    dihedral_geometry,
    pair_vectors,
)
from .neighbours import NEIGHBOUR_SKIN, NeighbourList

# The potential energy terms, in the order they are reported
POTENTIAL_TERMS = ("Bond", "Angle", "LJ", "Dihedral")


@dataclass
class ForceEvaluation:
    """Forces at a configuration, with the bond vectors computed on the way
    (reused by PhaseSpace). potential_energies maps each of POTENTIAL_TERMS
    to its value if it was requested, else it is None."""
    forces: np.ndarray
    bond_vectors: np.ndarray
    bond_lengths: np.ndarray
    potential_energies: dict | None


# ------------------------------- Force terms ---------------------------------
# Each term returns (atom indices, force vectors, energy). The force on atom
# idx[k] gets vec[k] added; energy is None unless compute_energy is True.

def bond_term(bond_vectors, bond_lengths, bonds, k_bond, bond_r0,
              compute_energy):
    """Harmonic bonds: E = 1/2 k_bond (D - r0)^2."""
    F = ((k_bond * (bond_r0 - bond_lengths) / bond_lengths)[:, None]
         * bond_vectors)
    energy = (0.5 * np.dot(k_bond, (bond_lengths - bond_r0)**2)
              if compute_energy else None)
    return (bonds[:, 0], bonds[:, 1]), (F, -F), energy


def angle_term(bond_vectors, bond_lengths, angle_bond_map, angles, k_angle,
               theta0, compute_energy):
    """Harmonic angles: E = 1/2 k_angle (theta - theta0)^2."""
    unit_ba, unit_bc, length_ab, length_bc, theta = angle_geometry(
        bond_vectors, bond_lengths, angle_bond_map)
    # Torque (-dE/dtheta) per angle
    torque = k_angle * (theta0 - theta)

    # Directions in the plane of the angle, perpendicular to each bond
    normal = np.cross(unit_ba, unit_bc)
    dir_a = np.cross(normal, unit_ba)
    dir_c = np.cross(unit_bc, normal)
    dir_a /= np.linalg.norm(dir_a, axis=1)[:, None]
    dir_c /= np.linalg.norm(dir_c, axis=1)[:, None]
    dir_a *= (torque / length_ab)[:, None]
    dir_c *= (torque / length_bc)[:, None]

    energy = (0.5 * np.dot(k_angle, (theta - theta0)**2) if compute_energy
              else None)
    return ((angles[:, 1], angles[:, 0], angles[:, 2]),
            (dir_a + dir_c, -dir_a, -dir_c), energy)


def dihedral_term(bond_vectors, bond_lengths, dihedral_bond_map, dihedrals,
                  rb_coefficients, compute_energy):
    """Ryckaert-Bellemans dihedrals:
    E = 1/2 (C1 (1 + cos t) + C2 (1 - cos 2t) + C3 (1 + cos 3t)
             + C4 (1 - cos 4t))."""
    (vec_ab, vec_bc, vec_cd, length_ab, length_bc, length_cd, normal_abc,
     normal_bcd, theta, angle_b, angle_c) = dihedral_geometry(
        bond_vectors, bond_lengths, dihedral_bond_map)
    C1, C2, C3, C4 = rb_coefficients.T

    # Precalculate the big factor
    sin_terms = (C1 * np.sin(theta) - 2 * C2 * np.sin(2 * theta)
                 + 3 * C3 * np.sin(3 * theta) - 4 * C4 * np.sin(4 * theta))

    # Forces on the end atoms a and d
    force_a = (0.5 * sin_terms
               / (length_ab * np.sin(angle_b)))[:, None] * normal_abc
    force_d = (0.5 * sin_terms
               / (length_cd * np.sin(angle_c)))[:, None] * normal_bcd

    # Force on c using the midpoint of bc (half_bc) and the torque about it
    half_bc = 0.5 * vec_bc
    torque_c = -(np.cross(half_bc, force_d) + 0.5 * np.cross(vec_cd, force_d)
                 + 0.5 * np.cross(-vec_ab, force_a))
    force_c = 4 / (length_bc**2)[:, None] * np.cross(torque_c, half_bc)

    energy = None
    if compute_energy:
        energy = np.sum(0.5 * (C1 * (1 + np.cos(theta))
                               + C2 * (1 - np.cos(2 * theta))
                               + C3 * (1 + np.cos(3 * theta))
                               + C4 * (1 - np.cos(4 * theta))))
    # The force on b follows from Newton's third law
    return (tuple(dihedrals.T),
            (force_a, -force_a - force_c - force_d, force_c, force_d), energy)


@njit(cache=True)
def lj_kernel(positions, pair_i, pair_j, epsilon, sigma, r_cutoff,
              cutoff_energy, cutoff_force, box_length, compute_energy, forces):
    """Lennard-Jones: E = 4 eps ((sig/D)^12 - (sig/D)^6) for the pairs
    (pair_i, pair_j) within their cutoff, using the minimum image if
    box_length != 0. Adds the forces to `forces` and returns the energy (0
    unless compute_energy).
    Shifted force: cutoff_energy and cutoff_force are the energy and force at
    the cutoff (0 for an infinite cutoff). The force is F(D) - cutoff_force
    and the energy E(D) - cutoff_energy + (D - r_cutoff) cutoff_force, so
    both go to 0 at the cutoff."""
    inv_box_length = 1. / box_length if box_length != 0. else 0.
    energy = 0.
    for k in range(len(pair_i)):
        a, b = pair_i[k], pair_j[k]
        dx = positions[a, 0] - positions[b, 0]
        dy = positions[a, 1] - positions[b, 1]
        dz = positions[a, 2] - positions[b, 2]
        if box_length != 0.:
            # Minimum image (floor is much faster than % in compiled code)
            dx -= box_length * np.floor(dx * inv_box_length + 0.5)
            dy -= box_length * np.floor(dy * inv_box_length + 0.5)
            dz -= box_length * np.floor(dz * inv_box_length + 0.5)
        D = np.sqrt(dx * dx + dy * dy + dz * dz)
        if not D < r_cutoff[k]:
            continue

        sr6 = (sigma[k] / D)**6
        force_magnitude = (24 * epsilon[k] * (2 * sr6**2 - sr6) / D
                           - cutoff_force[k])
        f = force_magnitude / D
        forces[a, 0] += f * dx
        forces[a, 1] += f * dy
        forces[a, 2] += f * dz
        forces[b, 0] -= f * dx
        forces[b, 1] -= f * dy
        forces[b, 2] -= f * dz
        if compute_energy:
            energy += 4 * epsilon[k] * (sr6**2 - sr6) - cutoff_energy[k]
            # cutoff_force is 0 for an infinite cutoff (avoids inf * 0)
            if cutoff_force[k] != 0.:
                energy += (D - r_cutoff[k]) * cutoff_force[k]
    return energy


# ------------------------------- Force field ---------------------------------
class ForceField:
    """Precomputes everything constant about the interactions of a topology,
    and evaluates forces and energies at given positions."""

    def __init__(self, topology, box_length, lj_cutoff, skin=NEIGHBOUR_SKIN):
        top = self.topology = topology
        self.box_length = float(box_length)

        # LJ parameters per pair of atom types (unique (epsilon, sigma)),
        # using the Lorentz-Berthelot mixing rules
        types, atom_type = np.unique(
            np.column_stack((top.lj_epsilon, top.lj_sigma)), axis=0,
            return_inverse=True)
        self.atom_type = atom_type.ravel()
        type_epsilon, type_sigma = types[:, 0], types[:, 1]
        self.epsilon_table = np.sqrt(type_epsilon[:, None]
                                     * type_epsilon[None, :])
        self.sigma_table = 0.5 * (type_sigma[:, None] + type_sigma[None, :])
        # Cutoff per type pair; inf (only used by the tests) includes all
        # (minimum-image) pairs
        self.r_cutoff_table = (np.full_like(self.sigma_table, np.inf)
                               if np.isinf(lj_cutoff)
                               else lj_cutoff * self.sigma_table)
        interacting = (self.epsilon_table > 0) & (self.sigma_table > 0)

        # Shifted force: LJ energy and force at the cutoff of each type pair
        # (0 where there is nothing to shift, e.g. an infinite cutoff)
        cutoff = self.r_cutoff_table
        has_shift = interacting & np.isfinite(cutoff)
        cutoff = np.where(has_shift, cutoff, 1.)
        sr6_cutoff = np.where(has_shift, self.sigma_table / cutoff, 0.)**6
        self.cutoff_energy_table = (
            4 * self.epsilon_table * (sr6_cutoff**2 - sr6_cutoff))
        self.cutoff_force_table = (24 * self.epsilon_table
                                   * (2 * sr6_cutoff**2 - sr6_cutoff)
                                   / cutoff)

        max_cutoff = np.max(self.r_cutoff_table[interacting], initial=0.)

        self.neighbour_list = NeighbourList(top, box_length, max_cutoff, skin)
        self._set_lj_pairs()

        # Map angle and dihedral data to the indices of corresponding bonds
        self.angle_bond_map = bond_index_map(top.angles, top.bonds)
        self.dihedral_bond_map = bond_index_map(top.dihedrals, top.bonds)

        # Accumulated computation time (s), for the computation time chart
        self.timings = {"Bond vectors": 0., "Neighbour list": 0.,
                        "LJ forces": 0., "Bonded forces": 0.}

    def _set_lj_pairs(self):
        """Per-pair LJ parameters for the pairs in the neighbour list."""
        self.pair_i = self.neighbour_list.i
        self.pair_j = self.neighbour_list.j
        type_i = self.atom_type[self.pair_i]
        type_j = self.atom_type[self.pair_j]
        self.pair_epsilon = self.epsilon_table[type_i, type_j]
        self.pair_sigma = self.sigma_table[type_i, type_j]
        self.pair_cutoff = self.r_cutoff_table[type_i, type_j]
        self.pair_cutoff_energy = self.cutoff_energy_table[type_i, type_j]
        self.pair_cutoff_force = self.cutoff_force_table[type_i, type_j]

    def compute_forces(self, positions, compute_energy=False):
        """Forces at the positions, and the potential energy per term if
        compute_energy."""
        top = self.topology
        t0 = time.perf_counter()
        bond_vectors, bond_lengths = pair_vectors(
            positions, top.bonds[:, 0], top.bonds[:, 1], 0.)
        t1 = time.perf_counter()
        if self.neighbour_list.refresh(positions):
            self._set_lj_pairs()
        t2 = time.perf_counter()

        F = np.zeros((top.nr_atoms, 3))
        lj_energy = lj_kernel(
            positions, self.pair_i, self.pair_j, self.pair_epsilon,
            self.pair_sigma, self.pair_cutoff, self.pair_cutoff_energy,
            self.pair_cutoff_force, self.box_length, compute_energy, F)
        t3 = time.perf_counter()

        terms = {
            "Bond": bond_term(bond_vectors, bond_lengths, top.bonds,
                              top.k_bond, top.bond_r0, compute_energy),
            "Angle": angle_term(bond_vectors, bond_lengths,
                                self.angle_bond_map, top.angles, top.k_angle,
                                top.theta0, compute_energy),
            "Dihedral": dihedral_term(bond_vectors, bond_lengths,
                                      self.dihedral_bond_map, top.dihedrals,
                                      top.rb_coefficients, compute_energy),
        }
        # Scatter the bonded force contributions to the atoms in one pass
        indices = np.concatenate([idx for term in terms.values()
                                  for idx in term[0]])
        vectors = np.concatenate([vec for term in terms.values()
                                  for vec in term[1]]).reshape(-1, 3)
        for d in range(3):
            F[:, d] += np.bincount(indices, weights=vectors[:, d],
                                   minlength=top.nr_atoms)
        t4 = time.perf_counter()
        for label, dt in zip(self.timings, (t1 - t0, t2 - t1, t3 - t2,
                                            t4 - t3), strict=True):
            self.timings[label] += dt

        energies = None
        if compute_energy:
            energies = {name: lj_energy if name == "LJ" else terms[name][2]
                        for name in POTENTIAL_TERMS}
        return ForceEvaluation(F, bond_vectors, bond_lengths, energies)
