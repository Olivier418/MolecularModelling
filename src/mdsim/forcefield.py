"""Force field setup (LJ mixing rules and cutoffs) and evaluation of forces
and energies. Each bonded term computes its geometry once and returns its
force contributions per atom, plus its energy when asked. LJ forces are
evaluated by a compiled kernel over the pairs of a neighbour list."""
import time
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import brentq

from .geometry import (angle_geometry, dihedral_geometry, map_to_bonds,
                       pair_vectors)
from .neighbours import NeighbourList


@dataclass
class ForceResult:
    """Forces at a configuration, with the bond vectors computed on the way
    (reused for measurements). energies maps each potential energy term to
    its value if it was requested, else it is None. min_LJ_ratio is the
    smallest D / sigma over the LJ pairs in the neighbour list."""
    forces: np.ndarray
    diff_bonds: np.ndarray
    D_bonds: np.ndarray
    min_LJ_ratio: float
    energies: dict | None = None


# -------------------------------- LJ setup -----------------------------------
def lj_cutoffs(r_cutoff_scalar, f_threshold, epsilon_LJ, sigma_LJ, boxsize):
    """Calculates the r_cutoff per LJ pair. Assumes that exactly one of
    r_cutoff_scalar and f_threshold is None. If f_threshold is set, r_cutoff
    is chosen such that any forces below the threshold fall outside of the
    cutoff range."""
    N = len(epsilon_LJ)
    if r_cutoff_scalar is not None:
        if r_cutoff_scalar == 0:
            print("    Using r_cutoff=0.")
            return np.zeros(N)
        if np.isinf(r_cutoff_scalar):
            print("    Using r_cutoff=∞.")
            return np.full(N, np.inf)
        print(f"    Using r_cutoff={r_cutoff_scalar}*sigma.")
        return r_cutoff_scalar * sigma_LJ

    if np.isinf(f_threshold):
        print("    Using r_cutoff=0.")
        return np.zeros(N)
    if f_threshold == 0:
        print("    Using r_cutoff=∞.")
        return np.full(N, np.inf)

    # Function to find the root of to find when F = f_threshold
    def func(r, eps, sig):
        r6 = r**6
        r12 = r6**2
        F = -24 * eps * (2 * sig**12 / r12 - sig**6 / r6) / r
        return F - f_threshold

    r_values = np.linspace(0.1, boxsize, 100)
    r_cutoff = np.full(N, -1, dtype=float)
    # Loop over each unique (epsilon, sigma) pair to compute r_cutoff
    for eps, sig in np.unique(np.array([epsilon_LJ, sigma_LJ]), axis=1).T:
        # Identify intervals where the sign of the function changes, and
        # refine the zeroes within them using Brent's method
        f_values = func(r_values, eps, sig)
        changes = np.nonzero(f_values[:-1] * f_values[1:] < 0)[0]
        zeroes = [brentq(func, r_values[k], r_values[k + 1], args=(eps, sig))
                  for k in changes]
        # This is always the largest zero, in case two are found.
        match_mask = (epsilon_LJ == eps) & (sigma_LJ == sig)
        r_cutoff[match_mask] = max(zeroes) if zeroes else np.inf
    print(f"    Using r_cutoffs such that forces < {f_threshold} are "
          f"neglected.")
    return r_cutoff


# ------------------------------- Force terms ---------------------------------
# Each term returns (atom indices, force vectors, energy). The force on atom
# idx[k] gets vec[k] added; energy is None unless compute_energy is True.

def bond_term(diff_bonds, D_bonds, bonds, kb, r0, compute_energy):
    """Harmonic bonds: E = 1/2 kb (D - r0)^2."""
    F = (kb * (r0 - D_bonds) / D_bonds)[:, None] * diff_bonds
    energy = 0.5 * np.dot(kb, (D_bonds - r0)**2) if compute_energy else None
    return (bonds[:, 0], bonds[:, 1]), (F, -F), energy


def angle_term(diff_bonds, D_bonds, angle_mapping, angles, ktheta, theta0,
               compute_energy):
    """Harmonic angles: E = 1/2 ktheta (theta - theta0)^2."""
    A, B, D_left, D_right, theta = angle_geometry(diff_bonds, D_bonds,
                                                  angle_mapping)
    # Force magnitude per angle
    Fr = ktheta * (theta0 - theta)

    # Directions in the plane of the angle, perpendicular to each bond
    n = np.cross(A, B)
    vec1 = np.cross(n, A)
    vec2 = np.cross(B, n)
    vec1 /= np.linalg.norm(vec1, axis=1)[:, None]
    vec2 /= np.linalg.norm(vec2, axis=1)[:, None]
    vec1 *= (Fr / D_left)[:, None]
    vec2 *= (Fr / D_right)[:, None]

    energy = (0.5 * np.dot(ktheta, (theta - theta0)**2) if compute_energy
              else None)
    return ((angles[:, 1], angles[:, 0], angles[:, 2]),
            (vec1 + vec2, -vec1, -vec2), energy)


def dihedral_term(diff_bonds, D_bonds, dihedral_mapping, dihedrals, C,
                  compute_energy):
    """Ryckaert-Bellemans dihedrals:
    E = 1/2 (C1 (1 + cos t) + C2 (1 - cos 2t) + C3 (1 + cos 3t)
             + C4 (1 - cos 4t))."""
    g = dihedral_geometry(diff_bonds, D_bonds, dihedral_mapping)
    C1, C2, C3, C4 = C.T
    theta = g.theta

    # Precalculate the big factor
    sin_terms = (C1 * np.sin(theta) - 2 * C2 * np.sin(2 * theta)
                 + 3 * C3 * np.sin(3 * theta) - 4 * C4 * np.sin(4 * theta))

    # Calculate fa and fd
    fa = (0.5 * sin_terms / (g.D_ab * np.sin(g.theta1)))[:, None] * g.p1
    fd = (0.5 * sin_terms / (g.D_cd * np.sin(g.theta2)))[:, None] * g.p2

    # Force on c using intermediate o and t
    oc = 0.5 * g.diff_bc
    tc = -(np.cross(oc, fd) + 0.5 * np.cross(g.diff_cd, fd)
           + 0.5 * np.cross(-g.diff_ab, fa))
    fc = 4 / (g.D_bc**2)[:, None] * np.cross(tc, oc)

    energy = None
    if compute_energy:
        energy = np.sum(0.5 * (C1 * (1 + np.cos(theta))
                               + C2 * (1 - np.cos(2 * theta))
                               + C3 * (1 + np.cos(3 * theta))
                               + C4 * (1 - np.cos(4 * theta))))
    # The force on b follows from Newton's third law
    return (tuple(dihedrals.T), (fa, -fa - fc - fd, fc, fd), energy)


@njit(cache=True)
def LJ_kernel(x, i, j, epsilon, sigma, r_cutoff, boxsize, cap_scalar,
              compute_energy, F):
    """Lennard-Jones: E = 4 eps ((sig/D)^12 - (sig/D)^6) for the pairs (i, j)
    within their cutoff, using the minimum image if boxsize != 0. Adds the
    forces to F and returns the energy (0 unless compute_energy) and the
    smallest D / sigma. If cap_scalar > 0, the force at D < cap_scalar * sig
    is the force at that distance (used to push overlapping atoms apart
    gently)."""
    inv_boxsize = 1. / boxsize if boxsize != 0. else 0.
    energy = 0.
    min_ratio = np.inf
    for k in range(len(i)):
        a, b = i[k], j[k]
        dx = x[a, 0] - x[b, 0]
        dy = x[a, 1] - x[b, 1]
        dz = x[a, 2] - x[b, 2]
        if boxsize != 0.:
            # Minimum image (floor is much faster than % in compiled code)
            dx -= boxsize * np.floor(dx * inv_boxsize + 0.5)
            dy -= boxsize * np.floor(dy * inv_boxsize + 0.5)
            dz -= boxsize * np.floor(dz * inv_boxsize + 0.5)
        D = np.sqrt(dx * dx + dy * dy + dz * dz)
        min_ratio = min(min_ratio, D / sigma[k])
        if not D < r_cutoff[k]:
            continue

        D_force = max(D, cap_scalar * sigma[k])
        SIGovrD6 = (sigma[k] / D_force)**6
        F_scalar = 24 * epsilon[k] * (2 * SIGovrD6**2 - SIGovrD6) / D_force
        f = F_scalar / D_force
        F[a, 0] += f * dx
        F[a, 1] += f * dy
        F[a, 2] += f * dz
        F[b, 0] -= f * dx
        F[b, 1] -= f * dy
        F[b, 2] -= f * dz
        if compute_energy:
            SIGovrD6 = (sigma[k] / D)**6
            energy += 4 * epsilon[k] * (SIGovrD6**2 - SIGovrD6)
    return energy, min_ratio


# ------------------------------- Force field ---------------------------------
class ForceField:
    """Precomputes everything constant about the interactions of a topology,
    and evaluates forces and energies at given positions."""

    def __init__(self, topology, boxsize, r_cutoff_scalar=None,
                 f_threshold=None, skin=0.1):
        t = self.topology = topology
        self.boxsize = boxsize

        print("  Computing LJ parameters and cutoffs...")
        # LJ parameters per pair of atom types (unique (epsilon, sigma)),
        # using the Lorentz-Berthelot mixing rules
        types, atom_type = np.unique(np.column_stack((t.epsilon, t.sigma)),
                                     axis=0, return_inverse=True)
        self.atom_type = atom_type.ravel()
        eps, sig = types[:, 0], types[:, 1]
        self.epsilon_table = np.sqrt(eps[:, None] * eps[None, :])
        self.sigma_table = 0.5 * (sig[:, None] + sig[None, :])
        self.r_cutoff_table = lj_cutoffs(
            r_cutoff_scalar, f_threshold, self.epsilon_table.ravel(),
            self.sigma_table.ravel(), boxsize).reshape(len(types), len(types))
        interacting = (self.epsilon_table > 0) & (self.sigma_table > 0)
        max_cutoff = np.max(self.r_cutoff_table[interacting], initial=0.)
        self.has_LJ = bool(max_cutoff > 0)

        self.neighbours = NeighbourList(t, boxsize, max_cutoff, skin)
        self._set_LJ_pairs()

        # Map angle and dihedral data to the indices of corresponding bonds
        self.angle_mapping = map_to_bonds(t.angles, t.bonds)
        self.dihedral_mapping = map_to_bonds(t.dihedrals, t.bonds)

        # Accumulated computation time (s), for the computation time chart
        self.timings = {"Bond vectors": 0., "Neighbour list": 0.,
                        "LJ forces": 0., "Bonded forces": 0.}

    def _set_LJ_pairs(self):
        """Per-pair LJ parameters for the pairs in the neighbour list."""
        self.i_LJ, self.j_LJ = self.neighbours.i, self.neighbours.j
        ti, tj = self.atom_type[self.i_LJ], self.atom_type[self.j_LJ]
        self.epsilon_LJ = self.epsilon_table[ti, tj]
        self.sigma_LJ = self.sigma_table[ti, tj]
        self.r_cutoff_LJ = self.r_cutoff_table[ti, tj]

    def compute(self, x, compute_energy=False, cap_scalar=None):
        """Forces at positions x, and the potential energy per term if
        compute_energy. If cap_scalar is set, LJ distances below
        cap_scalar * sigma are treated as that distance (soft start)."""
        t = self.topology
        t0 = time.perf_counter()
        diff_bonds, D_bonds = pair_vectors(x, t.bonds[:, 0], t.bonds[:, 1],
                                           0.)
        t1 = time.perf_counter()
        if self.neighbours.update(x):
            self._set_LJ_pairs()
        t2 = time.perf_counter()

        F = np.zeros((t.nr_atoms, 3))
        LJ_energy, min_ratio = LJ_kernel(
            x, self.i_LJ, self.j_LJ, self.epsilon_LJ, self.sigma_LJ,
            self.r_cutoff_LJ, float(self.boxsize),
            -1. if cap_scalar is None else float(cap_scalar),
            compute_energy, F)
        t3 = time.perf_counter()

        terms = {
            "Bond": bond_term(diff_bonds, D_bonds, t.bonds, t.kb, t.r0,
                              compute_energy),
            "Angular": angle_term(diff_bonds, D_bonds, self.angle_mapping,
                                  t.angles, t.ktheta, t.theta0,
                                  compute_energy),
            "Dihedral": dihedral_term(diff_bonds, D_bonds,
                                      self.dihedral_mapping, t.dihedrals,
                                      t.C, compute_energy),
        }
        # Scatter the bonded force contributions to the atoms in one pass
        indices = np.concatenate([idx for term in terms.values()
                                  for idx in term[0]])
        vectors = np.concatenate([vec for term in terms.values()
                                  for vec in term[1]]).reshape(-1, 3)
        for d in range(3):
            F[:, d] += np.bincount(indices, weights=vectors[:, d],
                                   minlength=t.nr_atoms)
        t4 = time.perf_counter()
        for label, dt in zip(self.timings, (t1 - t0, t2 - t1, t3 - t2,
                                            t4 - t3)):
            self.timings[label] += dt

        energies = None
        if compute_energy:
            energies = {"Bond": terms["Bond"][2],
                        "Angular": terms["Angular"][2],
                        "LJ": LJ_energy,
                        "Dihedral": terms["Dihedral"][2]}
        return ForceResult(F, diff_bonds, D_bonds, min_ratio, energies)
