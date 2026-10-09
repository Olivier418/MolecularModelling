"""Force field setup (LJ pairs, mixing rules, cutoffs) and evaluation of
forces and energies. Each term computes its geometry once and returns its
force contributions per atom, plus its energy when asked."""
import time
from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy.optimize import brentq

from .geometry import (angle_geometry, dihedral_geometry, map_to_bonds,
                       pair_vectors)


@dataclass
class ForceResult:
    """Forces at a configuration, with the bond and LJ vectors computed on
    the way (reused for measurements). energies maps each potential energy
    term to its value if it was requested, else it is None."""
    forces: np.ndarray
    diff_bonds: np.ndarray
    D_bonds: np.ndarray
    diff_LJ: np.ndarray
    D_LJ: np.ndarray
    energies: dict | None = None


# -------------------------------- LJ setup -----------------------------------
def lj_pairs(topology):
    """Computes the indices of pairs that experience LJ forces in the system.
    This excludes 0-epsilon or 0-sigma atoms or atoms that interact through
    bond/angular/dihedral forces."""
    t = topology
    n = t.nr_atoms
    # Create a mask that excludes the correct intramolecular pairs
    molec_mask = np.ones((n, n), dtype=bool)
    # Exclude the bonds (needed for diatomics, which have no angles)
    molec_mask[t.bonds[:, 0], t.bonds[:, 1]] = False
    molec_mask[t.bonds[:, 1], t.bonds[:, 0]] = False
    # Exclude atoms within the same angle or dihedral
    for group in (t.angles, t.dihedrals):
        pairs = np.array(list(permutations(range(group.shape[1]), 2)))
        molec_mask[group[:, pairs[:, 0]], group[:, pairs[:, 1]]] = False

    # Exclude pairs that have zero sigma or zero epsilon
    sigma_bool = t.sigma > 0
    eps_bool = t.epsilon > 0
    LJ_mask = (np.logical_or.outer(sigma_bool, sigma_bool)
               & np.logical_and.outer(eps_bool, eps_bool) & molec_mask)

    # Find the indices of the resulting mask that are above the main diagonal.
    i, j = np.triu_indices(n, k=1)
    keep = LJ_mask[i, j]
    return i[keep], j[keep]


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


def LJ_term(diff_LJ, D_LJ, epsilon_LJ, sigma_LJ, i_LJ, j_LJ, r_cutoff_LJ,
            cap, compute_energy):
    """Lennard-Jones: E = 4 eps ((sig/D)^12 - (sig/D)^6) for pairs within
    their cutoff. If cap is given, the force at D < cap is the force at cap
    (used to push overlapping atoms apart gently)."""
    # Extract relevant data for pairs within the cutoff
    valid = D_LJ < r_cutoff_LJ
    D = D_LJ[valid]
    EPS = epsilon_LJ[valid]
    SIG = sigma_LJ[valid]

    D_force = D if cap is None else np.maximum(D, cap[valid])
    SIGovrD6 = (SIG / D_force)**6
    F_scalar = 24 * EPS * (2 * SIGovrD6**2 - SIGovrD6) / D_force
    F = (F_scalar / D_force)[:, None] * diff_LJ[valid]

    energy = None
    if compute_energy:
        SIGovrD6 = (SIG / D)**6
        energy = np.sum(4 * EPS * (SIGovrD6**2 - SIGovrD6))
    return (i_LJ[valid], j_LJ[valid]), (F, -F), energy


# ------------------------------- Force field ---------------------------------
class ForceField:
    """Precomputes everything constant about the interactions of a topology,
    and evaluates forces and energies at given positions."""

    def __init__(self, topology, boxsize, r_cutoff_scalar=None,
                 f_threshold=None):
        t = self.topology = topology
        self.boxsize = boxsize

        print("  Computing LJ indices...")
        self.i_LJ, self.j_LJ = lj_pairs(t)

        print("  Computing epsilon and sigma...")
        # Lorentz-Berthelot mixing rules
        self.sigma_LJ = 0.5 * (t.sigma[self.i_LJ] + t.sigma[self.j_LJ])
        self.epsilon_LJ = np.sqrt(t.epsilon[self.i_LJ] * t.epsilon[self.j_LJ])

        print("  Computing r_cutoff...")
        self.r_cutoff_LJ = lj_cutoffs(r_cutoff_scalar, f_threshold,
                                      self.epsilon_LJ, self.sigma_LJ, boxsize)

        # Map angle and dihedral data to the indices of corresponding bonds
        self.angle_mapping = map_to_bonds(t.angles, t.bonds)
        self.dihedral_mapping = map_to_bonds(t.dihedrals, t.bonds)

        # Accumulated computation time (s), for the computation time chart
        self.timings = {"Diff & Dist": 0., "Forces": 0.}

    def compute(self, x, compute_energy=False, cap_scalar=None):
        """Forces at positions x, and the potential energy per term if
        compute_energy. If cap_scalar is set, LJ distances below
        cap_scalar * sigma are treated as that distance (soft start)."""
        t = self.topology
        t0 = time.perf_counter()
        diff_bonds, D_bonds = pair_vectors(x, t.bonds[:, 0], t.bonds[:, 1],
                                           0.)
        diff_LJ, D_LJ = pair_vectors(x, self.i_LJ, self.j_LJ, self.boxsize)
        t1 = time.perf_counter()

        cap = None if cap_scalar is None else cap_scalar * self.sigma_LJ
        terms = {
            "Bond": bond_term(diff_bonds, D_bonds, t.bonds, t.kb, t.r0,
                              compute_energy),
            "Angular": angle_term(diff_bonds, D_bonds, self.angle_mapping,
                                  t.angles, t.ktheta, t.theta0,
                                  compute_energy),
            "LJ": LJ_term(diff_LJ, D_LJ, self.epsilon_LJ, self.sigma_LJ,
                          self.i_LJ, self.j_LJ, self.r_cutoff_LJ, cap,
                          compute_energy),
            "Dihedral": dihedral_term(diff_bonds, D_bonds,
                                      self.dihedral_mapping, t.dihedrals,
                                      t.C, compute_energy),
        }

        # Scatter the force contributions to the atoms
        F = np.zeros((t.nr_atoms, 3))
        for indices, vectors, _ in terms.values():
            for idx, vec in zip(indices, vectors):
                np.add.at(F, idx, vec)
        t2 = time.perf_counter()
        self.timings["Diff & Dist"] += t1 - t0
        self.timings["Forces"] += t2 - t1

        energies = None
        if compute_energy:
            energies = {name: energy for name, (_, _, energy) in terms.items()}
        return ForceResult(F, diff_bonds, D_bonds, diff_LJ, D_LJ, energies)
