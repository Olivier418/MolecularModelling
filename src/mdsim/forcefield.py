"""Force field setup (LJ pairs, mixing rules, cutoffs) and evaluation."""
import time
from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy.optimize import brentq

from .energy import angular_energy, bond_energy, dihedral_energy, LJ_energy
from .forces import net_force
from .geometry import map_to_bonds, pair_vectors


@dataclass
class ForceResult:
    """Forces at a configuration, with the bond and LJ vectors computed on
    the way (reused for energies and measurements)."""
    forces: np.ndarray
    diff_bonds: np.ndarray
    D_bonds: np.ndarray
    diff_LJ: np.ndarray
    D_LJ: np.ndarray


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

    def compute(self, x, cap_scalar=None):
        """Forces at positions x. If cap_scalar is set, LJ distances below
        cap_scalar * sigma are treated as that distance (soft start)."""
        t = self.topology
        t0 = time.perf_counter()
        diff_bonds, D_bonds = pair_vectors(x, t.bonds[:, 0], t.bonds[:, 1],
                                           0.)
        diff_LJ, D_LJ = pair_vectors(x, self.i_LJ, self.j_LJ, self.boxsize)
        t1 = time.perf_counter()
        cap = None if cap_scalar is None else cap_scalar * self.sigma_LJ
        F = net_force(diff_bonds, D_bonds, diff_LJ, D_LJ, t.bonds,
                      t.nr_atoms, self.angle_mapping, self.dihedral_mapping,
                      t.kb, t.r0, t.angles, t.ktheta, t.theta0,
                      self.epsilon_LJ, self.sigma_LJ, t.dihedrals, *t.C.T,
                      self.i_LJ, self.j_LJ, self.r_cutoff_LJ, cap)
        t2 = time.perf_counter()
        self.timings["Diff & Dist"] += t1 - t0
        self.timings["Forces"] += t2 - t1
        return ForceResult(F, diff_bonds, D_bonds, diff_LJ, D_LJ)

    def energies(self, result):
        """Potential energy per term at the configuration of a ForceResult."""
        t = self.topology
        return {
            "Bond": bond_energy(result.D_bonds, t.kb, t.r0),
            "Angular": angular_energy(result.diff_bonds, result.D_bonds,
                                      self.angle_mapping, t.ktheta, t.theta0),
            "LJ": LJ_energy(result.D_LJ, self.epsilon_LJ, self.sigma_LJ,
                            self.r_cutoff_LJ),
            "Dihedral": dihedral_energy(result.diff_bonds, result.D_bonds,
                                        self.dihedral_mapping, *t.C.T),
        }
