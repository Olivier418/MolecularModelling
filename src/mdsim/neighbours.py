"""Which atom pairs interact through Lennard-Jones forces.

A Verlet neighbour list stores every LJ pair closer than the largest cutoff
plus a skin. It only needs rebuilding once some atom has moved more than half
the skin, because until then no pair outside the list can have come within
the cutoff. LJ forces evaluated on the list are therefore exactly those of
an all-pairs computation.
"""
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree


def excluded_pair_keys(topology):
    """Sorted keys i * N + j (i < j) of the atom pairs that share a bond,
    angle or dihedral. These interact through bonded forces only."""
    n = topology.nr_atoms
    parts = [np.zeros(0, dtype=np.int64)]
    for group in (topology.bonds, topology.angles, topology.dihedrals):
        for a, b in combinations(range(group.shape[1]), 2):
            i, j = group[:, a], group[:, b]
            parts.append(np.minimum(i, j).astype(np.int64) * n
                         + np.maximum(i, j))
    return np.unique(np.concatenate(parts))


def _is_member(keys, sorted_keys):
    if len(sorted_keys) == 0:
        return np.zeros(len(keys), dtype=bool)
    pos = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return sorted_keys[pos] == keys


def filter_LJ_pairs(topology, i, j, excluded):
    """Keeps the candidate pairs (i < j) that experience LJ forces: both
    atoms have epsilon > 0, at least one has sigma > 0, and they don't share
    a bond, angle or dihedral."""
    t = topology
    keep = ((t.epsilon[i] > 0) & (t.epsilon[j] > 0)
            & ((t.sigma[i] > 0) | (t.sigma[j] > 0)))
    # Only pairs within one molecule can share a bond, angle or dihedral
    same = np.nonzero(keep & (t.molec_index[i] == t.molec_index[j]))[0]
    keys = i[same].astype(np.int64) * t.nr_atoms + j[same]
    keep[same[_is_member(keys, excluded)]] = False
    return i[keep], j[keep]


def lj_pairs(topology):
    """All pairs (i < j) that experience LJ forces, regardless of distance."""
    active = np.nonzero(topology.epsilon > 0)[0]
    a, b = np.triu_indices(len(active), k=1)
    return filter_LJ_pairs(topology, active[a], active[b],
                           excluded_pair_keys(topology))


class NeighbourList:
    """LJ pairs within `cutoff` + `skin` (minimum image), kept up to date by
    update(x). With boxsize=0 (no periodic boundaries), or a cutoff that
    reaches half the box, it simply holds all LJ pairs."""

    def __init__(self, topology, boxsize, cutoff, skin):
        self.topology = topology
        self.boxsize = boxsize
        self.skin = skin
        self.radius = cutoff + skin
        self.all_pairs = not self.radius < boxsize / 2
        self.active = np.nonzero(topology.epsilon > 0)[0]
        self.excluded = excluded_pair_keys(topology)
        self.x_ref = None
        self.nr_builds = 0
        self.i = self.j = np.zeros(0, dtype=int)

    def update(self, x):
        """Rebuilds the list if needed. Returns whether it was rebuilt."""
        if self.x_ref is not None:
            if self.all_pairs:
                return False
            displacement = x[self.active] - self.x_ref
            if np.max(np.sum(displacement**2, axis=1),
                      initial=0.) <= (self.skin / 2)**2:
                return False
        self._build(x)
        return True

    def _build(self, x):
        active = self.active
        if self.all_pairs:
            self.i, self.j = lj_pairs(self.topology)
        else:
            L = self.boxsize
            wrapped = np.mod(x[active], L)
            wrapped[wrapped >= L] = 0.  # np.mod can round up to exactly L
            a, b = cKDTree(wrapped, boxsize=L).query_pairs(
                self.radius, output_type='ndarray').T
            # active is sorted and a < b, so i < j
            self.i, self.j = filter_LJ_pairs(self.topology, active[a],
                                             active[b], self.excluded)
        self.x_ref = x[active].copy()
        self.nr_builds += 1
