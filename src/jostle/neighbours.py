"""Which atom pairs interact through Lennard-Jones forces.

A Verlet neighbour list stores every LJ pair closer than the largest cutoff
plus a skin. It only needs rebuilding once some atom has moved more than half
the skin, because until then no pair outside the list can have come within
the cutoff. LJ forces evaluated on the list are therefore exactly those of
an all-pairs computation.
"""
from itertools import combinations

import numpy as np

from .geometry import periodic_pairs

# Extra distance (nm) beyond the largest cutoff kept in the neighbour list.
# Larger values mean fewer list rebuilds but more pairs per step. Results do
# not depend on it, only the speed does.
NEIGHBOUR_SKIN = 0.1


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


def _contains(keys, sorted_keys):
    if len(sorted_keys) == 0:
        return np.zeros(len(keys), dtype=bool)
    pos = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return sorted_keys[pos] == keys


def filter_lj_pairs(topology, i, j, excluded_keys):
    """Keeps the candidate pairs (i < j) that experience LJ forces: both
    atoms have epsilon > 0, at least one has sigma > 0, and they don't share
    a bond, angle or dihedral."""
    top = topology
    keep = ((top.lj_epsilon[i] > 0) & (top.lj_epsilon[j] > 0)
            & ((top.lj_sigma[i] > 0) | (top.lj_sigma[j] > 0)))
    # Only pairs within one molecule can share a bond, angle or dihedral
    same = np.nonzero(
        keep & (top.molecule_index[i] == top.molecule_index[j]))[0]
    keys = i[same].astype(np.int64) * top.nr_atoms + j[same]
    keep[same[_contains(keys, excluded_keys)]] = False
    return i[keep], j[keep]


def lj_pairs(topology):
    """All pairs (i < j) that experience LJ forces, regardless of distance."""
    return NeighbourList(topology, 0., np.inf, 0.)._candidates(None)


class NeighbourList:
    """LJ pairs within `cutoff` + `skin` (minimum image), kept up to date by
    refresh(positions). With box_length=0 (no periodic boundaries), or a
    cutoff that reaches half the box, it simply holds all LJ pairs."""

    def __init__(self, topology, box_length, cutoff, skin):
        self.topology = topology
        self.box_length = box_length
        self.skin = skin
        self.list_radius = cutoff + skin
        self.holds_all_pairs = not self.list_radius < box_length / 2
        self.lj_atoms = np.nonzero(topology.lj_epsilon > 0)[0]
        self.excluded_keys = excluded_pair_keys(topology)
        self.reference_positions = None
        self.nr_builds = 0
        self.i = self.j = np.zeros(0, dtype=int)

    def refresh(self, positions):
        """Rebuilds the list if needed. Returns whether it was rebuilt."""
        if self.reference_positions is not None:
            if self.holds_all_pairs:
                return False
            displacement = positions[self.lj_atoms] - self.reference_positions
            if np.max(np.sum(displacement**2, axis=1),
                      initial=0.) <= (self.skin / 2)**2:
                return False
        self._build(positions)
        return True

    def _candidates(self, positions):
        """The LJ pairs (i < j) among all pairs of active atoms, or among
        those within the radius at the given positions."""
        if self.holds_all_pairs:
            a, b = np.triu_indices(len(self.lj_atoms), k=1)
        else:
            a, b = periodic_pairs(positions[self.lj_atoms], self.box_length,
                                  self.list_radius)
        # lj_atoms is sorted and a < b, so i < j
        return filter_lj_pairs(self.topology, self.lj_atoms[a],
                               self.lj_atoms[b], self.excluded_keys)

    def _build(self, positions):
        self.i, self.j = self._candidates(positions)
        self.reference_positions = positions[self.lj_atoms].copy()
        self.nr_builds += 1
