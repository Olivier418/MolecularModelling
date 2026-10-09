"""Distances, periodic boundaries and index bookkeeping."""
import numpy as np
from numba import njit


@njit(cache=True)
def pair_vectors(positions, i, j, boxsize):
    """Efficiently computes pairwise differences x_i - x_j and distances on
    the provided pairs in i,j. Apply the minimum image convention if
    boxsize!=0."""
    diff = positions[i] - positions[j]

    # Apply periodic boundary conditions if needed
    if boxsize != 0.:
        half = boxsize / 2
        diff = (diff + half) % boxsize - half

    # Compute distances
    distances = np.sqrt(np.sum(diff * diff, axis=1))
    return diff, distances


def map_to_bonds(array, bonds):
    """Maps each consecutive atom pair in the rows of array (typically angles
    or dihedrals) to its index in bonds, so angles and dihedrals can reuse the
    bond vectors during integration. Also returns a sign per entry, used to
    flip the bond vector to the direction along the angle/dihedral."""
    # Create a dictionary for fast lookup
    bond_dict = {tuple(bond): k for k, bond in enumerate(bonds.tolist())}

    # Determine the number of bonds in array (either 2 or 3)
    bond_count = array.shape[1] - 1

    # Construct result arrays. Sign is used to flip directions later on
    indices = np.empty((array.shape[0], bond_count), dtype=int)
    signs = np.empty((array.shape[0], bond_count), dtype=int)

    for k, row in enumerate(array.tolist()):
        for m in range(bond_count):
            a, b = row[m], row[m + 1]
            indices[k, m] = bond_dict[(min(a, b), max(a, b))]
            signs[k, m] = -1 if a < b else 1

    return indices, signs


def wrap_molecules(x, molec_index, boxsize):
    """Shifts whole molecules by multiples of the box size so that the mean
    position of each molecule lies within the box. Molecules are never split
    across the boundary, so bonds stay intact for visualisation."""
    counts = np.bincount(molec_index)
    means = np.column_stack([np.bincount(molec_index, weights=x[:, d])
                             for d in range(3)]) / counts[:, None]
    offsets = np.floor_divide(means, boxsize)
    return x - offsets[molec_index] * boxsize
