"""Distances, angles, periodic boundaries and index bookkeeping."""
from dataclasses import dataclass

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


def _mapped_bonds(diff_bonds, D_bonds, mapping, k):
    """Vector and length of the k-th bond of each angle/dihedral, pointing
    along the angle/dihedral."""
    indices, signs = mapping
    return (signs[:, k][:, None] * diff_bonds[indices[:, k]],
            D_bonds[indices[:, k]])


def angle_geometry(diff_bonds, D_bonds, angle_mapping):
    """For angles a-b-c: the unit vectors A (b->a) and B (b->c), the bond
    lengths |ab| and |bc| and the angle theta between A and B."""
    diff_left, D_left = _mapped_bonds(diff_bonds, D_bonds, angle_mapping, 0)
    diff_right, D_right = _mapped_bonds(diff_bonds, D_bonds, angle_mapping, 1)

    A = -diff_left / D_left[:, None]
    B = diff_right / D_right[:, None]
    # Angle calculation (Clip to avoid numerical errors)
    theta = np.arccos(np.clip(np.sum(A * B, axis=1), -1.0, 1.0))
    return A, B, D_left, D_right, theta


@dataclass
class DihedralGeometry:
    """Geometry of dihedrals a-b-c-d. diff_* are bond vectors along the chain,
    ab/bc/cd their unit vectors, p1/p2 the unit normals of the planes abc and
    bcd, theta the signed dihedral angle and theta1/theta2 the bond angles
    at b and c."""
    diff_ab: np.ndarray
    diff_bc: np.ndarray
    diff_cd: np.ndarray
    D_ab: np.ndarray
    D_bc: np.ndarray
    D_cd: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    theta: np.ndarray
    theta1: np.ndarray
    theta2: np.ndarray


def dihedral_geometry(diff_bonds, D_bonds, dihedral_mapping):
    """Computes the DihedralGeometry of all dihedrals."""
    diff_ab, D_ab = _mapped_bonds(diff_bonds, D_bonds, dihedral_mapping, 0)
    diff_bc, D_bc = _mapped_bonds(diff_bonds, D_bonds, dihedral_mapping, 1)
    diff_cd, D_cd = _mapped_bonds(diff_bonds, D_bonds, dihedral_mapping, 2)

    # Normalized difference vectors between the three pairs
    ab = diff_ab / D_ab[:, None]
    bc = diff_bc / D_bc[:, None]
    cd = diff_cd / D_cd[:, None]

    # Construct and normalize direction vectors
    p1 = np.cross(-ab, bc)
    p2 = np.cross(cd, -bc)
    p1 /= np.linalg.norm(p1, axis=1, keepdims=True)
    p2 /= np.linalg.norm(p2, axis=1, keepdims=True)

    # Compute signed dihedral angle theta
    sign = np.sign(np.einsum('ij,ij->i', np.cross(p1, p2), bc))
    theta = sign * np.arccos(np.clip(np.einsum('ij,ij->i', p1, p2), -1., 1.))

    # Compute angles at b (theta1) and c (theta2)
    theta1 = np.arccos(np.clip(np.einsum('ij,ij->i', ab, -bc), -1.0, 1.0))
    theta2 = np.arccos(np.clip(np.einsum('ij,ij->i', bc, -cd), -1.0, 1.0))

    return DihedralGeometry(diff_ab, diff_bc, diff_cd, D_ab, D_bc, D_cd,
                            p1, p2, theta, theta1, theta2)


def wrap_molecules(x, molec_index, boxsize):
    """Shifts whole molecules by multiples of the box size so that the mean
    position of each molecule lies within the box. Molecules are never split
    across the boundary, so bonds stay intact for visualisation."""
    counts = np.bincount(molec_index)
    means = np.column_stack([np.bincount(molec_index, weights=x[:, d])
                             for d in range(3)]) / counts[:, None]
    offsets = np.floor_divide(means, boxsize)
    return x - offsets[molec_index] * boxsize
