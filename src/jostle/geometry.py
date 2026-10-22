"""Distances, angles, periodic boundaries and index bookkeeping."""
import numpy as np
from numba import njit
from scipy.spatial import cKDTree


@njit(cache=True)
def pair_vectors(positions, i, j, box_length):
    """Efficiently computes pairwise differences x_i - x_j and distances on
    the provided pairs in i,j. Apply the minimum image convention if
    box_length != 0 (0 means no periodic boundaries)."""
    diff = positions[i] - positions[j]

    # Apply periodic boundary conditions if needed: map into [-L/2, L/2)
    if box_length != 0.:
        diff -= box_length * np.floor(diff * (1. / box_length) + 0.5)

    # Compute distances
    distances = np.sqrt(np.sum(diff * diff, axis=1))
    return diff, distances


def bond_index_map(array, bonds):
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


def _mapped_bonds(bond_vectors, bond_lengths, mapping, k):
    """Vector and length of the k-th bond of each angle/dihedral, pointing
    along the angle/dihedral."""
    indices, signs = mapping
    return (signs[:, k][:, None] * bond_vectors[indices[:, k]],
            bond_lengths[indices[:, k]])


def angle_geometry(bond_vectors, bond_lengths, angle_bond_map):
    """For angles a-b-c: the unit vectors unit_ba (b->a) and unit_bc (b->c),
    the bond lengths |ab| and |bc| and the angle theta between the unit
    vectors."""
    vec_ab, length_ab = _mapped_bonds(bond_vectors, bond_lengths,
                                      angle_bond_map, 0)
    vec_bc, length_bc = _mapped_bonds(bond_vectors, bond_lengths,
                                      angle_bond_map, 1)

    unit_ba = -vec_ab / length_ab[:, None]
    unit_bc = vec_bc / length_bc[:, None]
    # Angle calculation (Clip to avoid numerical errors)
    theta = np.arccos(np.clip(np.sum(unit_ba * unit_bc, axis=1), -1.0, 1.0))
    return unit_ba, unit_bc, length_ab, length_bc, theta


def dihedral_geometry(bond_vectors, bond_lengths, dihedral_bond_map):
    """For dihedrals a-b-c-d: the bond vectors vec_ab, vec_bc, vec_cd along
    the chain and their lengths, the unit normals normal_abc and normal_bcd
    of the planes abc and bcd, the signed dihedral angle theta and the bond
    angles angle_b and angle_c at b and c."""
    vec_ab, length_ab = _mapped_bonds(bond_vectors, bond_lengths,
                                      dihedral_bond_map, 0)
    vec_bc, length_bc = _mapped_bonds(bond_vectors, bond_lengths,
                                      dihedral_bond_map, 1)
    vec_cd, length_cd = _mapped_bonds(bond_vectors, bond_lengths,
                                      dihedral_bond_map, 2)

    # Normalized difference vectors between the three pairs
    ab = vec_ab / length_ab[:, None]
    bc = vec_bc / length_bc[:, None]
    cd = vec_cd / length_cd[:, None]

    # Construct and normalize direction vectors
    normal_abc = np.cross(-ab, bc)
    normal_bcd = np.cross(cd, -bc)
    normal_abc /= np.linalg.norm(normal_abc, axis=1, keepdims=True)
    normal_bcd /= np.linalg.norm(normal_bcd, axis=1, keepdims=True)

    # Compute signed dihedral angle theta
    sign = np.sign(np.einsum('ij,ij->i', np.cross(normal_abc, normal_bcd),
                             bc))
    theta = sign * np.arccos(np.clip(
        np.einsum('ij,ij->i', normal_abc, normal_bcd), -1., 1.))

    # Compute angles at b (angle_b) and c (angle_c)
    angle_b = np.arccos(np.clip(np.einsum('ij,ij->i', ab, -bc), -1.0, 1.0))
    angle_c = np.arccos(np.clip(np.einsum('ij,ij->i', bc, -cd), -1.0, 1.0))

    return (vec_ab, vec_bc, vec_cd, length_ab, length_bc, length_cd,
            normal_abc, normal_bcd, theta, angle_b, angle_c)


def wrap_molecules(positions, molecule_index, box_length):
    """Shifts whole molecules by multiples of the box length so that the mean
    position of each molecule lies within the box. Molecules are never split
    across the boundary, so bonds stay intact for visualisation."""
    counts = np.bincount(molecule_index)
    means = np.column_stack([np.bincount(molecule_index,
                                         weights=positions[:, d])
                             for d in range(3)]) / counts[:, None]
    offsets = np.floor_divide(means, box_length)
    return positions - offsets[molecule_index] * box_length


def periodic_pairs(positions, box_length, radius):
    """Index pairs (a, b), a < b, of the points closer than radius, using the
    minimum image."""
    # cKDTree needs coordinates in [0, L); np.mod can round up to exactly L
    wrapped = np.mod(positions, box_length)
    wrapped[wrapped >= box_length] = 0.
    pairs = cKDTree(wrapped, boxsize=box_length).query_pairs(
        radius, output_type='ndarray')
    return pairs[:, 0], pairs[:, 1]
