# Import necessary libraries and modules
from auxiliary import (scale_velocity, normalize_molecule_position, 
                       rotate_molecule, normalize_molecule_rotation)
from parse_functions import xyz_to_array, itp_to_data
from config import molar_volume, avogadro

import os
import numpy as np

# -------------- Creating a random initial velocity ---------------------------
def initializeVelocity(masses, temp):
    """
    Returns initial velocities in random directions, normalized to give the
    system a chosen temperature.
    """
    v = np.random.normal(0, 1, (len(masses), 3))
    return scale_velocity(v, masses, temp)



# ------------------------------- Creating a mixed initial configuration -------------------------------

def random_config(boxsize: float, random_grid_offset: float, 
                  molecules: list, weights: list):
    """
    Creates the xyz and itp data for a grid of given box size, where each 
    molecule in the molecules list is present with its associated weight in the 
    weights list. The random_grid_offset parameter slightly randomizes the 
    offset away from the exact grid points and should be set between 0 and 0.5.
    """
    weights = np.array(weights)
    molecules = np.array(molecules)

    # Identify which molecules to include in the system
    mask = weights > 0
    weights = weights[mask]
    molecules = molecules[mask]

    # Normalize weights
    weights = weights / np.sum(weights)
    # Identify molar volumes per molecule
    molar_volumes = np.array([1 / molar_volume[mol] for mol in molecules])
    # Compute the total nr of molecules that should be present in the system
    nr_molecules = int(
        boxsize**3 * 1e-21 * avogadro * np.dot(weights, molar_volumes)
    )
    # Find nr of molecules per type, and 
    nr_molecules_per_type = (weights * nr_molecules).astype(int)
    total_molecules = np.sum(nr_molecules_per_type)
    
    # Compute the number of molecules along a single dimension of the box
    N = int(np.ceil(total_molecules ** (1 / 3)))
    
    if total_molecules==0:
        raise ValueError(f"Boxsize={boxsize} is too small to accomodate "
                         f"required density. Please increase it.")

    # Generate file paths
    xyz_files = [os.path.join("molecule_data", f"{m}.xyz") for m in molecules]
    itp_files = [os.path.join("molecule_data", f"{m}.itp") for m in molecules]

    # Import xyz and itp data into arrays
    atoms_list, positions_list = zip(*(xyz_to_array(xyz) for xyz in xyz_files))
    atoms = np.array(atoms_list, dtype=object)  # Array of arrays (strings)
    positions = np.array(positions_list, dtype=object)  # Array of N x 3 arrays

    itp_data = np.array([itp_to_data(itp) for itp in itp_files], dtype=object)

    (
        bonds,
        kb,
        r0,
        angles,
        ktheta,
        theta0,
        epsilon,
        sigma,
        dihedrals,
        c1,
        c2,
        c3,
        c4,
        masses,
    ) = itp_data.T

    # Compute the number of atoms per molecule and cumulative indices
    nr_atoms_per_molecule = np.array([len(i) for i in atoms])
    repeated_atoms = np.repeat(nr_atoms_per_molecule, nr_molecules_per_type)
    cumulative_indices = np.concatenate([[0], np.cumsum(repeated_atoms)[:-1]])

    # Create all arrays for the xyz and itp data of the final big grid
    ATOMS = np.concatenate(
        [
            np.tile(row, count)
            for row, count in zip(atoms, nr_molecules_per_type)
        ]
    )

    KB = np.concatenate(np.repeat(kb, nr_molecules_per_type))
    R0 = np.concatenate(np.repeat(r0, nr_molecules_per_type))
    KTHETA = np.concatenate(np.repeat(ktheta, nr_molecules_per_type))
    THETA0 = np.concatenate(np.repeat(theta0, nr_molecules_per_type))
    EPSILON = np.concatenate(np.repeat(epsilon, nr_molecules_per_type))
    SIGMA = np.concatenate(np.repeat(sigma, nr_molecules_per_type))
    C1 = np.concatenate(np.repeat(c1, nr_molecules_per_type))
    C2 = np.concatenate(np.repeat(c2, nr_molecules_per_type))
    C3 = np.concatenate(np.repeat(c3, nr_molecules_per_type))
    C4 = np.concatenate(np.repeat(c4, nr_molecules_per_type))
    MASSES = np.concatenate(np.repeat(masses, nr_molecules_per_type))
    MOLEC_INDEX = np.repeat(np.arange(total_molecules), repeated_atoms)
    MOLEC_NAMES = np.repeat(np.repeat(molecules, nr_molecules_per_type),
                            repeated_atoms)

    # Shift indices in bonds, angles, and dihedrals.
    BONDS = np.concatenate(np.repeat(bonds,nr_molecules_per_type)+
                           cumulative_indices)
    ANGLES = np.concatenate(np.repeat(angles,nr_molecules_per_type)+
                            cumulative_indices)
    DIHEDRALS = np.concatenate(np.repeat(dihedrals,nr_molecules_per_type)+
                               cumulative_indices)
    
    
    # Create array with all possible molecule positions
    points = np.arange(N)
    all_positions = np.array(np.meshgrid(points, points, points)
                                                             ).T.reshape(-1, 3)
    # Choose the proper amount of positions from all_positions
    selections = np.random.choice(N**3, total_molecules, False)
    molecule_positions = all_positions[selections]
    molecule_positions = np.repeat(molecule_positions, repeated_atoms, axis=0)
    
    
    
    # randomize offset
    random_offset = np.random.uniform(
        low=-random_grid_offset, high=random_grid_offset, 
        size=(total_molecules, 3)
    )
    random_offset = np.repeat(random_offset, repeated_atoms, axis=0)
    
    # Normalize the object position, and the rotation afterwards.
    normalized_positions = np.array(
        [normalize_molecule_rotation(normalize_molecule_position(pos)) 
         for pos in positions], dtype=object
    )
    
    # Repeat positions to match the molecule counts  
    repeated_positions = np.repeat(normalized_positions, 
                                   nr_molecules_per_type,axis=0)
    
    parity = np.mod(np.sum(all_positions,axis=1),2)
    theta = np.where(parity==1,np.pi,0)
    phi = np.where(parity==1,np.pi/4,3*np.pi/4)
    
    theta = theta[selections]
    phi = phi[selections]
    rotated_positions = np.array([rotate_molecule(pos, 0, 0) 
                                  for i, pos in enumerate(repeated_positions)],dtype=object)
    
    # Concatenate all rotated positions into a single array
    POSITIONS = np.array(np.vstack(rotated_positions),dtype='f')

    # Shift the positions by the random_offset and grid positions.
    POSITIONS = POSITIONS + boxsize / N * (random_offset + molecule_positions)

    print(f"  Initialized {total_molecules} molecules ({len(ATOMS)} atoms),")
    print(f"  in a grid of {N}x{N}x{N} positions. Effective weights:")
    for i in range(len(molecules)):
        print(f"    {molecules[i]}: "
              f"{(nr_molecules_per_type[i] / total_molecules):.3f}")
    return (
        np.array(ATOMS,dtype=str),
        POSITIONS,
        BONDS,
        KB,
        R0,
        ANGLES,
        KTHETA,
        THETA0,
        EPSILON,
        SIGMA,
        DIHEDRALS,
        C1,
        C2,
        C3,
        C4,
        MASSES,
        MOLEC_INDEX,
        np.array(MOLEC_NAMES,dtype=str),
    )
