# Import necessary libraries and modules
import numpy as np
from scipy.optimize import brentq
from scipy.optimize import differential_evolution
import os
from numba import njit, int32, float64
from itertools import permutations

from config import kboltz

# -------------------- Initialization help Functions --------------------------
def find_bond_indices(array, bonds):
    """Creates indices to map from array (typically angles or dihedrals) to
    bonds, for easier indexing during the integration process."""
    # Create a dictionary for fast lookup
    bond_dict = {tuple(bonds[i]): i for i in range(len(bonds))}
    
    # Determine the number of bonds in array (either 2 or 3)
    bond_count = array.shape[1] - 1

    # Construct result arrays. Sign is used to flip directions later on
    result = np.empty((array.shape[0], bond_count), dtype=int)
    signs = np.empty((array.shape[0], bond_count), dtype=int)

    # Fill the result and sign arrays
    for i in range(array.shape[0]):
        for j in range(bond_count):
            a, b = array[i, j], array[i, j + 1]
            sorted_bond = (min(a, b), max(a, b))  # Ensure correct ordering
            result[i, j] = bond_dict[sorted_bond]  # Get bond index
            signs[i, j] = -1 if (a<b) else 1 # Store sign information

    return result, signs

def cutoff_calculator(r_cutoff_scalar, f_threshold, epsilon_LJ, sigma_LJ, 
                      boxsize):
    """Calculates the 1-dimensional array of r_cutoff values to use during
    the LJ forces in the integration process. Assumes that exactly one of 
    r_cutoff_scalar and f_threshold is set to none. If f_threshold is set to
    a numerical value, r_cutoff will be computed such that any forces below
    the threshold will fall outside of the cutoff range."""
    N=len(epsilon_LJ)
    if r_cutoff_scalar is not None:
        if r_cutoff_scalar == 0:
            r_cutoff = np.zeros(N)
            print("    Using r_cutoff=0.")
        elif np.isinf(r_cutoff_scalar):
            r_cutoff = np.full(N, np.inf, dtype=float)
            print("    Using r_cutoff=∞.")
        else:
            r_cutoff = r_cutoff_scalar * sigma_LJ
            print(f"    Using r_cutoff={r_cutoff_scalar}*sigma.")
    elif f_threshold is not None:
        if np.isinf(f_threshold):
            r_cutoff = np.zeros(N)
            print("    Using r_cutoff=0.")
        elif f_threshold == 0:
            r_cutoff = np.full(N, np.inf, dtype=float)
            print("    Using r_cutoff=∞.")
        else:      
            #Identify unique epsilon and sigma in the system
            uniq_ep,uniq_sig = np.unique(np.array([epsilon_LJ,sigma_LJ]),axis=1)
            # Initialize the r_cutoff array with the same shape as epsilon and sigma
            r_cutoff = np.full(N, -1, dtype=float)
            
            # Function to find the root of to find when F = f_threshold
            def func(r, eps, sig):
                r6 = r**6
                r12 = r6**2
                F = -24 * eps * (2 * sig**12 / r12 - sig**6 / r6) / r
                return F - f_threshold
            
            # Generate r values within the specified range
            r_values = np.linspace(0.1, boxsize, 100)
            
            # Loop over each unique (epsilon, sigma) pair to compute r_cutoff
            for eps, sig in zip(uniq_ep,uniq_sig):
                # Calculate function values for all r values
                f_values = func(r_values[:, None], eps, sig)
            
                # Identify intervals where the sign of the function changes
                zero_intervals = [
                    (r_values[i], r_values[i + 1])
                    for i in range(len(r_values) - 1)
                    if f_values[i] * f_values[i + 1] < 0
                ]
            
                # Refine the zeroes within the identified intervals using Brent's method
                zeroes = []
                for interval in zero_intervals:
                    try:
                        root = brentq(func, *interval, args=(eps, sig))
                        zeroes.append(root)
                    except ValueError:
                        pass  # Skip intervals where Brent's method fails
            
                # Compute r_cutoff for this (epsilon, sigma) pair.
                # This is alwasy the largest zero, in case two are found.
                r_cutoff_value = max(zeroes) if zeroes else np.inf
            
                # Assign the value to the corresponding locations in the r_cutoff array
                match_mask = (epsilon_LJ == eps) & (sigma_LJ == sig)
                r_cutoff[match_mask] = r_cutoff_value
            print(f"    Using r_cutoffs such that forces < {f_threshold} are neglected.")
            
    return(r_cutoff)

def compute_LJ_indices(nr_atoms, bonds, angles, dihedrals, epsilon, sigma):
    """Computes the indices of pairs that experience LJ forces in the system.
    This excludes 0-epsilon or 0-sigma atoms or atoms that interact through
    bond/angular/dihedral forces."""
    # Create a mask that excludes the correct intramolecular pairs
    molec_mask = np.ones((nr_atoms, nr_atoms), dtype=bool)  # Start with all True
    # Exclude the bonds (needed for diatomics, which have no angles)
    molec_mask[bonds[:, 0], bonds[:, 1]] = False
    molec_mask[bonds[:, 1], bonds[:, 0]] = False
    # Exclude the dihedrals
    dihedral_pairs = np.array(list(permutations([0, 1, 2, 3], 2)))
    molec_mask[dihedrals[:, dihedral_pairs[:, 0]], 
               dihedrals[:, dihedral_pairs[:, 1]]] = False
    # Exclude then angles
    angle_pairs = np.array(list(permutations([0, 1, 2], 2)))
    molec_mask[angles[:, angle_pairs[:, 0]], 
               angles[:, angle_pairs[:, 1]]] = False
    
    # Exclude pairs that have zero sigma or zero epsilon
    sigma_bool = sigma>0
    sigma_mask = np.logical_or.outer(sigma_bool,sigma_bool)
    
    eps_bool = epsilon>0
    eps_mask = np.logical_and.outer(eps_bool,eps_bool)
     
    LJ_mask = sigma_mask & eps_mask & molec_mask
    
    # Find the indices of the resulting mask that are above the main diagonal.
    i,j = np.triu_indices(nr_atoms, k=1) 
    LJ_indices = LJ_mask[i,j]
    i_LJ, j_LJ = i[LJ_indices], j[LJ_indices]    
    
    return(i_LJ,j_LJ)

# ---------------------- Integration help Functions ---------------------------
@njit
def diff_and_dist_arr(positions, i, j, boxsize):
    """Efficiently computes pairwise differences and distances on the provided
    pairs in i,j. Apply modular reduction if boxsize!=0."""
    diff = positions[i] - positions[j]
    
    # Apply periodic boundary conditions if needed
    if boxsize!=0.:
        bnds = boxsize / 2
        diff = (diff + bnds) % boxsize - bnds 

    # Compute distances
    distances = np.sqrt(np.sum(diff * diff, axis=1))
    return diff, distances

def velocity_scale(v_i, masses, T_0):
    """
    Factor by which to scale velocities to reach temperature T_0.
    """
    KE = 0.5 * np.dot(masses, np.sum(v_i**2, axis=1))
    Nf = 3 * masses.shape[0]
    return np.sqrt(T_0 * Nf * kboltz / (2 * KE))

def scale_velocity(v_i, masses, T_0):
    """
    Scale velocities to ensure temperature control if required.
    """
    return v_i * velocity_scale(v_i, masses, T_0)

# ------------------------- position manipulation -----------------------------
def normalize_molecule_position(array):
    """
    Normalizes an xyz array such that the mean position lies at (0,0,0)
    """
    return array - np.mean(array, axis=0)


def rotate_molecule(array, theta, phi):
    """Rotates an xyz array around the z axis by theta, and then around the y 
    axis by phi. both theta and phi take values in [0,2pi)."""
    # Define the rotation matrices
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])
    K = array @ (Rz @ Ry).T
    # Apply the combined rotation matrix to all points
    return K

def normalize_molecule_rotation(array):
    """Rotates an array such that it is contained in the smallest box possible
    (where 'smallest' means that the largest dimension of the box is minimized)
    """
    # The target function to be minimized:
    def target_min(angles):
        theta, phi = angles
        X = rotate_molecule(array, theta, phi)
        return np.max(np.max(X, axis=0) - np.min(X, axis=0))
    
    bounds = [(-np.pi, np.pi), (-np.pi, np.pi)]  # Angles range from -π to π
    result = differential_evolution(target_min, bounds)
    theta_opt2, phi_opt2 = result.x
    
    # Rotate the molecule in the optimal way
    return rotate_molecule(array, theta_opt2, phi_opt2)


def project_to_box(X, molec_index, boxsize):
    """
    Project an array of atom positions into a bounding box by ensuring the mean 
    position of each molecule lies within the box.
    """    
    # Compute mean positions of each molecule
    uniq_molecules, inv_indices = np.unique(molec_index, return_inverse=True)
    molecule_means = np.zeros((len(uniq_molecules), X.shape[1]))
    np.add.at(molecule_means, inv_indices, X)
    counts = np.bincount(inv_indices, minlength=len(uniq_molecules))
    molecule_means /= counts[:, None]
    
    # Compute offsets for each molecule
    offsets = np.floor_divide(molecule_means, boxsize)
    
    # Apply offsets to all atoms in each molecule    
    return (X - offsets[inv_indices] * boxsize)

# ------------------------ Auxiliary functions -------------------------------
def validate_parameters(molecules, weights, boxsize, random_grid_offset, dt, 
                        endTime, save_steps, method, initial_temp,
                        temp_control, f_threshold, r_cutoff_scalar, VMD_vis, 
                        energy_graph, temp_graph, phase_space_diagram, RDF_vis, 
                        RDF_thin, RDF_dr, bonds_of_interest,pairs_of_interest,
                        r_threshold_scalar):
    """Validates all given parameters to prevent any weird user inputs."""
    # Assert all entries in list weights are >=0, and sum(weights)>0
    if any(w < 0 for w in weights):
        raise ValueError("All weights must be >= 0.")
    if sum(weights) <= 0:
        raise ValueError("The sum of weights must be > 0.")
    
    # Assert the xyz and itp data exists
    for mol in molecules:
        if not os.path.exists(f"molecule_data/{mol}.xyz"):
            raise FileNotFoundError(f"File molecule_data/{mol}.xyz"
                                    f" does not exist.")
        if not os.path.exists(f"molecule_data/{mol}.itp"):
            raise FileNotFoundError(f"File molecule_data/{mol}.itp"
                                    f" does not exist.")
    
    # Assert integer >=1 : save_steps
    if not (isinstance(save_steps, int) and save_steps >= 1):
        raise ValueError("save_steps must be an integer >= 1.")
    
    # Assert >0 : d, dt, endTime, initial_temp, r_cutoff_scalar
    if not (boxsize > 0):
        raise ValueError("boxsize must be > 0.")
    if not (dt > 0):
        raise ValueError("dt must be > 0.")
    if not (endTime > 0):
        raise ValueError("endTime must be > 0.")
    if not (initial_temp > 0):
        raise ValueError("initial_temp must be > 0.")
        
    # assert molecules and weights match in length
    if len(molecules) != len(weights):
        raise ValueError("Number of molecules must match weights list length.")

    
    # Assert >= 0 : f_threshold_scalar, random_grid_offset
    if r_cutoff_scalar is not None and not (r_cutoff_scalar >= 0):
        raise ValueError("r_cutoff_scalar must be >= 0.")
    if f_threshold is not None and f_threshold < 0:
        raise ValueError("f_threshold_scalar must be >= 0.")
        
    if r_threshold_scalar is not None and r_threshold_scalar<=0:
        raise ValueError("r_threshold_scalar must be > 0.")
        
    if RDF_vis:
        if RDF_thin >1 or RDF_thin<0:
            raise ValueError("RDF_thin must be between 0 and 1.")
        if RDF_dr<=0:
            raise ValueError("RDF_dr must be >0.")
        
    # Assert -0.5 < random_grid_offset < 0.5
    if random_grid_offset>0.5 or random_grid_offset<-0.5:
        raise ValueError("random_grid_offset must be between -0.5 and 0.5.")
    
    # Assert bool : temp_control, VMD_vis, energy_graph, temp_graph, 
    #               phase_space_diagram, RDF_vis
    for var, name in zip([temp_control, VMD_vis, energy_graph, temp_graph, 
                          phase_space_diagram, RDF_vis],
                         ["temp_control", "VMD_vis", "energy_graph", 
                          "temp_graph", "phase_space_diagram", "RDF_vis"]):
        if not isinstance(var, bool):
            raise ValueError(f"{name} must be a boolean.")
    
    # if phase_space_diagram: Assert len(bonds_of_interest) >= 1
    if phase_space_diagram and (not isinstance(bonds_of_interest, list) or 
                                len(bonds_of_interest) < 1):
        raise ValueError("bonds_of_interest must be a list with at least one "
                         "element when phase_space_diagram is True.")
    
    # if RDF_vis: Assert len(pairs_of_interest) >= 1
    if RDF_vis and (not isinstance(pairs_of_interest, list) or 
                    len(pairs_of_interest) < 1):
        raise ValueError("pairs_of_interest must be a list with at least one "
                         "element when RDF_vis is True.")
    
    # Ensure only one of f_threshold_scalar or r_cutoff_scalar is specified
    if not ((f_threshold is None) ^ (r_cutoff_scalar is None)):
        raise ValueError("Exactly one of f_threshold or r_cutoff_scalar must "
                         "be specified.")

def updt(total, progress, extra=""):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    
    block = int(round(barLength * progress))
    text = "\r  [{}] {:.0f}% {}{}".format(
        "#" * block + "-" * (barLength - block),
        round(progress * 100, 0), extra, status)
    print(text,end='')