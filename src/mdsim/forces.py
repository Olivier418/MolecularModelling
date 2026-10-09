# Import necessary libraries and modules
import numpy as np

# ------------------------------- Force Functions -----------------------------
def bond_force(diff_bonds, D_bonds, bonds, kb, r0, nr_atoms):
    """
    Calculates bond forces.
    """   
    # Compute the force magnitude for all bonds
    F_magnitude = kb * (r0 - D_bonds) 
    
    # Compute the force vectors
    F = (F_magnitude/ D_bonds)[:, None] * diff_bonds

    # Initialize the bond matrix to accumulate forces
    bond_mat = np.zeros((nr_atoms,3))
    
    # Scatter the forces to the respective atoms
    np.add.at(bond_mat, bonds[:, 0], F)  # Add force to atom A
    np.subtract.at(bond_mat, bonds[:, 1], F)  # Subtract force from atom B

    return bond_mat


def ang_force(diff_bonds, D_bonds, angle_mapping, angles, ktheta, theta0, 
              nr_atoms):
    """Calculates angular forces."""
    # Retrieve the differences and distances from the arrays using the mapping
    angle_indices, angle_signs = angle_mapping
    
    D_left = D_bonds[angle_indices[:,0]]
    D_right = D_bonds[angle_indices[:,1]]
    
    diff_left  = angle_signs[:,0][:,None] * diff_bonds[angle_indices[:,0]]
    diff_right = angle_signs[:,1][:,None] * diff_bonds[angle_indices[:,1]]
    
    # Normalized the left and right vectors
    A = -diff_left/D_left[:,None]
    B = diff_right/D_right[:,None]
    
    # Angle calculation (Clip to avoid numerical errors)
    dot_product = np.sum(A * B, axis=1)
    theta = np.arccos(np.clip(dot_product, -1.0, 1.0))  

    # Force magnitude per angle
    Fr = ktheta * (theta0 - theta)
    
    # Calculate direction vectors
    n = np.cross(A, B)  

    vec1 = np.cross(n, A)
    vec2 = np.cross(B, n)
    
    # Normalize direction vectors and scale with force 
    vec1 /= np.linalg.norm(vec1, axis=1)[:, None]
    vec2 /= np.linalg.norm(vec2, axis=1)[:, None]

    vec1 *= (Fr / D_left)[:,None]
    vec2 *= (Fr / D_right)[:,None]
    
    # Initialize angular force matrix
    ang_mat = np.zeros((nr_atoms,3))
    
    # Extract indices for left, middle, and right atoms
    np.add.at(ang_mat, angles[:, 1], vec1 + vec2)
    np.add.at(ang_mat, angles[:, 0], -vec1)
    np.add.at(ang_mat, angles[:, 2], -vec2)
    
    return ang_mat  

def LJ_force(diff_LJ, D_LJ, epsilon_LJ, sigma_LJ, i_LJ, j_LJ, nr_atoms, 
             r_cutoff_LJ=np.inf,cap=None):
    """
    Compute the Lennard-Jones forces acting on precomputed indices
    and leveraging antisymmetry for efficiency.
    """
    # Identify relevant pairs
    valid_pairs = D_LJ < r_cutoff_LJ
    
    
    # Extract relevant data for valid pairs
    D_masked = D_LJ[valid_pairs]
    diff_masked = diff_LJ[valid_pairs]
    EPS_masked = epsilon_LJ[valid_pairs]
    SIG_masked = sigma_LJ[valid_pairs]
    
    if cap is not None:
        D_masked = np.maximum(D_masked, cap[valid_pairs])
        

    # Compute Lennard-Jones force magnitude for valid pairs
    SIGovrD6 = (SIG_masked/D_masked)**6
    SIGovrD12 = SIGovrD6**2

    F_scalar = 24 * EPS_masked * (2 * SIGovrD12 - SIGovrD6) / D_masked

    
    # Compute force vectors for valid pairs
    F_ij_masked = (F_scalar/ D_masked)[:, None] * diff_masked

    
    # Accumulate forces
    F = np.zeros((nr_atoms, 3))

    # Add contributions from valid pairs using antisymmetry
    if F_ij_masked.size > 0:
        for dim in range(3):
            F[:, dim] = np.bincount(i_LJ[valid_pairs], weights=F_ij_masked[:, dim], 
                                    minlength=nr_atoms)
            F[:, dim]-= np.bincount(j_LJ[valid_pairs], weights=F_ij_masked[:, dim], 
                                     minlength=nr_atoms)
    
    return F

def dihedral_force(diff_bonds, D_bonds, dihedral_mapping, dihedrals, C1, C2, 
                   C3, C4, nr_atoms):
    """Computes the dihedral forces."""
    # Retrieve the differences and distances from the arrays using the mapping
    dihedral_indices, dihedral_signs = dihedral_mapping
    
    D_ab = D_bonds[dihedral_indices[:,0]]
    D_bc = D_bonds[dihedral_indices[:,1]]
    D_cd = D_bonds[dihedral_indices[:,2]]
    
    diff_ab = dihedral_signs[:,0][:,None] * diff_bonds[dihedral_indices[:,0]]
    diff_bc = dihedral_signs[:,1][:,None] * diff_bonds[dihedral_indices[:,1]]
    diff_cd = dihedral_signs[:,2][:,None] * diff_bonds[dihedral_indices[:,2]]
    
    # Normalized difference vectors between the three pairs
    ab = diff_ab/D_ab[:,None]
    bc = diff_bc/D_bc[:,None]
    cd = diff_cd/D_cd[:,None]
    
    # Construct and normalize direction vectors
    p1 = np.cross(-ab,bc)
    p2 = np.cross(cd,-bc)
    
    p1 /= np.linalg.norm(p1, axis=1, keepdims=True)
    p2 /= np.linalg.norm(p2, axis=1, keepdims=True)
    
    # Compute signed dihedral angle theta
    dot_p1_p2 = np.einsum('ij,ij->i', p1, p2)
    
    m1 = np.cross(p1, p2)
    sign = np.sign(np.einsum('ij,ij->i', m1, bc))
    
    theta_unsigned = np.arccos(np.clip(dot_p1_p2, -1.0, 1.0))
    theta = sign * theta_unsigned
    
    # Compute angles at b (theta1) and c (theta2)
    dot_ab_bc = np.einsum('ij,ij->i', ab,-bc)
    dot_bc_cd = np.einsum('ij,ij->i', bc,-cd)
    
    theta1 = np.arccos(np.clip(dot_ab_bc, -1.0, 1.0))
    theta2 = np.arccos(np.clip(dot_bc_cd, -1.0, 1.0))   

    # Precalculate the big factor
    sin_terms = C1 * np.sin(theta) - 2 * C2 * np.sin(2 * theta) + \
                3 * C3 * np.sin(3 * theta) - 4 * C4 * np.sin(4 * theta)
    
    # Calculate fa and fd
    fa = (0.5 * sin_terms / (D_ab * np.sin(theta1)))[:, None] * p1
    fd = (0.5 * sin_terms / (D_cd * np.sin(theta2)))[:, None] * p2

    # Force on c using intermediate o and t
    oc = 0.5 * diff_bc
    tc = -(np.cross(oc, fd) + 0.5 * np.cross(diff_cd, fd) + \
           0.5 * np.cross(-diff_ab, fa))
    fc = 4 / (D_bc**2)[:, None] * np.cross(tc, oc)

    # Aggregate forces
    F = np.zeros((nr_atoms, 3))
    
    np.add.at(F, dihedrals[:, 0], fa)
    np.add.at(F, dihedrals[:, 1], -fa - fc - fd) # Newton's law
    np.add.at(F, dihedrals[:, 2], fc)
    np.add.at(F, dihedrals[:, 3], fd)

    return F


def net_force(diff_bonds, D_bonds, diff_LJ, D_LJ, bonds, nr_atoms, 
               angle_mapping, dihedral_mapping, kb, r0, angles, ktheta, theta0, 
               epsilon_LJ, sigma_LJ, dihedrals, C1, C2, C3, C4, i_LJ, j_LJ, 
               r_cutoff=np.inf,cap_LJ=None):
    """
    Calculate the total net force on atoms in a system by summing
    bond, angular, dihedral and LJ forces.
    """
    bond_forces = bond_force(diff_bonds, D_bonds, bonds, kb, r0, nr_atoms)

    angular_forces = ang_force(diff_bonds, D_bonds, angle_mapping, angles, 
                               ktheta, theta0, nr_atoms)

    LJ_forces = LJ_force(diff_LJ, D_LJ, epsilon_LJ, sigma_LJ, i_LJ, j_LJ, 
                         nr_atoms, r_cutoff, cap_LJ)

    di_forces = dihedral_force(diff_bonds, D_bonds, dihedral_mapping, 
                               dihedrals, C1, C2, C3, C4, nr_atoms)

    return bond_forces + angular_forces + di_forces + LJ_forces