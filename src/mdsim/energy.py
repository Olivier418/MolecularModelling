# Import necessary libraries and modules
import numpy as np

def bond_energy(D_bonds,kb,r0):
    """Calculates bond energy"""
    BE = 0.5 * np.dot(kb, (D_bonds - r0)**2)
    return(BE)

def kinetic_energy(masses,V):
    """Calculates kinetic energy"""
    KE = 0.5 * np.dot(masses, np.sum(V**2, axis=1))
    return(KE)

def angular_energy(diff_bonds, D_bonds, angle_mapping, ktheta, theta0):
    """Calculates angular energy"""
    # Retrieve the differences and distances from the arrays using the mapping
    angle_indices, angle_signs = angle_mapping
    
    D_left = D_bonds[angle_indices[:,0]]
    D_right = D_bonds[angle_indices[:,1]]
    
    diff_left  = angle_signs[:,0][:,None] * diff_bonds[angle_indices[:,0]]
    diff_right = angle_signs[:,1][:,None] * diff_bonds[angle_indices[:,1]]
    
    # Normalized vectors per angle
    A = -diff_left/D_left[:,None]
    B = diff_right/D_right[:,None]
    
    # Angle calculation (Clip to avoid numerical errors)
    dot_product = np.sum(A * B, axis=1)
    theta = np.arccos(np.clip(dot_product, -1.0, 1.0))   
    AE = 0.5 * np.dot(ktheta, (theta - theta0)**2)
    return(AE)

def LJ_energy(D_LJ, epsilon_LJ, sigma_LJ, r_cutoff_LJ=np.inf):
    """
    Compute the Lennard-Jones energy of precomputed indices
    and leveraging antisymmetry for efficiency.
    """
    # Identify relevant pairs
    valid_pairs = D_LJ < r_cutoff_LJ
    
    # Extract relevant data for valid pairs
    D_masked = D_LJ[valid_pairs]
        
    EPS_masked = epsilon_LJ[valid_pairs]
    SIG_masked = sigma_LJ[valid_pairs]

    # Compute Lennard-Jones force magnitude for valid pairs
    SIGovrD6 = (SIG_masked/D_masked)**6
    SIGovrD12 = SIGovrD6**2
    
    U_scalar = 4 * EPS_masked * (SIGovrD12 - SIGovrD6)
    LJE = np.sum(U_scalar)   
    return LJE


def dihedral_energy(diff_bonds, D_bonds, dihedral_mapping, C1, C2, C3, C4):
    """Compute the dihedral energy"""
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
    
    # Total energy    
    DE = np.sum(0.5*(C1*(1+np.cos(theta))+C2*(1-np.cos(2*theta))+
                     C3*(1+np.cos(3*theta))+C4*(1-np.cos(4*theta))))
    
    return(DE)