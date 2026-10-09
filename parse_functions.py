import numpy as np

# ---------------------- xyz read/write file functions ------------------------
def xyz_to_array(filename):
    """Opens the file in filename, and returns the molecules listed and an 
    array with the molecule positions over time."""
    # Open the file and get initial info
    with open(filename, 'r') as file:
        nr_of_atoms = int(file.readline().strip())  # Read number of atoms
        # Calculate the number of lines in the file
        file.readline()
        atoms=np.empty((nr_of_atoms),dtype=str)
        for i in range(nr_of_atoms):
            atoms[i]=file.readline().strip()[0] 

    # Initialize empty array for storing coordinates
    output_array = np.empty((nr_of_atoms, 3), dtype='f')

    # Re-open file to start reading data
    with open(filename, 'r') as file:
        file.readline()  # Skip nr_of_atoms line
        file.readline()  # Skip molecule name line
        
        # Read atom coordinates for each timestep
        for atom_index in range(nr_of_atoms):
            line = file.readline().strip().split()
            # Extract x, y, z coordinates and fill the array
            x, y, z = map(float, line[1:])
            output_array[atom_index] = [x, y, z]
                
    # The code works in nm, not Angstrom. So scale units.
    return atoms,0.1*output_array 

def array_to_xyz(array, atoms, path, t, method='append'):
    """
    Saves the trajectory data to a .xyz file readable by visualization tools.
    """
    nr_atoms, _ = array.shape
    #VMD works with Anstrom, not nm. So scale the positions
    array*=10
    
    action='w'
    if method=='append':
        action='a'
    
    with open(path, action) as f:
        f.write(f"{nr_atoms}\n")
        f.write(f"Some text: t={t}\n")
        for i in range(nr_atoms):
            xyz = array[i].astype(str)
            f.write(f"{atoms[i]}         {xyz[0][0:10]}      "
                    f"{xyz[1][0:10]}      {xyz[2][0:10]}\n")
            
# --------------------- itp read/write file functions -------------------------
def itp_to_data(filename):
    """
    Parse an ITP-like file to extract molecule data.
    """
    with open(filename, 'r') as file:
        # Parse the number of bonds
        num_bonds = int(file.readline().strip()[6:])
        bonds = np.empty((num_bonds,2), dtype=int)
        bond_constants = np.empty(num_bonds)
        bond_lengths = np.empty(num_bonds)
        
        # Read bond data
        for i in range(num_bonds):
            line = file.readline().strip().split()
            atom1, atom2, k_b, r0 = map(float, line)
            # Import the bonds in sorted order
            bonds[i] = int(min(atom1,atom2)), int(max(atom1,atom2))
            bond_constants[i] = 100*k_b
            bond_lengths[i] = 0.1*r0
        
        # Parse the number of angles
        num_angles = int(file.readline().strip()[7:])
        angles = np.empty((num_angles,3), dtype=int)
        angle_constants = np.empty(num_angles)
        equilibrium_angles = np.empty(num_angles)
        
        # Read angle data
        for i in range(num_angles):
            line = file.readline().strip().split()
            atom1, atom2, atom3, k_theta, theta0 = map(float, line)
            angles[i] = int(atom1), int(atom2), int(atom3)
            angle_constants[i] = k_theta
            equilibrium_angles[i] = theta0
            
        # Parse the number of dihedrals
        num_dihedrals = int(file.readline().strip()[10:])
        dihedrals = np.empty((num_dihedrals,4), dtype=int)
        C1 = np.empty(num_dihedrals)
        C2 = np.empty(num_dihedrals)
        C3 = np.empty(num_dihedrals)
        C4 = np.empty(num_dihedrals)
        
        # Read dihedral data
        for i in range(num_dihedrals):
            line = file.readline().strip().split()
            atom1, atom2, atom3, atom4, c1, c2, c3, c4 = map(float, line)
            dihedrals[i] = int(atom1), int(atom2), int(atom3), int(atom4)
            C1[i] = c1
            C2[i] = c2
            C3[i] = c3
            C4[i] = c4
        
        num_atoms = int(file.readline().strip()[6:])
        epsilon = np.empty(num_atoms, dtype=float)
        sigma = np.empty(num_atoms, dtype=float)
        masses = np.empty(num_atoms, dtype=float)
        for i in range(num_atoms):
            line = file.readline().strip().split()
            e, s, m = map(float, line)
            epsilon[i] = e
            sigma[i] = s
            masses[i] = m
        
    
    return bonds, bond_constants, bond_lengths, angles, angle_constants, \
         equilibrium_angles, epsilon, sigma, dihedrals, C1, C2, C3, C4, masses

def data_to_itp(filename, bonds, bond_constants, bond_lengths, angles, 
                angle_constants, equilibrium_angles, epsilon, sigma, dihedrals, 
                C1, C2, C3, C4, masses):
     """
     Parse molecule data to produce an ITP file.
     """   
     with open(filename, 'w') as f:
        f.write(f"bonds {len(bonds)}\n")
        for idx, bond in enumerate(bonds):
            f.write(f"{bond[0]} {bond[1]} {0.01*bond_constants[idx]}"
                    f" {10*bond_lengths[idx]}\n")
        f.write(f"angles {len(angles)}\n")   
        for idx, angle in enumerate(angles):
            f.write(f"{angle[0]} {angle[1]} {angle[2]} {angle_constants[idx]}"
                    f" {equilibrium_angles[idx]}\n")
        f.write(f"dihedrals {len(dihedrals)}\n")   
        for idx, dihedral in enumerate(dihedrals):
            f.write(f"{dihedral[0]} {dihedral[1]} {dihedral[2]} {dihedral[3]}"
                    f" {C1[idx]} {C2[idx]} {C3[idx]} {C4[idx]}\n")
        f.write(f"atoms {len(masses)}\n")  
        for idx in range(len(masses)):
            f.write(f"{epsilon[idx]} {sigma[idx]} {masses[idx]}\n")