# Import necessary libraries and modules
from initial_config_generation import random_config
from integrators import integrate, velo_verlet, beeman, verlet, euler
from auxiliary import validate_parameters

import os
import numpy as np

# Simulation parameters
# Define molecule types and their weights
molecules = ["Water", "Ethanol", "Oxygen", "Hydrogen"]

# Will be normalized later on, can be >1
water_weight = float(input("Enter water weight: "))
ethanol_weight = float(input("Enter ethanol weight: "))

weights = [water_weight, ethanol_weight, 0, 0]

 # Size of the box (nm)
boxsize = 4  

random_grid_offset = 0          # random offset from initial grid. abs<0.5
r_threshold_scalar = 0.65       # Scaling factor for min LJ-distances in init. 
                                # Lower values risk instability, higher values 
                                # increase init time. Recommended: 0.5–0.7.
                                # Set to None for no initialization (unsafe)

# Integration parameters

dt = 0.001                         
# Time step (ps)
endTime = 200
# Total simulation time (ps)
save_steps = 100                 
# Save results every 'save_steps'

method = velo_verlet                # Integration method
initial_temp =  298.15              # Initial temperature (K). room temp = 298.15
temp_control = True                 # Boolean to enable/disable constant temp

# Choose a value for one of the two variables below, and set the other to None
f_threshold = None                  # Any force (10^15 N) lower than this will 
                                    # not be simulated. Set to 0 to simulate all 
                                    # LJ forces, set to np.inf to simulate no LJ 
                                    # forces. Typically set between 0 and 0.1
                                    
r_cutoff_scalar = 2                 # Scalar for the cutoff distance (nm) for 
                                    # LJ force. Set to 0 to simulate no 
                                    # LJ forces, set to np.inf to simulate all LJ 
                                    # forces. Typically set to 2 or 2.5. 

# Visualization parameters
VMD_vis = False
energy_graph = True
temp_graph = True
phase_space_diagram = True
comptime_vis = True 
RDF_vis = True
RDF_thin = 1                       # Thinning value for RDF samples.  >0 & <1 
RDF_dr = 0.005                     # Bin size in the histograms of the RDF. >0


# Bonds to plot in the phase space diagram
bonds_of_interest = ['O-H (Water)',
                     'O-H (Ethanol)',
                     'C-O (Ethanol)',
                     'C-H (Ethanol)',
                     'O-O (Oxygen)']

# Pairs to plot in the RDF visualizations
pairs_of_interest = ['O~H (Water~Ethanol)',
                     'O~H (Ethanol~Ethanol)',
                     'O~H (Water~Water)',
                     'O~O (Water~Water)',
                     'O~O (Water~Ethanol)',
                     'O~O (Ethanol~Ethanol)']

# Validate all parameters
validate_parameters(molecules, weights, boxsize, random_grid_offset, dt, 
                    endTime, save_steps, method, initial_temp, temp_control,
                    f_threshold, r_cutoff_scalar, VMD_vis, energy_graph, 
                    temp_graph, phase_space_diagram, RDF_vis, RDF_thin, RDF_dr, 
                    bonds_of_interest, pairs_of_interest, r_threshold_scalar)
    
# Initialization step
print("Creating initial configuration...")
(atoms, x, bonds, kb, r0, angles, ktheta, theta0, 
epsilon, sigma, dihedrals, C1, C2, C3, C4, 
masses, molec_index, molec_names) = random_config(boxsize, random_grid_offset, 
                                                  molecules, weights)
                                                 
# Generate folder name based on parameters
name = '_'.join([f"{weights[i]}{molecules[i]}" 
                 for i in range(len(molecules)) if weights[i] > 0])

# Use os.path.join to construct the folder path
base_folder = "trajectories"
folder_name = os.path.join(base_folder, f"boxsize={boxsize:.2f}_roff={random_grid_offset}__{name}")

# Create the folder if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Evolve the initial configuration
integrate(
    x, bonds, atoms, kb, r0, angles, ktheta, theta0, epsilon, sigma, dihedrals, 
    C1, C2, C3, C4, masses, method, dt, endTime, save_steps, boxsize, 
    folder_name, molec_index, molec_names, r_cutoff_scalar, f_threshold,
    initial_temp, temp_control, bonds_of_interest, pairs_of_interest, RDF_thin, 
    RDF_dr, VMD_vis, energy_graph, temp_graph, phase_space_diagram, RDF_vis, 
    comptime_vis, r_threshold_scalar
)
