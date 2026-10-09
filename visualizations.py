# Import necessary libraries and modules
from config import kboltz
from auxiliary import updt

import subprocess    
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.stats import norm

def computation_piechart(computation_times, save_steps, total_time, 
                         folder_name):  
    """
    Generate and save a pie chart of computation time distribution.
    """
    # Compute total time per category
    times = np.sum(computation_times, axis=0)
    
    # Scale dist and force times, since more time was spent on them outside
    # of the save_visualization_data function.
    times[[0, 1]] *= save_steps
    
    # Identify the leftover time and append it.
    other = total_time - np.sum(times)
    times = np.concatenate((times, np.array([other])))

    # Labels and colors
    labels = np.array(["Diff & Dist", "Forces", "Writing XYZ", "RDF", "Energy", 
                       "temperature", "PS diag", "Other"])
    colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#7f7f7f','lightgrey'])
    
    # Filter small values
    mask = times > 0.01 * total_time
    times, labels, colors = times[mask], labels[mask], colors[mask]
    
    # Plot the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(times, labels=labels, colors=colors, 
            autopct=lambda p: f'{p * sum(times) / 100:.2f} s', startangle=140)
    plt.title(f"Total computation time for "
              f"{save_steps * len(computation_times)} iterations: "
              f"{total_time:.2f} s (save_steps={save_steps})")
    
    # Save and show
    
    plt.savefig(os.path.join(folder_name,'comptimes_piechart.png'), dpi=300, 
                bbox_inches='tight')
    plt.show()

def visualize_xyz_with_vmd(xyz_filepath,boxsize,d_max=0.15):
    """Launch vmd, with additional arguments to use boxsize and dynamic bonds
    with d_max."""
    # Ensure the file exists
    if not os.path.isfile(xyz_filepath):
        raise FileNotFoundError(f"The file {xyz_filepath} does not exist.")
    # Generate the Tcl script for VMD 
    tcl_script = f"""
    # Load the .xyz file
    mol new {{{xyz_filepath}}} type xyz
    
    # Add a representation for the molecule
    mol representation DynamicBonds {d_max}
    mol selection all
    mol material Opaque
    mol color Name
    mol modstyle 0 0 DynamicBonds
    
    # Draw a box around the simulation
    graphics top color red
    """
    # Define the corners of the box
    corners = np.array(
        np.meshgrid([0, boxsize], [0, boxsize], [0, boxsize], 
                    indexing='ij')).T.reshape(-1, 3)
    # Add the edges of the box
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), 
        (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
    ]
    for edge in edges:
        p1 = corners[edge[0]]
        p2 = corners[edge[1]]
        tcl_script += (f"graphics top line {{{p1[0]} {p1[1]} {p1[2]}}} "
                       f"{{{p2[0]} {p2[1]} {p2[2]}}}\n")

    # Save the Tcl script to a temporary file
    tcl_script_path = "temp_vmd_script.tcl"
    with open(tcl_script_path, "w") as f:
        f.write(tcl_script)
    
    try:
        # Use the full path to VMD executable (update path if needed)
        vmd_path = r"C:\Program Files\VMD\vmd.exe"
        subprocess.run([vmd_path, "-e", tcl_script_path], check=True)
    finally:
        # Clean up the temporary Tcl script
        if os.path.exists(tcl_script_path):
            os.remove(tcl_script_path)

def temperature_plot(KE, Nf, endTime, folder_name):
    """Plots the temperature using the data in KineticEnergy. """
    # Generate time points for the x-axis
    time_points = np.linspace(0, endTime, len(KE))
    
    #Compute temperature
    T = 2*KE/(kboltz*Nf)
    
    plt.plot(time_points, T, label="Temperature")
    
    # Add labels, legend, and title
    plt.xlabel("Time (ps)")
    plt.ylabel("Temperature (K)")
    plt.title("Plot of system temperature")
    plt.legend()
    plt.grid()
    
    # Save and show the plot
    
    plt.savefig(os.path.join(folder_name,'temperature.png'))
    plt.show()


def energy_plot(energy_matrix, endTime, folder_name):
    """Creates a plot of the 5 types of energy in the system, as well as the
    total energy."""
    # Generate time points for the x-axis
    num_points = energy_matrix.shape[0]
    time_points = np.linspace(0, endTime, num_points)
    
    # Plot energy components and total energy
    plt.plot(time_points, energy_matrix[:, 0], label="Bond energy")
    plt.plot(time_points, energy_matrix[:, 1], label="Angular energy")
    plt.plot(time_points, energy_matrix[:, 2], label="LJ energy")
    plt.plot(time_points, energy_matrix[:, 3], label="Dihedral energy")
    plt.plot(time_points, energy_matrix[:, 4], label="Kinetic energy")
    plt.plot(time_points, np.sum(energy_matrix, axis=1), label="Total energy")
    
    # Add labels, legend, and title
    plt.xlabel("Time (ps)")
    plt.ylabel("Energy (kJ/mol)")
    plt.title("Plot of energy in the system")
    plt.legend()
    plt.grid()
    
    # Save and show the plot
    plt.savefig(os.path.join(folder_name,'energy.png'))
    plt.show()
    
    plt.close()
    
# ------------------------- phase space diagram functions ---------------------
def initialize_ps_diag(T, molecule_names, atom_types, bonds,bonds_of_interest):
    """Identifies bonds to keep track of during integration, computes
    mask to achieve this as well as arrays to store this data in."""
    # Identify the unique bonds present in the system
    bond_atoms = np.sort(np.array(
        [atom_types[bonds[:,0]],atom_types[bonds[:,1]]]).T,axis=1)
    bond_molecs = molecule_names[bonds[:,0]]
    
    bonds = np.column_stack((bond_atoms, bond_molecs))
    uniq_system_bonds = np.unique(bonds,axis=0)
    
    # Process the user input (which is given in the form of a list of strings)
    split_data = [entry.split(' ', 1) for entry in bonds_of_interest]
    atom_pairs = np.array([item[0] for item in split_data])
    molecules = np.array([item[1][1:-1] for item in split_data])
    atoms = np.char.split(atom_pairs, '-')
    atoms = np.array([np.sort(pair) for pair in atoms])
    interest_bonds = np.unique(np.column_stack((atoms, molecules)),axis=0)
    
    # Identify which bonds of interest are present in the system.
    is_in = np.all(np.isin(interest_bonds,uniq_system_bonds),axis=1)
    valid_bonds = np.array(interest_bonds)[is_in]
    nr_valid_bonds = len(valid_bonds) 
    
    # Compute masks to keep track of each bond.
    mask_per_bond = np.array([np.all(bonds==bond,axis=1) 
                              for bond in valid_bonds])
    nr_per_bond = np.array([np.sum(mask) for mask in mask_per_bond])
    
    # Create arrays to store data in.
    data_arrays = np.empty(nr_valid_bonds,dtype=object)
    for i in range(nr_valid_bonds):
        data_arrays[i] = np.empty((T,nr_per_bond[i]))
    
    return(valid_bonds, mask_per_bond, data_arrays)

def show_ps_diag(valid_bonds, data_arrays, dt, folder_name):
    """Creates a phase space diagram for each bond in valid_bonds using the 
    data in data_arrays."""
    for i, bond in enumerate(valid_bonds):
        # Extract trajectory data and compute D and V (index to equal lengths.)
        dist_data = data_arrays[i]
        V = np.diff(dist_data, axis=0) / dt  
        D = dist_data[1:] 

        T, nr_trajectories = dist_data.shape

        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a list of segments
        segments = [np.column_stack((D[:, j], V[:, j])) 
                    for j in range(nr_trajectories)]
        lc = LineCollection(segments, cmap='viridis', alpha=0.7)
        lc.set_array(np.linspace(0, 1, nr_trajectories))
        ax.add_collection(lc)

        # Add labels and titles
        ax.set_xlabel('Bond Distance (D)')
        ax.set_ylabel('Bond Velocity (V)')
        width = D.max()-D.min()
        height = V.max()-V.min()
        
        # Set axis limits to properly view the the trajectory.
        ax.set_xlim(D.min()-0.05*width ,D.max()+0.05*width )
        ax.set_ylim(V.min()-0.05*height,V.max()+0.05*height)
        title = f"{bond[0]}~{bond[1]} ({bond[2]})"
        ax.set_title(f'Phase Space Diagram for {title} Bonds')

        # Save and close the figure to save memory.
        plt.savefig(os.path.join(folder_name,f"ps_diag_{title}.png"), bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
# ----------------------------------- RDF functions --------------------------
def initialize_RDF(T, molecule_names, i_LJ, j_LJ, bonds, atom_types, 
                   pairs_of_interest, box_size, RDF_thin, dr=0.05):
    """Identifies the pairs in the system to keep track of, as well as indices 
    to retrieve data during the integration and arrays to store this data in.
    """
    # Calculate the number of bins
    domain = box_size/2
    nr_bins = int(domain / dr)
    bin_edges = np.linspace(0, domain, nr_bins + 1)

    # Process the user input (which is given in the form of a list of strings)
    split_poi = [entry.split(' ', 1) for entry in pairs_of_interest]
    atom_pairs_of_interest = [sorted(pair.split('~')) for pair, _ in split_poi]
    molecule_pairs_of_interest = [sorted(mol.strip('()').split('~')) 
                                  for _, mol in split_poi]
    
    # Identify all possible atom and molecule pairs
    unique_data, _ = np.unique(np.array([atom_types, molecule_names]), axis=1, 
                               return_index=True)
    atom_pairs = np.array([np.array(np.meshgrid(row, row)).T.reshape(-1, 2) 
                           for row in unique_data])
    molecule_pairs = np.array([np.array(np.meshgrid(row, row)).T.reshape(-1, 2) 
                               for row in unique_data])
    # Sort pairs for consistency
    atom_pairs = np.sort(atom_pairs, axis=1)
    molecule_pairs = np.sort(molecule_pairs, axis=1)
    # Get unique pairs
    atom_pairs, molecule_pairs = np.unique(np.array([atom_pairs, 
                                                     molecule_pairs]), axis=1)
    
    # Identify which pairs of interest are present in the system.
    is_in_pairs = np.isin(
        np.array([atom_pairs_of_interest, molecule_pairs_of_interest]),
        np.array([atom_pairs, molecule_pairs])
    )
    mask = np.all(is_in_pairs, axis=(0, 2))
    valid_atoms = np.array(atom_pairs_of_interest)[mask]
    valid_molecules = np.array(molecule_pairs_of_interest)[mask]
    
    # Create pair names for valid atom and molecule pairs
    atom_pair_names = np.char.add(np.char.add(valid_atoms[:, 0], 
                                              '~'), 
                                  valid_atoms[:, 1])
    molecule_pair_names = np.char.add(
        np.char.add(' (',
            np.char.add(
                np.char.add(valid_molecules[:, 0], '~'),
                valid_molecules[:, 1])
        ), ')'
    )
    pair_names = np.char.add(atom_pair_names, molecule_pair_names)

    nr_valid_pairs = len(valid_atoms)
    
    
    # Initialize arrays to store indices for each valid pair of interest
    RDF_LJ_indices = np.empty(nr_valid_pairs, dtype=object)
    RDF_bond_indices = np.empty(nr_valid_pairs, dtype=object)
    RDF_leftover_indices = np.empty((nr_valid_pairs,2), dtype=object)
    
    LJ_pairs = np.core.records.fromarrays([i_LJ, j_LJ], dtype=[('i', 'i4'), 
                                                               ('j', 'i4')])
    bond_pairs = np.core.records.fromarrays([bonds[:, 0], bonds[:, 1]], 
                                            dtype=[('i', 'i4'), ('j', 'i4')])
    
    # Sort LJ_pairs and bond_pairs for fast membership lookup
    LJ_pairs_sorted = np.sort(LJ_pairs)
    bond_pairs_sorted = np.sort(bond_pairs)
    
    # Loop over valid pairs
    for pair_index, (valid_atom_pair, valid_molecule_pair) in enumerate(
                                                        zip(valid_atoms, 
                                                            valid_molecules)):
        
        # compare atoms in system to valid atoms (and molecules too)
        atom_pair_mask = (atom_types[:, None] == valid_atom_pair[0]) & (
            atom_types[None, :] == valid_atom_pair[1])
        molecule_pair_mask = (molecule_names[:, None] == valid_molecule_pair[0]
                              ) & (molecule_names[None, :] == 
                                   valid_molecule_pair[1])
        
        # Compute mask for pairs of interest
        B = atom_pair_mask & molecule_pair_mask
        distances_of_interest = np.triu(B | B.T, k=1)
        i_all, j_all = np.where(distances_of_interest) # Retrieve indices
    
        # Convert to Fast set membership check with sorted arrays
        true_indices = np.core.records.fromarrays([i_all, j_all], 
                                                  dtype=[('i', 'i4'), 
                                                         ('j', 'i4')])
        mask_LJ = np.in1d(true_indices, LJ_pairs_sorted  , assume_unique=True)
        mask_bond=np.in1d(true_indices, bond_pairs_sorted, assume_unique=True)
    
        # Identify pairs of interest that are not tracked by LJdist or bonddist
        mask = ~(mask_LJ | mask_bond)
        i_leftover, j_leftover = i_all[mask], j_all[mask]
        
        i_leftover = np.array(i_leftover,dtype=int)
        j_leftover = np.array(j_leftover,dtype=int)
    
        # Identify indices for each of the three categories
        LJ_indices = np.where(distances_of_interest[i_LJ, j_LJ])[0]
        bond_indices = np.where(distances_of_interest[bonds[:, 0], 
                                                      bonds[:, 1]])[0]
        leftover_indices = np.array([i_leftover, j_leftover])
        
        # Select an RDF_thin percentage of all indices to reduce time spent 
        # on computing distances.
        nr_LJ = LJ_indices.shape[0]
        nr_bond = bond_indices.shape[0]
        nr_leftover = leftover_indices.shape[1]
        
        nr_sel_LJ = int(np.ceil(nr_LJ*RDF_thin))
        nr_sel_bonds = int(np.ceil(nr_bond*RDF_thin))
        nr_sel_leftover = int(np.ceil(nr_leftover*RDF_thin))
        
        # Store the resulting indices in their corresponding arrays.
        RDF_LJ_indices[pair_index] =LJ_indices[np.random.choice(nr_LJ, 
                                                                nr_sel_LJ, 
                                                                False)]
        RDF_bond_indices[pair_index] = bond_indices[np.random.choice(nr_bond, 
                                                                 nr_sel_bonds, 
                                                                 False)]
        
        leftover_indices_selection = leftover_indices[:,np.random.choice(
                                                              nr_leftover, 
                                                              nr_sel_leftover, 
                                                              False)]
        RDF_leftover_indices[pair_index] = (leftover_indices_selection[0],
                                            leftover_indices_selection[1])
        
    # Create the array 
    RDF_array = np.empty((T, len(valid_atoms), nr_bins))
    
    bin_volumes = 4/3*np.pi*(bin_edges[1:]**3-bin_edges[:-1]**3)

    return (RDF_LJ_indices, RDF_bond_indices, RDF_leftover_indices, 
            bin_edges, bin_volumes, pair_names, RDF_array)


def RDF_visualize(hist_values, bin_edges, bin_volumes, unique_pairs, timestep, 
                 folder_name, r0, RDF_conf=0.9):    
    """Creates gifs of histograms that track the frequencies of the 
    distances between pairs of interest over time"""  
    T, nr_unique_pairs, nr_bins = hist_values.shape
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pairs))) 

    # Ensure the folder exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Normalize data for all pairs at once
    data = hist_values / bin_volumes[None, None, :]
    initial_sums = np.sum(data[:, :, :], axis=2, keepdims=True)  # Sum over bins for normalization
    data /= initial_sums[:, :, 0, None]  # Normalize to the sum of the first frame for each pair

    # Vectorized computation of cumulative statistics
    cumulative_data = np.cumsum(data, axis=0)  # Cumulative sum for all time steps
    n = np.arange(1, T + 1)[:, None, None]  # Sample size at each timestep (T x 1 x 1)
    means = cumulative_data / n  # Means up to each timestep
    squared_diffs = (data - means) ** 2
    std_devs = np.sqrt(np.cumsum(squared_diffs, axis=0) / n)  # Standard deviation
    
    # Confidence intervals (95%)
    margin_of_error = norm.ppf(1 - 0.5 * (1 - RDF_conf)) * (std_devs / np.sqrt(n))
    conf_intervals = np.zeros((T, nr_unique_pairs, nr_bins, 3))
    conf_intervals[:, :, :, 0] = means - margin_of_error  # Lower bound
    conf_intervals[:, :, :, 1] = means  # Mean
    conf_intervals[:, :, :, 2] = means + margin_of_error  # Upper bound



    # Identify the largest frequency over time to set ylim properly later
    
    helpdist = int(1.2*np.max(r0)/bin_edges[-1]*nr_bins)
    upper = conf_intervals[-1, :, helpdist:, 2]
    max_freq = np.max(upper[np.isfinite(upper)])  # Use the 95th percentile of upper bounds

    # Plotting and saving for each pair
    for i, pair in enumerate(unique_pairs):
        pair_folder = os.path.join(folder_name, str(pair))
        if not os.path.exists(pair_folder):
            os.makedirs(pair_folder)

        # Plotting
        fig, ax = plt.subplots(figsize=(9, 6))
        line_mean, = ax.plot(bin_edges[:-1], conf_intervals[0, i, :, 1], 
                             color=colors[i], label="mean")
        line_lower, = ax.plot(bin_edges[:-1], conf_intervals[0, i, :, 0], 
                              color="lightgray", linewidth=0.75)
        line_upper, = ax.plot(bin_edges[:-1], conf_intervals[0, i, :, 2], 
                              color="lightgray", linewidth=0.75, label=f"{int(100 * RDF_conf)}% confidence interval")

        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.set_ylim(0, 1.25 * max_freq)
        ax.set_title(f"RDF of {pair} at t=0")
        ax.legend(loc='upper right')

        plt.savefig(os.path.join(pair_folder, f'first_frame_{pair}.png'))

        def update(frame):
            line_mean.set_ydata(conf_intervals[frame, i, :, 1])
            line_lower.set_ydata(conf_intervals[frame, i, :, 0])
            line_upper.set_ydata(conf_intervals[frame, i, :, 2])
            ax.set_title(f"RDF of {pair} at t={frame * timestep:.4f}")
            
            if frame > 0:
                updt((T - 1) * (nr_unique_pairs+1), frame + (T - 1) * i)

        ani = FuncAnimation(fig, update, frames=range(T), interval=1)
        ani.save(os.path.join(pair_folder, f'RDF_{pair}.gif'), writer='pillow')
        plt.savefig(os.path.join(pair_folder, f'last_frame_{pair}.png'))
        plt.close()
       
       
    # Identify the largest frequency over time to set ylim properly later
    
    upper = conf_intervals[-1, :, helpdist:, 2]
    max_freq = np.max(upper[np.isfinite(upper)])  # Use the 95th percentile of upper bounds

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 6))
    line_means=[]
    
    for i, pair in enumerate(unique_pairs):
        line_mean, = ax.plot(bin_edges[:-1], conf_intervals[0, i, :, 1], 
                             color=colors[i], label=f"{pair}")
        line_means.append(line_mean)
        

    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_ylim(0, 1.25 * max_freq)
    ax.set_title("RDF at t=0")
    ax.legend(loc='upper right')

    plt.savefig(os.path.join(folder_name, 'first_frame.png'))

    def update(frame):
        for i, pair in enumerate(unique_pairs):
            line_means[i].set_ydata(conf_intervals[frame, i, :, 1])
        ax.set_title(f"RDF at t={frame * timestep:.4f}")
        
        if frame > 0:
            updt((T - 1) * (nr_unique_pairs+1), frame + (T - 1) * nr_unique_pairs)

    ani = FuncAnimation(fig, update, frames=range(T), interval=1)
    ani.save(os.path.join(folder_name, 'RDF.gif'), writer='pillow')
    plt.savefig(os.path.join(folder_name, 'last_frame.png'))
    plt.close()
