# Import necessary libraries and modules
from auxiliary import (scale_velocity, updt, project_to_box, cutoff_calculator, 
                      diff_and_dist_arr, find_bond_indices, compute_LJ_indices)
from forces import net_force
from energy import (bond_energy, kinetic_energy, LJ_energy, 
                      dihedral_energy, angular_energy)

from initial_config_generation import initializeVelocity
from parse_functions import array_to_xyz, data_to_itp
from visualizations import (energy_plot, temperature_plot,
                           visualize_xyz_with_vmd, initialize_RDF, 
                           RDF_visualize, initialize_ps_diag, 
                           show_ps_diag, computation_piechart)

import numpy as np
import os
import time
# --------------------------- Integrator Functions ----------------------------
euler = {
    'positions': lambda x_im1, x_im2, v_im1, a_im1, a_im2, dt: 
                    x_im1 + dt * v_im1 + (dt**2) / 2 * a_im1,
    'velocities': lambda x_i, x_im1, x_im2, v_im1, a_i, a_im1, a_im2, dt: 
                    v_im1 + dt * a_im1
}
    
verlet = {
    'positions': lambda x_im1, x_im2, v_im1, a_im1, a_im2, dt: 
        2 * x_im1 - x_im2 + (dt**2) * a_im1,
    'velocities': lambda x_i, x_im1, x_im2, v_im1, a_i, a_im1, a_im2, dt: 
        1 / (2 * dt) * (x_i - x_im2)  
}
    
velo_verlet = {
    'positions': lambda x_im1, x_im2, v_im1, a_im1, a_im2, dt: 
        x_im1 + dt * v_im1 + (dt**2) / 2 * a_im1,
    'velocities': lambda x_i, x_im1, x_im2, v_im1, a_i, a_im1, a_im2, dt: 
        v_im1 + dt / 2 * (a_im1 + a_i)  
}

beeman = {
    'positions': lambda x_im1, x_im2, v_im1, a_im1, a_im2, dt: 
        x_im1 + dt * v_im1 + (4*a_im1-a_im2)*(1/6)*dt**2,
    'velocities': lambda x_i, x_im1, x_im2, v_im1, a_i, a_im1, a_im2, dt: 
        v_im1 + (2*a_i+5*a_im1-a_im2)*dt/6
}
  
def integrator(method, x_im1, x_im2, v_im1, a_im1, a_im2, force_parameters,
               bonds, nr_atoms, dt, masses, boxsize, temp, temp_control, i_LJ, 
               j_LJ, comptime_vis):
    """
    Updates x, v and a using update functions defined above. If temp_control 
    is true, velocities will be rescaled to achieve a temperature temp. If 
    comptime_vis is True, time measurements will be made. 
    """    
    # Compute new positions
    x_i = method['positions'](x_im1, x_im2, v_im1, a_im1, a_im2, dt)
    
    if comptime_vis:
        t0 = time.perf_counter()

    # Compute distances for bonds and LJ pairs      
    diff_bonds, D_bonds = diff_and_dist_arr(x_i, bonds[:,0], bonds[:,1],0.)
    diff_LJ, D_LJ = diff_and_dist_arr(x_i, i_LJ, j_LJ, boxsize)
    
    if comptime_vis:
        t1 = time.perf_counter()
        
    # Compute forces
    F_i = net_force(diff_bonds, D_bonds, diff_LJ, D_LJ, bonds, nr_atoms, 
                    *force_parameters)
    
    
    if comptime_vis:
        t2 = time.perf_counter()
        diff_dist_time = t1 - t0
        force_time = t2 - t1        
    else:
        force_time = 0
        diff_dist_time = 0
        
        
    # Compute acceleration
    a_i = F_i / masses[:, None]
    
    # Compute new velocities
    v_i = method['velocities'](x_i, x_im1, x_im2, v_im1, a_i, a_im1, a_im2, dt)
    
    # Optional: Temperature scaling
    if temp_control:
        v_i = scale_velocity(v_i, masses, temp)


    return (x_i, v_i, a_i, diff_bonds, D_bonds, 
            diff_LJ, D_LJ, diff_dist_time, force_time)

# -------------------------- Main Integration Function ------------------------  

def integrate(initial_x, bonds, atoms, kb, r0, angles, ktheta, theta0, epsilon, 
              sigma, dihedrals, C1, C2, C3, C4, masses, method, dt, endTime, 
              save_steps, boxsize, folder_name, molec_index, molec_names, 
              r_cutoff_scalar, f_threshold, temp, temp_control = False, 
              bonds_of_interest = None, pairs_of_interest = None, RDF_thin=1.0, 
              RDF_dr=0.05, VMD_vis=True, energy_graph=True, temp_graph=True,
              phase_space_diagram=True, RDF_vis=True, comptime_vis=True, 
              r_threshold_scalar=None):
    """
    Integrates the initial state given by the input parameters by iteratively
    applying the integrator function above. Optionally outputs graphs of
    measurements if the corresponding boolean parameter is set to True.
    """    
    # Computing simulation parameters
    print("Computing simulation parameters...")
    nr_atoms, _ = initial_x.shape
    print("  Computing LJ indices...")
    # Determine pairs that experience LJ force
    i_LJ, j_LJ = compute_LJ_indices(nr_atoms, dihedrals, angles, 
                                    epsilon, sigma, molec_index)
    print("  Computing epsilon and sigma...")
    
    #Precompute epsilon, sigma and r_cutoff
    unique_sig, inverse_idx_sig = np.unique(sigma, return_inverse=True)
    sigma_LJ = 0.5*(unique_sig[inverse_idx_sig[i_LJ]] + 
                    unique_sig[inverse_idx_sig[j_LJ]])
    
    unique_eps, inverse_idx_eps = np.unique(epsilon, return_inverse=True)
    epsilon_LJ = np.sqrt(unique_eps[inverse_idx_eps[i_LJ]] * 
                         unique_eps[inverse_idx_eps[j_LJ]])
    
    print("  Computing r_cutoff...")
    r_cutoff_LJ = cutoff_calculator(r_cutoff_scalar, f_threshold, 
                                      epsilon_LJ, sigma_LJ, boxsize)   
    
    # Map angle and dihedral data to the indices of corresponding bonds
    angle_mapping = find_bond_indices(angles,bonds)
    dihedral_mapping = find_bond_indices(dihedrals,bonds)
    
    # Computing initial configuration
    if r_threshold_scalar is not None:
        print("  Computing equilibrium configuration...")
        r_threshold = r_threshold_scalar*sigma_LJ
        # Pack the constant parameters for the integration process
        force_parameters = (angle_mapping, dihedral_mapping, kb, r0, angles, 
                            ktheta, theta0, epsilon_LJ, sigma_LJ, dihedrals, C1, 
                            C2, C3, C4, i_LJ, j_LJ, r_cutoff_LJ,r_threshold)
        
        integration_parameters = (bonds, nr_atoms, dt, masses, boxsize, 
                                  temp, False, i_LJ, j_LJ, False)
        
        diff_bonds, D_bonds = diff_and_dist_arr(initial_x, bonds[:,0], bonds[:,1], 0.)
        diff_LJ, D_LJ = diff_and_dist_arr(initial_x, i_LJ, j_LJ, boxsize)   
        
        F= net_force(diff_bonds, D_bonds, diff_LJ, D_LJ, bonds, nr_atoms, 
                       *force_parameters)
    
        Aim2 = F / masses[:, None]
        
        (Xim1, Vim1, Aim1, diff_bonds, D_bonds, 
        diff_LJ, D_LJ, diff_dist_time, force_time) = integrator(euler,
            initial_x, _, 0, Aim2, _, force_parameters, *integration_parameters
        )
                                                                
        Xim2 = initial_x
        
        # Perform the rest of the integration with the chosen method
        start = np.sum(D_LJ>r_threshold)
        total = np.sum(D_LJ<r_threshold)
        
        smallest_dist = np.min(D_LJ/sigma_LJ)
        while np.any(D_LJ<r_threshold):
            X_temp, A_temp = Xim1, Aim1
            (Xim1, Vim1, Aim1, diff_bonds, 
             D_bonds, diff_LJ, D_LJ, 
             diff_dist_time, force_time) = integrator(method, Xim1, Xim2, Vim1, 
                                                      Aim1, Aim2, force_parameters,
                                                      *integration_parameters)
            Xim2, Aim2 = X_temp, A_temp
            updt(total, np.sum(D_LJ>r_threshold) - start)
        print()
        print(f"  Equilibrium configuration initialized. Smallest distance"
              f"  increased from {smallest_dist:.3f} to {np.min(D_LJ/sigma_LJ):.3f}...")
            
        initial_x = Xim2
        
    print("Saving initial configuration data to files...")

    subdirectory = f"dt={dt}_T={endTime}_steps={save_steps}_"
    
    
    # Define folder_name based on input choice
    if f_threshold:
        subdirectory += f"fthr={f_threshold}_"
    else:
        subdirectory += f"rcut-sclr={r_cutoff_scalar}_"
    
    subdirectory += f"temp={temp}_control={temp_control}"
    
    folder_name = os.path.join(folder_name, subdirectory)
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    # Save initial itp to files
    data_to_itp(
        os.path.join(folder_name,"INITIAL.itp"), bonds, kb, r0, angles, ktheta, theta0, 
        epsilon, sigma, dihedrals, C1, C2, C3, C4, masses
    )
    array_to_xyz(
        project_to_box(initial_x, molec_index, boxsize), atoms, 
        os.path.join(folder_name,"INITIAL.xyz"), 0, 'write'
    )
    
    # Pack the constant parameters for the integration process
    force_parameters = (angle_mapping, dihedral_mapping, kb, r0, angles, 
                        ktheta, theta0, epsilon_LJ, sigma_LJ, dihedrals, C1, 
                        C2, C3, C4, i_LJ, j_LJ, r_cutoff_LJ,None)
    
    integration_parameters = (bonds, nr_atoms, dt, masses, boxsize, 
                              temp, temp_control, i_LJ, j_LJ, comptime_vis)   
        
    
    print("Initializing data arrays...")
    timesteps = int(np.ceil(endTime / dt))
    total_saves = total_saves = 1 + (timesteps) // save_steps
    
    if RDF_vis:
        print("  Initializing RDF...")
        (RDF_LJ_indices, RDF_bond_indices, 
         RDF_leftover_indices, bin_edges, bin_volumes, 
         unique_pairs, RDF_MATRIX) = initialize_RDF(total_saves, molec_names, 
                                                    i_LJ, j_LJ, bonds, atoms, 
                                                    pairs_of_interest, boxsize, 
                                                    RDF_thin, RDF_dr)
    if energy_graph:
        Bond_E = np.empty(total_saves)
        Angular_E = np.empty(total_saves)
        LJ_E = np.empty(total_saves)
        Dihedral_E = np.empty(total_saves)
        Kinetic_E = np.empty(total_saves)
    if temp_graph:
        Kinetic_E = np.empty(total_saves)
    if phase_space_diagram:  
        print("  Initializing phase space...")
        (ps_bonds, mask_per_bond, 
         ps_arrays) = initialize_ps_diag(total_saves, molec_names, atoms, 
                                                bonds, bonds_of_interest)
    if comptime_vis:
        computation_times = np.zeros((total_saves-1,7))
           
    def save_visualization_data(save_idx, X, diff_bonds, D_bonds, diff_LJ,D_LJ,
                                V, save_mode, i, diff_dist_time, force_time):
        """
        Helper function to save data for later visualization.
        """
        t0 = time.perf_counter()
        if RDF_vis:
            # Find LJ_dists and bond_dists using indices
            LJ_dists = [D_LJ[idx] for idx in RDF_LJ_indices]  
            bond_dists = [D_bonds[idx] for idx in RDF_bond_indices]
            
            # Compute distances not computed by simulation
            _, other_dists = zip(*[
                diff_and_dist_arr(X, RDF_leftover_indices[i, 0], 
                                  RDF_leftover_indices[i, 1], boxsize) 
                    for i in range(RDF_leftover_indices.shape[0])])
            
            # Save histogram data
            for i in range(RDF_leftover_indices.shape[0]):
                D = np.hstack((LJ_dists[i], bond_dists[i], other_dists[i]))
                RDF_MATRIX[save_idx, i] = np.histogram(D, bins=bin_edges)[0]
                
        t1 = time.perf_counter()
        
        if energy_graph:
            Bond_E[save_idx] = bond_energy(D_bonds,kb,r0)
            Angular_E[save_idx] = angular_energy(diff_bonds, D_bonds, 
                                                 angle_mapping, ktheta, theta0)
            LJ_E[save_idx] = LJ_energy(D_LJ, epsilon_LJ, sigma_LJ, 
                                       r_cutoff_LJ)
            Dihedral_E[save_idx] = dihedral_energy(diff_bonds, D_bonds, 
                                                   dihedral_mapping, C1, C2, 
                                                   C3, C4)
            
        t2 = time.perf_counter()
        
        if temp_graph or energy_graph: #the kinetic energy is stored in
                                            #energy matrix, but in case it's
                                            #not, we compute it here.
            Kinetic_E[save_idx] = kinetic_energy(masses,V)
        
        t3 = time.perf_counter()
        
        # Always save the data to an xyz file, after projecting it to be in 
        # the box.
        array_to_xyz(project_to_box(X, molec_index, boxsize), atoms, 
                     os.path.join(folder_name,"INTEGRATED.xyz"), i*dt, save_mode)
        
        t4 = time.perf_counter()
        
        if phase_space_diagram:            
            for i,mask in enumerate(mask_per_bond):
                ps_arrays[i] [save_idx] = D_bonds[mask]               
        t5 = time.perf_counter()
        # After measuring all times above, save them in the computation_times
        # array to plot at the end (togehter with times of dist and force.)
        if comptime_vis and diff_dist_time and force_time:          
            computation_times[save_idx-1,0] = diff_dist_time
            computation_times[save_idx-1,1] = force_time
            computation_times[save_idx-1,2] = t4-t3
            computation_times[save_idx-1,3] = t1-t0
            computation_times[save_idx-1,4] = t2-t1
            computation_times[save_idx-1,5] = t3-t2
            computation_times[save_idx-1,6] = t5-t4
            
    # Integration starts here
    print("Starting integration...")
    
    start_integration_initialization = time.perf_counter()
    # Compute the initial variables
    Xim2 = initial_x
    Vim2 = initializeVelocity(masses, temp)
    
    save_idx = 0
    updt(timesteps, 0)
        
    diff_bonds, D_bonds = diff_and_dist_arr(Xim2, bonds[:,0], bonds[:,1], 0.)
    diff_LJ, D_LJ = diff_and_dist_arr(Xim2, i_LJ, j_LJ, boxsize)   
    
    F= net_force(diff_bonds, D_bonds, diff_LJ, D_LJ, bonds, nr_atoms, 
                   *force_parameters)

    Aim2 = F / masses[:, None]
    
    save_visualization_data(save_idx, initial_x, diff_bonds, D_bonds, diff_LJ, 
                            D_LJ, Vim2, 'write', 0, None, None)  
    
    # After computing the initial variables, we perform the initial Euler step.
    save_idx+=1
    
    start_integration = time.perf_counter()
    
    (Xim1, Vim1, Aim1, diff_bonds, D_bonds, 
    diff_LJ, D_LJ, diff_dist_time, force_time) = integrator(euler,
        Xim2, _, Vim2, Aim2, _, force_parameters, *integration_parameters
    )
          

    
    # Save first step if required
    if save_steps == 1:        
        save_visualization_data(save_idx, Xim1, diff_bonds, D_bonds, diff_LJ, 
                                D_LJ, Vim1, 'append', 1, diff_dist_time, 
                                force_time)
        
        save_idx += 1

    updt(timesteps, 1)
    
    # Perform the rest of the integration with the chosen method
    for i in range(2, timesteps+1):
        # save current X and A values in temporary values to later copy
        # to Xim2 and Aim2.
        X_temp, A_temp = Xim1, Aim1
        (Xim1, Vim1, Aim1, diff_bonds, 
         D_bonds, diff_LJ, D_LJ, 
         diff_dist_time, force_time) = integrator(method, Xim1, Xim2, Vim1, 
                                                  Aim1, Aim2, force_parameters,
                                                  *integration_parameters)
        Xim2, Aim2 = X_temp, A_temp
        
        # Save data every save_steps steps
        if i % save_steps == 0:            
            save_visualization_data(save_idx, Xim1, diff_bonds, D_bonds, 
                                    diff_LJ, D_LJ, Vim1, 'append', i, 
                                    diff_dist_time, force_time)
            save_idx += 1
        
        updt(timesteps, i)  # Show updates with the updt function
        
    final_time = time.perf_counter()    
    print()
    print(f"  Integration took "
          f"{(final_time-start_integration_initialization):.4f} seconds")

    if RDF_vis:
        RDF_path = os.path.join(folder_name,"RDF")
        if not os.path.exists(RDF_path):
            os.makedirs(RDF_path)
        print("Generating RDF figures.")
        RDF_visualize(RDF_MATRIX, bin_edges, bin_volumes, unique_pairs, 
                     dt*save_steps, RDF_path, r0)
        print()
        print("Finished RDF figures.")
    if energy_graph:
        energies = np.array([Bond_E,Angular_E,LJ_E,Dihedral_E,Kinetic_E])
        energy_plot(energies.T,endTime, folder_name)
    if temp_graph:
        temperature_plot(Kinetic_E, 3*nr_atoms, endTime, folder_name)
    if phase_space_diagram:
        ps_path = os.path.join(folder_name ,"ps_diagram")
        if not os.path.exists(ps_path):
            os.makedirs(ps_path)
        show_ps_diag(ps_bonds, ps_arrays, dt*save_steps, ps_path)
    if comptime_vis:
        computation_piechart(computation_times, save_steps, 
                             final_time - start_integration, folder_name)
    if VMD_vis:
        print("Opening VMD")
        d_max = 1.05*np.max(D_bonds) # Used for showing dynamic bonds in VMD.
        visualize_xyz_with_vmd(folder_name+"\\INTEGRATED.xyz",
                               10*boxsize,10*d_max)