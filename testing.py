# test_math_utils.py
import unittest
import numpy as np

from auxiliary import diff_and_dist_arr, find_bond_indices
from parse_functions import itp_to_data
from forces import bond_force, ang_force, net_force, LJ_force, dihedral_force
from integrators import integrator, euler, verlet, beeman, velo_verlet
from initial_config_generation import random_molecule

# Constants
eps = 1e-12
nr_tests = 15  # Number of random samples for validation

class TestMathUtils(unittest.TestCase):
    # ------------------------------- auxiliary -------------------------------
    def test_dist(self):
        """Test the distance calculation function."""
        X = np.array([[1., 0., 0.], [0., 1., 0.]])
        i_LJ = np.array([0])
        j_LJ = np.array([1])
       
        diff, dist = diff_and_dist_arr(X,i_LJ,j_LJ,0.)
        difference = np.array([[ 1, -1,  0]])
        distance = np.array([2 ** 0.5])
 
        self.assertTrue((np.abs(diff-difference) < eps).all())
        self.assertTrue((np.abs(dist-distance) < eps).all())
        
        X = np.array([[0., 0., 0.], [0., 1., 0.]])
        
        diff, dist = diff_and_dist_arr(X,i_LJ,j_LJ,0.)
        difference = np.array([[ 0, -1,  0]])
        distance = np.array([1])
        self.assertTrue((np.abs(diff-difference) < eps).all())
        self.assertTrue((np.abs(dist-distance) < eps).all())
                
    #------------------------------- forces -------------------------------
    def test_bond(self):
        """Test bond forces for different scenarios."""
        X = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        bonds = np.array([[0,1]])
        i_LJ = bonds[:,0]
        j_LJ = bonds[:,1]
        
        diff, D = diff_and_dist_arr(X,i_LJ,j_LJ,0.)

        # Test: H-H bondseparated by more than equilibrium distance
        kb=np.array([24531])
        r0=np.array([0.074])
        
        
        F1 = bond_force(diff, D, bonds, kb, r0, 2)

        self.assertTrue(F1[0, 0] > 0 and 
                        F1[1, 0] < 0 and 
                        F1[0, 0] == -F1[1, 0])

        # Test: O-O bond separated by less than equilibrium distance
        kb=np.array([49836])
        r0=np.array([0.12075])
        F2 = bond_force(diff, D, bonds, kb, r0, 2)
        self.assertTrue(F2[0, 0] < 0 and 
                        F2[1, 0] > 0 and 
                        F2[0, 0] == -F2[1, 0])

        # Test: C-O bond at equilibrium distance
        kb=np.array([107650])
        r0=np.array([0.11282])
        X = np.array([[0.11282, 0.0, 0.0], [0.0, 0.0, 0.0]])
        diff, D = diff_and_dist_arr(X,i_LJ,j_LJ,0.)
        F3 = bond_force(diff, D, bonds, kb, r0, 2)

        self.assertTrue((np.abs(F3) < eps).all())

    def test_ang(self):
        """Test angular forces for various scenarios."""
        O_pos = np.array([0, 0, 0])
        H1_pos = np.array([1, 0, 0])

        ktheta=np.array([628.02])
        theta0=np.array([104.52 / 360 * 2 * np.pi])
        
        # Test: Equilibrium angle
        H2_pos = np.array([np.cos(theta0[0]), np.sin(theta0[0]), 0])
        X = np.array([O_pos, H1_pos, H2_pos])

        bonds = np.array([[0,1],[0,2]])
        i_LJ = bonds[:,0]
        j_LJ = bonds[:,1]
        diff_bonds, D_bonds = diff_and_dist_arr(X,i_LJ,j_LJ,0.)
        
        angles = np.array([[1,0,2]])
        
        angle_mapping = find_bond_indices(angles,bonds)
        F = ang_force(diff_bonds, D_bonds, angle_mapping, angles, ktheta, 
                      theta0, 3)

        self.assertTrue((np.abs(F) < eps).all())
        
        
        # Test: Angle larger than equilibrium
        H2_pos = np.array([np.cos(theta0[0]+0.1), np.sin(theta0[0]+0.1), 0])
        X = np.array([O_pos, H1_pos, H2_pos])
        diff_bonds, D_bonds = diff_and_dist_arr(X,i_LJ,j_LJ,0.)

        F = ang_force(diff_bonds, D_bonds, angle_mapping, angles, ktheta, 
                      theta0, 3)

        # Expected directions
        F_target = np.array([ [-1, -1, 0],[0, 1, 0], [1, 1, 0]])  
        self.assertTrue((np.abs(np.sign(F) - F_target) < eps).all())

    

        # Test: Angle larger than equilibrium
        H2_pos = np.array([np.cos(theta0[0]-0.1), np.sin(theta0[0]-0.1), 0.])
        X = np.array([O_pos, H1_pos, H2_pos])
        diff_bonds, D_bonds = diff_and_dist_arr(X,i_LJ,j_LJ,0)

        F = ang_force(diff_bonds, D_bonds, angle_mapping, angles, ktheta, 
                      theta0, 3)
        # Expected directions
        F_target = np.array([ [1, 1, 0],[0, -1, 0],[-1, -1, 0]])  
        self.assertTrue((np.abs(np.sign(F) - F_target) < eps).all())
    
    def test_LJ(self):
        X = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        i_LJ=np.array([0])
        j_LJ=np.array([1])
        diff_LJ, D_LJ = diff_and_dist_arr(X,i_LJ,j_LJ,0.)
        
        epsilon = np.array([0.66386,0.66386])
        sigma = np.array([0.315061,0.315061])
        sigma_LJ = 0.5*(sigma[:,None]+sigma[None,:])
        epsilon_LJ = np.sqrt(epsilon[:,None]+epsilon[None,:])
        
        sigma_LJ,epsilon_LJ = sigma_LJ[i_LJ,j_LJ],epsilon_LJ[i_LJ,j_LJ]
              
        r_cutoff_LJ = np.array([np.inf])
        
        F=LJ_force(diff_LJ, D_LJ, epsilon_LJ, sigma_LJ, 
                   i_LJ, j_LJ, 2, r_cutoff_LJ)
        
        self.assertTrue(np.array_equal(np.sign(F),np.array([[-1,0,0],
                                                            [1,0,0]])))
        
        X = np.array([[0.0, 0.0, 0.0], [100, 0.0, 0.0]])
        diff_LJ, D_LJ = diff_and_dist_arr(X,i_LJ,j_LJ,0.)
        F=LJ_force(diff_LJ, D_LJ, epsilon_LJ, sigma_LJ, 
                   i_LJ, j_LJ, 2, r_cutoff_LJ)
        
        self.assertTrue(np.array_equal(np.sign(F),np.array([[1,0,0],
                                                            [-1,0,0]])))
        
    def test_dihedral(self):      
        dihedrals=np.array([[0,1,2,3]])
        C1 = np.array([-0.44310])
        C2 = np.array([3.83255])
        C3 = np.array([0.72801])
        C4 = np.array([-4.11705])
        
        X = np.array([[ 1.2699,      0,      0.1772], 
                      [-0.2033,      0,      0.5345],
                      [-0.9394,      0,     -0.6674],
                      [-1.8540,      0,     -0.4252]])
        
        bonds = np.array([[0,1],[1,2],[2,3]])
        dihedral_mapping = find_bond_indices(dihedrals,bonds)
        diff_bonds, D_bonds = diff_and_dist_arr(X,bonds[:,0],bonds[:,1],0.)
        
        
        F = dihedral_force(diff_bonds, D_bonds, dihedral_mapping, dihedrals, 
                           C1, C2, C3, C4, 4)
        self.assertTrue(np.array_equal(np.sign(F),np.array([[0,0,0],
                                                            [0,0,0],
                                                            [0,0,0],
                                                            [0,0,0]])))
        
    def test_net(self):
        """Test the computation of net forces."""
        X = 0.1*np.array([[1.93617934, 2.31884508, 1.72261570],
                          [1.78931374, 3.24075634, 1.51114298],
                          [2.30448689, 1.98045541, 0.90160232]])
        F_example = np.array([[16.0,   13.66, -55.08],
                              [-10.08,  4.37,  25.81],
                              [-5.92, -18.04,  29.28]])

        bonds, kb, r0, angles, ktheta, theta0, epsilon, sigma, dihedrals, \
            C1, C2, C3, C4, _ = itp_to_data("molecule_data\\Water.itp")
        
        diff_bonds, D_bonds = diff_and_dist_arr(X,bonds[:,0],bonds[:,1],0.)
        diff_LJ, D_LJ = np.array([]), np.array([])
        epsilon_LJ = np.array([])
        sigma_LJ = np.array([])
        i_LJ, j_LJ = np.array([]), np.array([])
        
        dihedral_mapping = find_bond_indices(dihedrals,bonds)
        angle_mapping = find_bond_indices(angles,bonds)
        
        F_computed = net_force(diff_bonds, D_bonds, diff_LJ, D_LJ, bonds, 3, 
                               angle_mapping, dihedral_mapping, kb, r0, angles, 
                               ktheta, theta0, epsilon_LJ, sigma_LJ, dihedrals, 
                               C1, C2, C3, C4, i_LJ, j_LJ)
        
        F_rounded = np.round(0.1*F_computed, 2)
        self.assertTrue((F_example - F_rounded < eps).all())
        
    # ------------------------------- integrators -------------------------------
    def test_integrators(self):
        """Test the three integrators."""
        # Test: (nonsense) Molecules with zero velocity remain stationary
        N=10
        for method in euler, verlet, beeman, velo_verlet:
            for _ in range(nr_tests):
                (X, atoms, bonds, kb, r0, angles, ktheta, theta0, 
                 epsilon_LJ, sigma_LJ, r_cutoff_LJ, i_LJ, j_LJ, 
                 dihedrals, C1, C2, C3, C4, masses) = random_molecule(N)
                
                V = np.zeros((N, 3),dtype= 'f')
                A = np.zeros((N, 3),dtype= 'f')
                
                dihedral_mapping = find_bond_indices(dihedrals,bonds)
                angle_mapping = find_bond_indices(angles,bonds)
                
                force_parameters = (angle_mapping, dihedral_mapping, kb, r0, 
                                    angles, ktheta, theta0, epsilon_LJ, 
                                    sigma_LJ, dihedrals, C1, C2, C3, C4, i_LJ, 
                                    j_LJ, r_cutoff_LJ,None)
                X_new,_,_,_,_,_,_,_,_ = integrator(method, X, X, V, A, A, 
                                                   force_parameters, bonds, N, 
                                                   1, masses, 0., _, False, 
                                                   i_LJ, j_LJ, False)
                
                

                self.assertTrue(((X - X_new) < eps).all())

        #Test: X is updated in the direction of V (not applicable for Verlet)
        for method in euler, beeman, velo_verlet:
            for _ in range(nr_tests):
                (X, atoms, bonds, kb, r0, angles, ktheta, theta0, 
                 epsilon_LJ, sigma_LJ, r_cutoff_LJ, i_LJ, j_LJ, 
                 dihedrals, C1, C2, C3, C4, masses) = random_molecule(N)
                
                A = np.zeros((N, 3),dtype= 'f')
                V = (2 * np.random.rand(N, 3) - 1).astype(float)
                
                dihedral_mapping = find_bond_indices(dihedrals,bonds)
                angle_mapping = find_bond_indices(angles,bonds)
                
                force_parameters = (angle_mapping, dihedral_mapping, kb, r0, 
                                    angles, ktheta, theta0, epsilon_LJ, 
                                    sigma_LJ, dihedrals, C1, C2, C3, C4, i_LJ, 
                                    j_LJ, r_cutoff_LJ,None)
                
                X_new,_,_,_,_,_,_,_,_ = integrator(method, X, X, V, A, A, 
                                                   force_parameters, bonds, N, 
                                                   1, masses, 0., _, False, 
                                                   i_LJ, j_LJ, False)

                self.assertTrue(
                    (np.abs(np.sign(X_new - X) - np.sign(V)) < eps).all()
                    )

# ------------------------------- Main Script -------------------------------
if __name__ == "__main__":
    unittest.main()
